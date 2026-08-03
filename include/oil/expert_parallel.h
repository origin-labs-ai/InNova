#pragma once
#include "oil/tensor.h"
#include "oil/moe_variants.h"
#include <vector>
#include <cstdint>
#include <string>
#include <functional>
#include <memory>
#include <mutex>
#include <atomic>

namespace oil {
namespace expert {

struct ClusterNode {
    int64_t node_id = -1;
    std::string address;
    int64_t port = 0;
    bool alive = true;
    int64_t num_experts = 0;
    int64_t capacity_tokens = 0;
    double load_factor = 0.0;
    int64_t total_tokens_processed = 0;
    int64_t total_bytes_sent = 0;
    int64_t total_bytes_recv = 0;
};

struct ExpertAssignment {
    int64_t expert_id = -1;
    int64_t node_id = -1;
    int64_t token_count = 0;
    int64_t capacity = 0;
    float avg_weight = 0.0f;
};

struct RoutingDecision {
    Tensor expert_indices;
    Tensor expert_weights;
    std::vector<int64_t> tokens_per_node;
    std::vector<int64_t> tokens_per_expert;
    float load_balance_loss = 0.0f;
    float z_loss = 0.0f;
    int64_t tokens_dropped = 0;
    int64_t overflow_tokens = 0;
};

struct AllToAllPlan {
    struct Transfer {
        int64_t src_node;
        int64_t dst_node;
        int64_t expert_id;
        int64_t token_count;
        int64_t data_offset;
        int64_t data_bytes;
    };
    std::vector<Transfer> transfers;
    int64_t total_bytes = 0;
    int64_t max_peers = 0;
};

struct ClusterConfig {
    int64_t hidden_size = 4096;
    int64_t num_experts = 64;
    int64_t top_k = 2;
    int64_t expert_capacity_factor = 2;
    float load_balance_coef = 0.01f;
    float z_loss_coef = 0.001f;
    float capacity_factor = 1.25f;
    int64_t max_tokens_per_expert = 0;
    bool use_async_comm = true;
    bool overlap_comm_compute = true;
    int64_t chunk_size = 1024;
    int64_t heartbeat_interval_ms = 1000;
    int64_t fault_timeout_ms = 5000;
    int64_t max_retries = 3;
};

struct ExpertParallelStats {
    int64_t total_tokens_routed = 0;
    int64_t total_tokens_dropped = 0;
    int64_t total_comm_bytes = 0;
    int64_t total_expert_forward_calls = 0;
    float avg_load_imbalance = 0.0f;
    float max_load_imbalance = 0.0f;
    int64_t node_failures_handled = 0;
    int64_t expert_redistributions = 0;
    double total_forward_time_ms = 0.0;
    double total_comm_time_ms = 0.0;
};

class TCPCommBackend {
public:
    TCPCommBackend();
    ~TCPCommBackend();

    bool listen(int64_t port);
    bool connect(const std::string& address, int64_t port);
    void disconnect(int64_t node_id);

    bool send(int64_t node_id, const void* data, int64_t bytes);
    bool recv(int64_t node_id, void* buffer, int64_t max_bytes, int64_t* actual_bytes);
    bool send_async(int64_t node_id, const void* data, int64_t bytes);
    bool recv_async(int64_t node_id, void* buffer, int64_t max_bytes, int64_t* actual_bytes);
    void wait_all();

    bool is_connected(int64_t node_id) const;
    int64_t bytes_sent(int64_t node_id) const;
    int64_t bytes_received(int64_t node_id) const;

    void shutdown();

private:
    struct Impl;
    Impl* impl_;
};

class ExpertDispatcher {
public:
    ExpertDispatcher(const ClusterConfig& cfg);
    ~ExpertDispatcher();

    RoutingDecision route_tokens(const Tensor& logits, int64_t num_tokens,
                                  int64_t local_node_id, int64_t num_nodes);
    AllToAllPlan plan_all_to_all(const RoutingDecision& routing,
                                  int64_t local_node_id);
    Tensor dispatch_tokens(const Tensor& tokens, const AllToAllPlan& plan,
                           int64_t local_node_id);
    Tensor gather_results(const Tensor& expert_outputs, const AllToAllPlan& plan,
                          int64_t local_node_id);
    void compute_load_balance(const RoutingDecision& routing, float& lb_loss, float& z_loss);

    void assign_experts_to_nodes(int64_t num_nodes);
    ExpertAssignment get_expert_assignment(int64_t expert_id) const;
    std::vector<ExpertAssignment> get_all_assignments() const;

    void redistribute_expert(int64_t expert_id, int64_t from_node, int64_t to_node);

    ClusterConfig config;

private:
    std::vector<ExpertAssignment> assignments_;
    mutable std::mutex mtx_;
};

class ExpertParallelManager {
public:
    ExpertParallelManager(const ClusterConfig& cfg = ClusterConfig());
    ~ExpertParallelManager();

    bool init_cluster(int64_t local_node_id, int64_t port);
    bool join_cluster(const std::string& coordinator_addr, int64_t coordinator_port);
    void leave_cluster();
    bool is_cluster_active() const;

    int64_t local_node_id() const;
    int64_t num_nodes() const;
    std::vector<ClusterNode> get_nodes() const;

    moe::MoEOutput forward_expert_parallel(const Tensor& x,
                                       std::vector<moe::ExpertFFN>& local_experts,
                                       bool training = true);

    Tensor expert_parallel_mlp(const Tensor& x,
                                std::vector<moe::ExpertFFN>& local_experts);

    void sync_gradients(std::vector<moe::ExpertFFN>& local_experts);
    void all_reduce_gradients(Tensor& grad);

    void add_node(const std::string& address, int64_t port);
    void remove_node(int64_t node_id);
    void handle_node_failure(int64_t node_id);

    void rebalance_experts();
    void redistribute_experts_after_failure(int64_t failed_node);

    bool checkpoint_experts(const std::vector<moe::ExpertFFN>& experts,
                            const std::string& path);
    bool restore_experts(std::vector<moe::ExpertFFN>& experts,
                         const std::string& path);

    ExpertParallelStats get_stats() const;
    void reset_stats();

    // Restore manager state previously saved by a checkpoint (used by
    // FaultToleranceManager::restore_state). The node list and routing stats
    // are written back verbatim so a checkpoint round-trips.
    void restore_nodes(const std::vector<ClusterNode>& nodes);
    void restore_stats(const ExpertParallelStats& stats);

    ClusterConfig config;

private:
    struct Impl;
    Impl* impl_;
};

struct PipelineStage {
    int64_t stage_id = -1;
    std::vector<int64_t> expert_ids;
    int64_t node_id = -1;
    bool ready = false;
    Tensor input_buffer;
    Tensor output_buffer;
};

class PipelineScheduler {
public:
    PipelineScheduler(int64_t num_stages, int64_t micro_batch_size);
    ~PipelineScheduler();

    void schedule_forward(const Tensor& input, int64_t stage_id);
    void schedule_backward(const Tensor& grad_output, int64_t stage_id);
    void execute_pipeline(std::vector<ExpertParallelManager*>& managers,
                          std::vector<std::vector<moe::ExpertFFN>*>& all_experts);

    PipelineStage get_stage(int64_t stage_id) const;
    bool all_stages_done() const;
    void reset();

    int64_t num_stages;
    int64_t micro_batch_size;

private:
    std::vector<PipelineStage> stages_;
    int64_t current_step_ = 0;
    int64_t total_steps_ = 0;
    mutable std::mutex mtx_;
};

class FaultToleranceManager {
public:
    FaultToleranceManager(ExpertParallelManager* manager);
    ~FaultToleranceManager();

    void start_heartbeat(int64_t interval_ms);
    void stop_heartbeat();
    bool check_node_health(int64_t node_id) const;
    void on_node_failure(int64_t node_id);
    void on_node_recovery(int64_t node_id);
    std::vector<int64_t> get_failed_nodes() const;

    void checkpoint_state(const std::string& path);
    void restore_state(const std::string& path);

private:
    ExpertParallelManager* manager_;
    std::vector<int64_t> failed_nodes_;
    mutable std::mutex mtx_;
    std::atomic<bool> running_{false};
};

} // namespace expert
} // namespace oil
