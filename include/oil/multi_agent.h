#pragma once
#include "oil/tensor.h"
#include "oil/model.h"
#include "oil/trainer.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <memory>
#include <cstdint>
#include <mutex>
#include <atomic>

namespace oil {
namespace multi_agent {

enum class AgentRole : uint8_t {
    SUPERVISOR = 0,
    PLANNER    = 1,
    CODER      = 2,
    REVIEWER   = 3,
    TESTER     = 4,
    OPTIMIZER  = 5,
    CUSTOM     = 6
};

enum class CommTopology : uint8_t {
    STAR = 0,
    RING = 1,
    MESH = 2,
    HIERARCHICAL = 3
};

enum class MessageType : uint8_t {
    TASK_ASSIGN     = 0,
    TASK_RESULT     = 1,
    VOTE_REQUEST    = 2,
    VOTE_CAST       = 3,
    FEEDBACK        = 4,
    STATUS_UPDATE   = 5,
    ERROR_REPORT    = 6,
    CONFLICT_ALERT  = 7,
    PLAN_BROADCAST  = 8,
    SUBTASK_DECOMP  = 9
};

struct AgentMessage {
    int64_t id = 0;
    int from_agent = -1;
    int to_agent = -1;
    MessageType type = MessageType::STATUS_UPDATE;
    std::string content;
    int64_t timestamp = 0;
    float priority = 0.5f;
    std::vector<int> references;
};

struct Subtask {
    int64_t id = 0;
    std::string description;
    std::string status;
    int assigned_agent = -1;
    float progress = 0.0f;
    std::vector<int64_t> dependencies;
    std::string result;
    int priority = 0;
    double start_time = 0.0;
    double end_time = 0.0;
};

struct AgentMetrics {
    int tasks_completed = 0;
    int tasks_failed = 0;
    float avg_quality = 0.0f;
    double total_time_ms = 0.0;
    int messages_sent = 0;
    int messages_received = 0;
    int votes_cast = 0;
    float consensus_score = 0.0f;
};

struct AgentState {
    int agent_id = -1;
    AgentRole role = AgentRole::CUSTOM;
    std::string name;
    std::vector<std::string> context_window;
    std::vector<std::string> history;
    std::string status;
    AgentMetrics metrics;
    bool active = true;
    int64_t tasks_assigned = 0;
};

struct BlackboardEntry {
    std::string key;
    std::string value;
    int author_agent = -1;
    int64_t timestamp = 0;
    int64_t ttl = -1;
};

struct Vote {
    int voter_agent = -1;
    std::string proposal;
    bool approve = false;
    float confidence = 0.0f;
    std::string rationale;
};

struct HierarchicalPlan {
    std::string goal;
    std::vector<std::string> high_level_steps;
    std::vector<Subtask> detailed_subtasks;
    bool completed = false;
    float overall_progress = 0.0f;
};

class MultiAgentSystem {
public:
    MultiAgentSystem(int n_agents, CommTopology topology = CommTopology::STAR);
    ~MultiAgentSystem();

    int add_agent(AgentRole role, const std::string& name = "");
    void remove_agent(int agent_id);
    AgentState& get_agent(int agent_id);
    const AgentState& get_agent(int agent_id) const;
    int agent_count() const;

    void send_message(const AgentMessage& msg);
    std::vector<AgentMessage> receive_messages(int agent_id);
    std::vector<AgentMessage> broadcast(int from_agent, MessageType type, const std::string& content);
    bool communicate_direct(int from, int to, const std::string& content);

    void assign_task(int agent_id, const std::string& task);
    void assign_task_to_role(AgentRole role, const std::string& task);
    std::string get_task_status(int64_t task_id) const;

    void decompose_task(const std::string& complex_task, int max_subtasks);
    const std::vector<Subtask>& get_subtasks() const;
    void assign_subtask(int64_t subtask_id, int agent_id);
    std::vector<Subtask> get_ready_subtasks() const;

    void submit_vote(int agent_id, const std::string& proposal, bool approve, float confidence, const std::string& rationale);
    bool check_consensus(float threshold = 0.6f);
    std::vector<Vote> get_votes() const;
    void reset_votes();

    void write_blackboard(const std::string& key, const std::string& value, int agent_id);
    std::string read_blackboard(const std::string& key) const;
    std::vector<BlackboardEntry> search_blackboard(const std::string& prefix) const;
    void cleanup_blackboard();

    HierarchicalPlan create_plan(const std::string& goal, int max_subtasks);
    bool execute_plan(HierarchicalPlan& plan);
    float get_plan_progress(const HierarchicalPlan& plan) const;

    void supervisor_monitor();
    void supervisor_resolve_conflicts();
    void supervisor_reassign_failed(int subtask_id);
    void supervisor_escalate(const std::string& issue);

    void update_metrics(int agent_id);
    AgentMetrics get_metrics(int agent_id) const;
    std::vector<AgentMetrics> get_all_metrics() const;
    std::string metrics_summary() const;

    void run_round(int max_rounds = 10);
    void run_episode(int steps);

    void configure(CommTopology topology);
    void set_max_queue_size(int64_t max_size);
    void set_communication_delay(int64_t delay_ms);

    std::vector<std::string> get_histories() const;

private:
    std::vector<AgentState> agents_;
    std::vector<AgentMessage> message_queue_;
    std::vector<Subtask> subtasks_;
    std::vector<Vote> votes_;
    std::vector<BlackboardEntry> blackboard_;
    HierarchicalPlan current_plan_;
    CommTopology topology_;
    int64_t next_message_id_ = 0;
    int64_t next_subtask_id_ = 0;
    int64_t max_queue_size_ = 10000;
    int64_t communication_delay_ms_ = 0;

    void route_message(const AgentMessage& msg);
    bool is_connected(int from, int to) const;
    bool check_dependency_satisfied(const Subtask& st) const;
    void assign_subtask_by_role(AgentRole role, int64_t subtask_id);
    int find_best_agent_for_task(const std::string& task_description) const;
    float compute_task_agent_affinity(const std::string& task, const AgentState& agent) const;
    void update_agent_context(int agent_id, const std::string& entry);
    AgentRole string_to_role(const std::string& role_str) const;
    std::string role_to_string(AgentRole role) const;
};

} // namespace multi_agent
} // namespace oil
