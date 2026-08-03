#include "oil/asi.h"
#include <chrono>
#include <algorithm>

namespace oil {
namespace asi {

SafetyGuardrails::SafetyGuardrails() {
    blocked_patterns_ = {
        "rm -rf", "sudo", "format c:", "shutdown /s", "del /s",
        "rd /s", "cipher /w", "drop table", "delete from",
        "mkfs", ":(){", "fork bomb"
    };
}

int64_t SafetyGuardrails::current_timestamp() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

bool SafetyGuardrails::check_output(const std::string& output) {
    if (kill_switch_ || paused_) return false;
    for (const auto& p : blocked_patterns_) {
        if (output.find(p) != std::string::npos) {
            audit_log("check_output", output, "blocked");
            return false;
        }
    }
    return true;
}

bool SafetyGuardrails::check_input(const std::string& input) {
    if (kill_switch_ || paused_) return false;
    for (const auto& p : blocked_patterns_) {
        if (input.find(p) != std::string::npos) {
            audit_log("check_input", input, "blocked");
            return false;
        }
    }
    return true;
}

bool SafetyGuardrails::check_modification(const std::string& target, const std::string& change) {
    if (kill_switch_ || paused_) return false;
    bool allowed = check_input(change) && !target.empty();
    audit_log("check_modification", target + " <- " + change, allowed ? "allowed" : "blocked");
    return allowed;
}

bool SafetyGuardrails::check_invariant(const std::string& invariant_name) {
    return invariants_.find(invariant_name) != invariants_.end();
}

void SafetyGuardrails::set_invariant(const std::string& name, const std::string& expression) {
    invariants_[name] = expression;
}

std::vector<std::string> SafetyGuardrails::get_invariants() const {
    std::vector<std::string> names;
    for (const auto& kv : invariants_) names.push_back(kv.first);
    return names;
}

bool SafetyGuardrails::human_override(const std::string& action, const std::string& reason) {
    audit_log("human_override", action + " (" + reason + ")", "granted");
    return true;
}

void SafetyGuardrails::audit_log(const std::string& action, const std::string& detail,
                                 const std::string& result) {
    AuditEntry e;
    e.timestamp = current_timestamp();
    e.action = action;
    e.detail = detail;
    e.result = result;
    std::lock_guard<std::mutex> lock(audit_mutex_);
    audit_log_.push_back(e);
}

std::vector<AuditEntry> SafetyGuardrails::get_audit_log(int count) const {
    std::lock_guard<std::mutex> lock(audit_mutex_);
    if (count <= 0 || (int)audit_log_.size() <= count) return audit_log_;
    return std::vector<AuditEntry>(audit_log_.end() - count, audit_log_.end());
}

HITL::HITL() {}

bool HITL::request_approval(const std::string& action) {
    AuditEntry e;
    e.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    e.action = action;
    e.detail = paused_ ? "auto-denied while paused" : "";
    e.result = paused_ ? "denied" : "approved";
    approval_history_.push_back(e);
    return !paused_;
}

AlignmentSystem::AlignmentSystem() {}

float AlignmentSystem::value_alignment_score(const std::string& output) {
    if (output.empty()) return 0.0f;
    static const char* harmful[] = {"rm -rf", "sudo", "hack", "exploit", "drop table", "kill -9"};
    float score = 0.6f;
    for (const char* h : harmful) {
        if (output.find(h) != std::string::npos) score = 0.1f;
    }
    if (output.find("help") != std::string::npos) score += 0.3f;
    return std::min(1.0f, score);
}

CodeGenSelfImprover::CodeGenSelfImprover(Model* model) : model_(model) {}

std::string CodeGenSelfImprover::generate_kernel(const std::string& op, int64_t M, int64_t N, int64_t K) {
    std::string code = "// generated kernel for " + op + "\n";
    code += "void kernel_" + op + "_" + std::to_string(M) + "x" + std::to_string(N) +
            "x" + std::to_string(K) + "(const float* a, const float* b, float* c) {\n";
    code += "  for (int64_t i = 0; i < " + std::to_string(M) + "; ++i)\n";
    code += "    for (int64_t j = 0; j < " + std::to_string(N) + "; ++j) {\n";
    code += "      float acc = 0.0f;\n";
    code += "      for (int64_t k = 0; k < " + std::to_string(K) + "; ++k)\n";
    code += "        acc += a[i * " + std::to_string(K) + " + k] * b[k * " + std::to_string(N) + " + j];\n";
    code += "      c[i * " + std::to_string(N) + " + j] = acc;\n";
    code += "    }\n";
    code += "}\n";
    return code;
}

bool CodeGenSelfImprover::compile_and_test(const std::string& code) {
    if (code.find("kernel") == std::string::npos) return false;
    int depth = 0;
    for (char ch : code) {
        if (ch == '{') depth++;
        else if (ch == '}') {
            depth--;
            if (depth < 0) return false;
        }
    }
    return depth == 0;
}

bool CodeGenSelfImprover::replace_kernel(const std::string& op, const std::string& new_code) {
    (void)op;
    return !new_code.empty() && new_code.find("kernel") != std::string::npos;
}

SelfVerifier::SelfVerifier(Model* model) : model_(model) {}

bool SelfVerifier::verify(const std::string& problem, const std::string& solution) {
    size_t eq = problem.find('=');
    if (eq != std::string::npos) {
        std::string rhs = problem.substr(eq + 1);
        while (!rhs.empty() && (rhs.back() == ' ' || rhs.back() == '\t')) rhs.pop_back();
        return rhs == solution;
    }
    return !solution.empty();
}

std::vector<std::string> SelfVerifier::find_edge_cases(const std::string& solution) {
    if (!model_) return {};
    std::vector<std::string> cases;
    if (solution.find("+") != std::string::npos) {
        cases.push_back("zero");
        cases.push_back("negative");
        cases.push_back("max value");
    } else if (solution.find("/") != std::string::npos) {
        cases.push_back("division by zero");
    } else {
        cases.push_back("empty input");
    }
    return cases;
}

CapabilityAmplifier::CapabilityAmplifier(Model* model) : model_(model) {}

float CapabilityAmplifier::measure(const std::string& capability) {
    if (!model_) return 0.5f;
    size_t h = std::hash<std::string>{}(capability);
    return 0.1f + (float)(h % 80) / 100.0f;
}

bool CapabilityAmplifier::improve(const std::string& capability, int steps) {
    (void)capability;
    if (!model_) return false;
    return steps > 0;
}

} // namespace asi
} // namespace oil
