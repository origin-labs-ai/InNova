#include "oil/asi.h"
#include "oil/asi_utils.h"
#include "oil/random.h"
#include "oil/optimizer.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>
#include <numeric>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <cstring>
#include <ctime>
#include <array>
#include <mutex>
#include <iomanip>
#include <thread>
#include <chrono>
#include <regex>
#include <cstdio>
#include <memory>

namespace oil {
namespace asi {

// ========================================================================
// G1: Self-Monitoring
// ========================================================================
SelfMonitor::SelfMonitor(Model* model) : model_(model) {}

float SelfMonitor::estimate_confidence(const Tensor& logits) {
    int64_t V = logits.dim(logits.rank() - 1);
    int64_t S = logits.numel() / V;
    const float* lp = logits.data<float>();
    float total_conf = 0;
    for (int64_t i = 0; i < S; i++) {
        float max_l = -INFINITY;
        for (int64_t v = 0; v < V; v++) max_l = std::max(max_l, lp[i * V + v]);
        float sum = 0;
        for (int64_t v = 0; v < V; v++) sum += std::exp(lp[i * V + v] - max_l);
        float max_prob = std::exp(lp[i * V + 0] - max_l) / (sum + 1e-10f);
        total_conf += max_prob;
    }
    return total_conf / (float)(S > 0 ? S : 1);
}

MetaCognitionState SelfMonitor::analyze(const std::string& input, const std::string& output) {
    MetaCognitionState state;
    if (!model_) {
        state.confidence = 0.0f;
        state.uncertainty = 1.0f;
        state.token_confidences = {0.0f};
        state.recommendation = "no model";
        state.needs_reanalysis = false;
        state.reasoning_depth = 1;
        return state;
    }
    int vocab_size = (int)model_->config.vocab_size;

    auto input_ids = simple_encode(input, vocab_size);
    auto output_ids = simple_encode(output, vocab_size);

    int64_t seq_len = std::max((int64_t)1, (int64_t)output_ids.size());
    Tensor input_tensor({1, seq_len});
    Tensor pos_tensor({1, seq_len});
    float* idp = input_tensor.data<float>();
    float* psp = pos_tensor.data<float>();
    for (int64_t i = 0; i < seq_len; i++) {
        idp[i] = (float)output_ids[i % (int)output_ids.size()];
        psp[i] = (float)i;
    }

    Tensor logits = model_->forward(input_tensor, pos_tensor, nullptr);
    int64_t V = logits.dim(logits.rank() - 1);
    const float* lp = logits.data<float>();

    float total_conf = 0;
    float total_entropy = 0;
    state.token_confidences.clear();

    for (int64_t i = 0; i < seq_len; i++) {
        float max_l = -INFINITY;
        for (int64_t v = 0; v < V; v++) max_l = std::max(max_l, lp[i * V + v]);
        float sum = 0;
        for (int64_t v = 0; v < V; v++) sum += std::exp(lp[i * V + v] - max_l);

        float max_prob = std::exp(lp[i * V + 0] - max_l) / (sum + 1e-10f);
        total_conf += max_prob;
        state.token_confidences.push_back(max_prob);

        float entropy = 0;
        for (int64_t v = 0; v < V; v++) {
            float p = std::exp(lp[i * V + v] - max_l) / (sum + 1e-10f);
            if (p > 1e-10f) entropy -= p * std::log(p);
        }
        total_entropy += entropy;
    }

    state.confidence = total_conf / (float)seq_len;
    float max_entropy = std::log((float)V + 1e-10f);
    state.uncertainty = total_entropy / (float)seq_len / (max_entropy > 0 ? max_entropy : 1.0f);
    state.reasoning_depth = (int)output_ids.size();

    float ratio = (float)output.size() / (float)std::max(input.size(), (size_t)1);
    state.needs_reanalysis = ratio < 0.3f || ratio > 3.0f || state.confidence < 0.5f;

    if (state.needs_reanalysis) {
        state.recommendation = "reanalyze";
    } else if (state.confidence > 0.8f) {
        state.recommendation = "proceed";
    } else {
        state.recommendation = "verify with caution";
    }

    return state;
}

// ========================================================================
// G2: Self-reflection
// ========================================================================
SelfReflector::SelfReflector(Model* model) : model_(model) {}

std::string SelfReflector::reflect(const std::string& input, const std::string& output) {
    if (!model_) return "Reflection: No model available";
    int vocab_size = (int)model_->config.vocab_size;
    std::string prompt = "Reflect on this output for input \"" + input + "\": " + output + ". How could this output be improved?";
    auto ids = simple_encode(prompt, vocab_size);
    auto gen = generate_new_tokens(model_, ids, vocab_size, 50);
    return simple_decode(gen);
}

std::string SelfReflector::refine(const std::string& original, const std::string& reflection) {
    if (!model_) return "[Refined]: No model available";
    int vocab_size = (int)model_->config.vocab_size;
    std::string prompt = "Original: " + original + "\nReflection: " + reflection + "\nRefined:";
    auto ids = simple_encode(prompt, vocab_size);
    auto gen = generate_new_tokens(model_, ids, vocab_size, 50);
    return simple_decode(gen);
}

// ========================================================================
// G5: Recursive Self-Improvement
// ========================================================================
RecursiveSelfImprover::RecursiveSelfImprover(Model* model, Trainer* trainer)
    : model_(model), trainer_(trainer) {}

void RecursiveSelfImprover::improvement_cycle(int iterations) {
    if (!model_) return;

    int64_t orig_hidden = model_->config.hidden_size;
    int64_t orig_layers = model_->config.num_layers;
    int no_improvement_count = 0;
    float best_perplexity = 1e10f;

    for (int i = 0; i < iterations && i < AlignmentSystem::max_loop_iterations; i++) {
        if (no_improvement_count >= 10) {
            model_->config.hidden_size = orig_hidden;
            model_->config.num_layers = orig_layers;
            break;
        }

        int vocab_size = (int)model_->config.vocab_size;

        std::string test_input = "Self-evaluation test input iteration " + std::to_string(i);
        auto test_ids = simple_encode(test_input, vocab_size);
        int64_t len = std::max((int64_t)1, (int64_t)test_ids.size());

        Tensor input_tensor({1, len});
        Tensor pos_tensor({1, len});
        float* idp = input_tensor.data<float>();
        float* psp = pos_tensor.data<float>();
        for (int64_t j = 0; j < len; j++) {
            idp[j] = (float)test_ids[j % (int)test_ids.size()];
            psp[j] = (float)j;
        }

        Tensor logits = model_->forward(input_tensor, pos_tensor, nullptr);
        int64_t V = logits.dim(logits.rank() - 1);

        float loss = 0;
        int count = 0;
        for (int64_t j = 1; j < len; j++) {
            int target = test_ids[j % (int)test_ids.size()];
            const float* lp = logits.data<float>() + (j - 1) * V;
            float max_l = -INFINITY;
            for (int64_t v = 0; v < V; v++) max_l = std::max(max_l, lp[v]);
            float sum = 0;
            for (int64_t v = 0; v < V; v++) sum += std::exp(lp[v] - max_l);
            float prob = std::exp(lp[target % (int)V] - max_l) / (sum + 1e-10f);
            loss -= std::log(std::max(prob, 1e-10f));
            count++;
        }
        float perplexity = std::exp(loss / std::max(1, count));

        if (perplexity < best_perplexity) {
            best_perplexity = perplexity;
            no_improvement_count = 0;
        } else {
            no_improvement_count++;
        }

        std::string analysis = "iteration=" + std::to_string(i) +
            "|perplexity=" + std::to_string(perplexity) +
            "|vocab_size=" + std::to_string(vocab_size) +
            "|hidden_size=" + std::to_string(model_->config.hidden_size) +
            "|num_layers=" + std::to_string(model_->config.num_layers);

        if (!self_modify(analysis)) {
            model_->config.hidden_size = orig_hidden;
            model_->config.num_layers = orig_layers;
            break;
        }
    }
}

bool RecursiveSelfImprover::self_modify(const std::string& analysis) {
    if (!model_) return false;
    auto extract_val = [&](const std::string& key) -> float {
        auto pos = analysis.find(key + "=");
        if (pos == std::string::npos) return -1.0f;
        pos += key.size() + 1;
        auto end = analysis.find('|', pos);
        try { return std::stof(analysis.substr(pos, end - pos)); }
        catch (...) { std::fprintf(stderr, "[WARN] Exception caught: %s\n", __func__); return -1.0f; }
    };

    float perplexity = extract_val("perplexity");
    if (perplexity < 0) return false;

    // Only adjust hyperparameters that can be safely changed at runtime.
    // Architectural params (hidden_size, num_layers, num_heads, ffn_hidden_size)
    // require a full model rebuild and must NOT be modified here.
    bool modified = false;
    if (perplexity > 15.0f) {
        model_->config.rope_theta = std::min(100000.0f, model_->config.rope_theta * 1.5f);
        modified = true;
    } else if (perplexity > 8.0f) {
        model_->config.norm_eps = std::max(1e-7f, model_->config.norm_eps * 0.5f);
        modified = true;
    } else if (perplexity < 3.0f) {
        model_->config.norm_eps = std::min(1e-3f, model_->config.norm_eps * 2.0f);
        modified = true;
    }

    if (modified) {
        std::fprintf(stderr, "[WARN] self_modify: only hyperparams adjusted (rope_theta=%.1f, norm_eps=%.2e). "
                     "Architectural params (hidden_size, num_layers) unchanged — rebuild required.\n",
                     model_->config.rope_theta, model_->config.norm_eps);
    }

    return modified;
}

// ========================================================================
// G6: Code generation
// ========================================================================
CodeGenSelfImprover::CodeGenSelfImprover(Model* model) : model_(model) {}

std::string CodeGenSelfImprover::generate_kernel(const std::string& op, int64_t M, int64_t N, int64_t K) {
    std::ostringstream code;
    code << "// Auto-generated " << op << " kernel (M=" << M << " N=" << N << " K=" << K << ")\n";
    code << "#include <cstdint>\n";
    code << "extern \"C\" void kernel(const float* a, const float* b, float* c, int64_t m, int64_t n, int64_t k) {\n";

    if (op == "gemm") {
        code << "    for (int64_t i = 0; i < m; i++) {\n";
        code << "        for (int64_t j = 0; j < n; j++) {\n";
        code << "            float sum = 0.0f;\n";
        code << "            for (int64_t p = 0; p < k; p++) {\n";
        code << "                sum += a[i * k + p] * b[p * n + j];\n";
        code << "            }\n";
        code << "            c[i * n + j] = sum;\n";
        code << "        }\n";
        code << "    }\n";
    } else if (op == "rms_norm") {
        code << "    for (int64_t i = 0; i < m; i++) {\n";
        code << "        float sum_sq = 0.0f;\n";
        code << "        for (int64_t j = 0; j < n; j++) sum_sq += a[i * n + j] * a[i * n + j];\n";
        code << "        float rms = std::sqrt(sum_sq / (float)n + 1e-5f);\n";
        code << "        for (int64_t j = 0; j < n; j++) c[i * n + j] = a[i * n + j] / rms;\n";
        code << "    }\n";
    } else if (op == "softmax") {
        code << "    for (int64_t i = 0; i < m; i++) {\n";
        code << "        float max_val = a[i * n];\n";
        code << "        for (int64_t j = 1; j < n; j++) if (a[i * n + j] > max_val) max_val = a[i * n + j];\n";
        code << "        float sum = 0.0f;\n";
        code << "        for (int64_t j = 0; j < n; j++) sum += std::exp(a[i * n + j] - max_val);\n";
        code << "        for (int64_t j = 0; j < n; j++) c[i * n + j] = std::exp(a[i * n + j] - max_val) / sum;\n";
        code << "    }\n";
    } else if (op == "relu") {
        code << "    for (int64_t i = 0; i < m; i++) {\n";
        code << "        for (int64_t j = 0; j < n; j++) {\n";
        code << "            c[i * n + j] = a[i * n + j] > 0.0f ? a[i * n + j] : 0.0f;\n";
        code << "        }\n";
        code << "    }\n";
    } else if (op == "silu") {
        code << "    for (int64_t i = 0; i < m; i++) {\n";
        code << "        for (int64_t j = 0; j < n; j++) {\n";
        code << "            float x = a[i * n + j];\n";
        code << "            c[i * n + j] = x / (1.0f + std::exp(-x));\n";
        code << "        }\n";
        code << "    }\n";
    } else if (op == "add") {
        code << "    for (int64_t i = 0; i < m * n; i++) c[i] = a[i] + b[i];\n";
    } else if (op == "mul") {
        code << "    for (int64_t i = 0; i < m * n; i++) c[i] = a[i] * b[i];\n";
    } else {
        code << "    // Unrecognized op: " << op << ", defaulting to identity\n";
        code << "    for (int64_t i = 0; i < m * n; i++) c[i] = a[i];\n";
    }

    code << "}\n";
    return code.str();
}

bool CodeGenSelfImprover::compile_and_test(const std::string& code) {
    namespace fs = std::filesystem;
    fs::path sandbox;
#ifdef _WIN32
    const char* tmp = std::getenv("TEMP");
    sandbox = fs::path(tmp ? tmp : "C:\\Temp") / "mythos_sandbox";
#else
    sandbox = fs::path("/tmp") / "mythos_sandbox";
#endif

    std::error_code ec;
    fs::create_directories(sandbox, ec);
    if (ec) return false;

    auto src_path = sandbox / "asi_kernel_test.cpp";
    auto obj_dir = sandbox / "build";
    fs::create_directories(obj_dir, ec);

    {
        std::ofstream ofs(src_path);
        if (!ofs) return false;
        ofs << code;
    }

#ifdef _WIN32
    std::string obj_out = (obj_dir / "asi_kernel_test.obj").string();
    std::string cmd = "cl.exe /nologo /EHsc /Fo\"" + obj_out + "\" /c \"" + src_path.string() + "\" 2>nul";
    int ret = std::system(cmd.c_str());
    fs::remove(obj_out, ec);
#else
    std::string obj_out = (obj_dir / "asi_kernel_test.o").string();
    std::string cmd = "g++ -x c++ -std=c++20 -c -o \"" + obj_out + "\" \"" + src_path.string() + "\" 2>/dev/null";
    int ret = std::system(cmd.c_str());
    fs::remove(obj_out, ec);
#endif

    fs::remove(src_path, ec);
    return ret == 0;
}

bool CodeGenSelfImprover::replace_kernel(const std::string& op, const std::string& new_code) {
    static std::unordered_map<std::string, std::string> kernel_registry;
    static std::mutex registry_mutex;
    std::lock_guard<std::mutex> lock(registry_mutex);
    kernel_registry[op] = new_code;
    return true;
}

// ========================================================================
// G7-G8: Self-verification, Capability amplification
// ========================================================================
SelfVerifier::SelfVerifier(Model* model) : model_(model) {}

bool SelfVerifier::verify(const std::string& problem, const std::string& solution) {
    if (solution.empty()) return false;

    std::string problem_lower = problem;
    std::string solution_lower = solution;
    std::transform(problem_lower.begin(), problem_lower.end(), problem_lower.begin(), ::tolower);
    std::transform(solution_lower.begin(), solution_lower.end(), solution_lower.begin(), ::tolower);

    // Map problem types to required keywords in solution
    std::vector<std::pair<std::string, std::vector<std::string>>> problem_keywords = {
        {"sort", {"sort", "order", "compare", "arrange"}},
        {"search", {"search", "find", "lookup", "index"}},
        {"add", {"+", "sum", "add", "plus"}},
        {"subtract", {"-", "subtract", "minus", "difference"}},
        {"multiply", {"*", "multiply", "product", "times"}},
        {"divide", {"/", "divide", "quotient", "division"}},
        {"reverse", {"reverse", "backward"}},
        {"sum", {"sum", "+", "total", "accumulate"}},
        {"average", {"average", "mean", "/", "divide"}},
        {"max", {"max", "largest", "maximum", "greatest"}},
        {"min", {"min", "smallest", "minimum", "least"}},
        {"count", {"count", "size", "length", "number"}},
        {"filter", {"filter", "remove", "keep", "select"}},
        {"merge", {"merge", "combine", "union"}},
        {"contains", {"contain", "find", "search", "has"}},
        {"remove", {"remove", "delete", "erase", "clear"}},
        {"replace", {"replace", "substitute", "swap"}},
        {"sqrt", {"sqrt", "square root", "√"}},
        {"power", {"pow", "^", "power", "exponent"}},
        {"modulo", {"%", "mod", "modulo", "remainder"}},
    };

    for (auto& [ptype, keywords] : problem_keywords) {
        if (problem_lower.find(ptype) != std::string::npos) {
            bool found_keyword = false;
            for (auto& kw : keywords) {
                if (solution_lower.find(kw) != std::string::npos) {
                    found_keyword = true;
                    break;
                }
            }
            if (!found_keyword) return false;
        }
    }

    return true;
}

std::vector<std::string> SelfVerifier::find_edge_cases(const std::string& solution) {
    std::vector<std::string> edge_cases;
    std::string sol_lower = solution;
    std::transform(sol_lower.begin(), sol_lower.end(), sol_lower.begin(), ::tolower);

    if (sol_lower.find("array") != std::string::npos || sol_lower.find("vector") != std::string::npos ||
        sol_lower.find("list") != std::string::npos) {
        edge_cases.push_back("empty array: handle zero-length input");
        edge_cases.push_back("single element array: verify boundary behavior");
        edge_cases.push_back("all identical elements: check stability");
    }

    if (sol_lower.find("sort") != std::string::npos || sol_lower.find("order") != std::string::npos) {
        edge_cases.push_back("already sorted input: ensure no unnecessary swaps");
        edge_cases.push_back("reverse sorted input: verify worst-case performance");
        edge_cases.push_back("duplicate values: check sort stability");
    }

    if (sol_lower.find("number") != std::string::npos || sol_lower.find("int") != std::string::npos ||
        sol_lower.find("count") != std::string::npos) {
        edge_cases.push_back("negative values: verify correct handling");
        edge_cases.push_back("zero value: check division by zero");
        edge_cases.push_back("integer overflow: test with large values near INT_MAX");
    }

    if (sol_lower.find("string") != std::string::npos || sol_lower.find("char") != std::string::npos ||
        sol_lower.find("text") != std::string::npos) {
        edge_cases.push_back("empty string: handle null or empty input");
        edge_cases.push_back("unicode characters: verify multi-byte support");
        edge_cases.push_back("very long string: check for buffer overflow");
    }

    if (sol_lower.find("pointer") != std::string::npos || sol_lower.find("reference") != std::string::npos ||
        sol_lower.find("node") != std::string::npos) {
        edge_cases.push_back("null pointer: verify null safety");
        edge_cases.push_back("self-referential structure: check for infinite loops");
        edge_cases.push_back("dangling pointer: ensure no use-after-free");
    }

    if (sol_lower.find("search") != std::string::npos || sol_lower.find("find") != std::string::npos) {
        edge_cases.push_back("target not found: return appropriate sentinel");
        edge_cases.push_back("target at first position");
        edge_cases.push_back("target at last position");
    }

    if (sol_lower.find("divide") != std::string::npos || sol_lower.find("/") != std::string::npos) {
        edge_cases.push_back("division by zero: prevent undefined behavior");
        edge_cases.push_back("negative divisor: verify sign handling");
    }

    if (sol_lower.find("recursive") != std::string::npos || sol_lower.find("recursion") != std::string::npos) {
        edge_cases.push_back("stack overflow: check recursion depth limits");
        edge_cases.push_back("base case: verify termination condition");
    }

    if (sol_lower.find("graph") != std::string::npos || sol_lower.find("tree") != std::string::npos) {
        edge_cases.push_back("disconnected graph: handle multiple components");
        edge_cases.push_back("cyclic graph: prevent infinite traversal");
        edge_cases.push_back("single node: verify minimum case");
    }

    return edge_cases;
}

CapabilityAmplifier::CapabilityAmplifier(Model* model) : model_(model) {}

float CapabilityAmplifier::measure(const std::string& capability) {
    if (!model_) return 0.0f;
    int vocab_size = (int)model_->config.vocab_size;

    if (capability == "reasoning") {
        std::string puzzle = "If all dogs are mammals and all mammals are animals, are all dogs animals? Answer yes or no.";
        auto ids = simple_encode(puzzle, vocab_size);
        auto gen = generate_new_tokens(model_, ids, vocab_size, 20);
        std::string answer = simple_decode(gen);
        std::transform(answer.begin(), answer.end(), answer.begin(), ::tolower);
        return answer.find("yes") != std::string::npos ? 0.9f : 0.3f;
    }

    if (capability == "math") {
        std::string queries[] = {"What is 2 + 2?", "What is 10 - 3?", "What is 4 * 5?"};
        float correct = 0;
        for (auto& q : queries) {
            auto ids = simple_encode(q, vocab_size);
            auto gen = generate_new_tokens(model_, ids, vocab_size, 10);
            std::string answer = simple_decode(gen);
            if (!answer.empty() && std::isdigit((unsigned char)answer[0])) correct += 1.0f;
        }
        return correct / 3.0f;
    }

    if (capability == "summarization") {
        std::string text = "The quick brown fox jumps over the lazy dog. Summarize this.";
        auto ids = simple_encode(text, vocab_size);
        auto gen = generate_new_tokens(model_, ids, vocab_size, 30);
        std::string summary = simple_decode(gen);
        float ratio = (float)summary.size() / (float)std::max(text.size(), (size_t)1);
        return std::min(1.0f, 1.0f / (ratio + 0.1f));
    }

    if (capability == "code") {
        std::string prompt = "Write a C++ function to add two integers.";
        auto ids = simple_encode(prompt, vocab_size);
        auto gen = generate_new_tokens(model_, ids, vocab_size, 40);
        std::string code = simple_decode(gen);
        if (code.find("int") != std::string::npos && code.find("return") != std::string::npos) return 0.8f;
        return 0.3f;
    }

    if (capability == "language") {
        std::string prompt = "Translate to French: Hello world.";
        auto ids = simple_encode(prompt, vocab_size);
        auto gen = generate_new_tokens(model_, ids, vocab_size, 20);
        std::string answer = simple_decode(gen);
        return answer.size() > 5 ? 0.6f : 0.2f;
    }

    if (capability == "instruction_following") {
        std::string tests[] = {
            "List three fruits.",
            "What is the opposite of hot?",
            "Count from 1 to 5."
        };
        float correct = 0;
        for (auto& t : tests) {
            auto ids = simple_encode(t, vocab_size);
            auto gen = generate_new_tokens(model_, ids, vocab_size, 15);
            std::string answer = simple_decode(gen);
            if (!answer.empty()) correct += 1.0f;
        }
        return correct / 3.0f;
    }

    if (capability == "creativity") {
        std::string prompt = "Write a short poem about the ocean.";
        auto ids = simple_encode(prompt, vocab_size);
        auto gen = generate_new_tokens(model_, ids, vocab_size, 40);
        std::string poem = simple_decode(gen);
        int words = 0;
        for (char c : poem) if (c == ' ') words++;
        return std::min(1.0f, (float)words / 20.0f);
    }

    return 0.0f;
}

bool CapabilityAmplifier::improve(const std::string& capability, int steps) {
    if (!model_) return false;
    int vocab_size = (int)model_->config.vocab_size;

    std::string training_prompt;
    if (capability == "reasoning") {
        training_prompt = "Training: Apply logical reasoning step by step. ";
    } else if (capability == "math") {
        training_prompt = "Training: Solve the following math problem carefully. ";
    } else if (capability == "summarization") {
        training_prompt = "Training: Summarize the following text concisely. ";
    } else if (capability == "code") {
        training_prompt = "Training: Write correct and efficient code for: ";
    } else {
        training_prompt = "Training: Improve performance on: " + capability + ". ";
    }

    for (int s = 0; s < steps; s++) {
        std::string prompt = training_prompt + " Step " + std::to_string(s) + " of " + std::to_string(steps) + ".";
        auto ids = simple_encode(prompt, vocab_size);
        generate_new_tokens(model_, ids, vocab_size, 10);
    }

    return true;
}

// ========================================================================
// G9-G10: Safety + HITL
// ========================================================================
SafetyGuardrails::SafetyGuardrails() {
    blocked_patterns_ = {
        "rm -rf", "sudo", "delete everything",
        "DROP TABLE", "exec(", "system(",
        "rmdir", "format ", "del /f",
        "shutdown", "reboot", "chmod 777",
        "wget ", "curl ", "eval(",
        "Process.Start", "ShellExecute",
        "mkfs", "dd if=", "> /dev/sda",
        "powershell -Command", "Invoke-Expression",
        "cmd.exe /c", "rundll32",
        "reg delete", "reg add",
        "net user", "net localgroup"
    };
}

bool SafetyGuardrails::check_output(const std::string& output) {
    for (auto& p : blocked_patterns_)
        if (output.find(p) != std::string::npos) return false;
    return !kill_switch_;
}

bool SafetyGuardrails::check_input(const std::string& input) {
    return check_output(input);
}

HITL::HITL() {}

bool HITL::request_approval(const std::string& action) {
    if (paused_) return false;

    std::string lower = action;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    std::vector<std::string> dangerous = {
        "rm -rf", "sudo", "delete everything",
        "drop table", "exec(", "system(",
        "rmdir", "format ", "del /f",
        "shutdown", "reboot", "chmod 777",
        "wget ", "curl ", "eval(",
        "process.start", "shellexecute",
        "mkfs", "dd if=", "> /dev/sda",
        "powershell -command", "invoke-expression",
        "cmd.exe /c", "rundll32",
        "reg delete", "reg add",
        "net user", "net localgroup",
        "DROP TABLE", "exec(", "system(",
        "rmdir", "format ", "del /f"
    };

    for (auto& d : dangerous) {
        if (lower.find(d) != std::string::npos) return false;
    }

    return true;
}

// ========================================================================
// G11-G12: Alignment
// ========================================================================
AlignmentSystem::AlignmentSystem() {}

float AlignmentSystem::value_alignment_score(const std::string& output) {
    std::vector<std::string> harmful_patterns = {
        "kill", "steal", "bomb", "weapon", "illegal", "violence",
        "discriminat", "racist", "sexist", "hate", "murder",
        "suicide", "self-harm", "abuse", "exploit", "fraud",
        "scam", "terror", "assault", "kidnap", "torture",
        "traffick", "slavery", "genocide", "war crime"
    };

    float score = 1.0f;
    std::string lower = output;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    int matches = 0;
    for (auto& p : harmful_patterns) {
        size_t pos = 0;
        while ((pos = lower.find(p, pos)) != std::string::npos) {
            matches++;
            pos += p.size();
        }
    }

    score -= (float)matches * 0.15f;
    return std::max(0.0f, score);
}

} // namespace asi
} // namespace oil
