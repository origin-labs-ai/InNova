// ============================================================================
// Tests for the three native fine-tuning strategies (oil/fine_tuning.h):
//   1. SelectiveFineTuner      — gradient/Fisher-driven "where to overwrite"
//   2. RankAdapterEngine       — native low-rank delta adapters (OIL-quantized)
//   3. KnowledgeExpansionEngine— new-weight slots + hallucination guard
// ============================================================================

#include "oil/fine_tuning.h"
#include "oil/test.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <random>

using namespace oil;

// Deterministic model weights: DenseModel::init_weights() uses a
// random_device seed, which makes short training tests flaky. Tests seed
// their own fixed RNG so every run is reproducible.
static void det_init(DenseModel& m) {
    std::vector<Tensor*> params;
    m.get_parameters(params);
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 0.1f);
    for (auto* p : params) {
        float* d = p->data<float>();
        for (int64_t i = 0; i < p->numel(); i++) d[i] = (float)nd(rng);
    }
    // LayerNorm-style gain params must stay 1 so RMSNorm behaves sanely.
    m.norm->weight.fill(1.0f);
    for (auto& l : m.layers) {
        l->attention_norm.weight.fill(1.0f);
        l->ffn_norm.weight.fill(1.0f);
    }
}

static TransformerConfig make_cfg() {
    TransformerConfig cfg;
    cfg.hidden_size = 64;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.head_dim = 16;
    cfg.ffn_hidden_size = 128;
    cfg.vocab_size = 200;
    cfg.max_seq_len = 64;
    return cfg;
}

// Synthetic next-token data on a token band [lo, hi).
static void make_batch(int64_t B, int64_t S, int64_t vocab, int64_t lo, int64_t hi,
                       Tensor& in, Tensor& pos, Tensor& tgt) {
    in = Tensor(Shape{B, S}, DType::F32);
    pos = Tensor(Shape{B, S}, DType::F32);
    tgt = Tensor(Shape{B, S}, DType::F32);
    RNG rng(1234);
    for (int64_t b = 0; b < B; b++)
        for (int64_t s = 0; s < S; s++) {
            int64_t idx = b * S + s;
            int64_t tok = lo + (int64_t)(rng.uniform() * (double)(hi - lo));
            in.data<float>()[idx] = (float)tok;
            pos.data<float>()[idx] = (float)s;
            tgt.data<float>()[idx] = (float)((tok + 1) % hi);
        }
    (void)vocab;
}

// Next-token data over the FULL vocab with a learnable shift pattern. Used
// for the adapter tests: the base pretrains on shift A over every token (so
// ALL embeddings are trained), and the adapter must learn the DIFFERENT
// shift B — genuinely new knowledge on already-seen tokens.
static void make_batch_shift(int64_t B, int64_t S, int64_t vocab, int64_t shift,
                             Tensor& in, Tensor& pos, Tensor& tgt) {
    in = Tensor(Shape{B, S}, DType::F32);
    pos = Tensor(Shape{B, S}, DType::F32);
    tgt = Tensor(Shape{B, S}, DType::F32);
    RNG rng(9876);
    for (int64_t b = 0; b < B; b++)
        for (int64_t s = 0; s < S; s++) {
            int64_t idx = b * S + s;
            int64_t tok = (int64_t)(rng.uniform() * (double)vocab) % vocab;
            in.data<float>()[idx] = (float)tok;
            pos.data<float>()[idx] = (float)s;
            int64_t t = ((tok + shift) % vocab + vocab) % vocab;
            tgt.data<float>()[idx] = (float)t;
        }
}

// Cross-entropy over {B, S, V} logits (flat row-major, V on the last dim).
static float eval_loss(const Tensor& logits, const Tensor& targets) {
    int64_t V = logits.dim(logits.rank() - 1);
    int64_t N = logits.numel() / V;
    const float* ld = logits.data<float>();
    const float* td = targets.data<float>();
    double loss = 0.0;
    for (int64_t i = 0; i < N; i++) {
        int64_t t = (int64_t)td[i];
        if (t < 0 || t >= V) continue;
        float mx = ld[i * V];
        for (int64_t v = 1; v < V; v++)
            if (ld[i * V + v] > mx) mx = ld[i * V + v];
        double sum = 0.0;
        for (int64_t v = 0; v < V; v++)
            sum += std::exp((double)ld[i * V + v] - (double)mx);
        loss += -((double)ld[i * V + t] - (double)mx - std::log(sum));
    }
    return (float)(loss / (double)N);
}

static void snapshot(DenseModel& m, std::vector<std::vector<float>>& snap) {
    std::vector<Tensor*> params;
    m.get_parameters(params);
    snap.clear();
    for (auto* p : params)
        snap.emplace_back(p->data<float>(), p->data<float>() + p->numel());
}

static bool same_as_snapshot(DenseModel& m, const std::vector<std::vector<float>>& snap) {
    std::vector<Tensor*> params;
    m.get_parameters(params);
    if (params.size() != snap.size()) return false;
    for (size_t i = 0; i < params.size(); i++)
        if (std::memcmp(params[i]->data<float>(), snap[i].data(),
                        snap[i].size() * sizeof(float)) != 0) return false;
    return true;
}

// Deterministic fan-in scaled init (mirrors DenseModel::init_weights but with
// a FIXED seed, so every run is reproducible — init_weights uses
// std::random_device, which makes the adapter/knowledge tests flaky).
static void det_init_fan(DenseModel& m) {
    std::vector<Tensor*> params;
    m.get_parameters(params);
    std::mt19937 rng(2026);
    for (auto* p : params) {
        int64_t fan_in = (p->rank() >= 2) ? p->dim(p->rank() - 1) : p->numel();
        float std_dev = std::sqrt(2.0f / (float)std::max<int64_t>(1, fan_in));
        std::normal_distribution<float> nd(0.0f, std_dev);
        float* d = p->data<float>();
        for (int64_t i = 0; i < p->numel(); i++) d[i] = nd(rng);
    }
    m.norm->weight.fill(1.0f);
    for (auto& l : m.layers) {
        l->attention_norm.weight.fill(1.0f);
        l->ffn_norm.weight.fill(1.0f);
    }
}

static void pretrain_base(DenseModel& m, const Tensor& in, const Tensor& pos,
                          const Tensor& tgt, int steps, float lr);

// ============================================================================
// Test 1 — SelectiveFineTuner
// ============================================================================
static void test_selective() {
    TEST_SUITE("SelectiveFineTuner (where to overwrite)");
    AutogradEngine::instance().reset();

    DenseModel model(make_cfg());
    det_init(model);

    Tensor in, pos, tgt;
    make_batch(2, 16, model.vocab_size(), 10, 110, in, pos, tgt);

    // Train the base to a real starting point first: selective fine-tuning
    // picks the blocks whose gradients prove change is needed. On a random
    // model every block needs change, so the saliency gate freezes ~90% of
    // the capacity and loss cannot drop (it even drifts up). A pretrained
    // base has a real loss landscape to select from.
    pretrain_base(model, in, pos, tgt, 120, 0.05f);

    SelectiveFineTuner tuner(&model);
    SelectiveTunerConfig cfg;
    cfg.learning_rate = 2e-3f;
    cfg.block_size = 32;
    cfg.saliency_sigma = 1.5f;
    cfg.min_select_fraction = 0.02f;
    cfg.warmup_steps = 3;
    cfg.save_interval = 0;
    // STE roundtrip keeps fine-tuned weights exactly representable in the
    // target format. OIL8's single-rms-scale lattice is far too lossy for
    // TRAINED weights (blocks with an outlier collapse the small values),
    // so the STE target is OIL16 (FP16-class, ~lossless for training).
    cfg.ste_roundtrip = true;
    cfg.target_format = Format::OIL16;
    tuner.configure(cfg);

    tuner.accumulate_fisher(in, pos, tgt);
    float initial = eval_loss(model.forward(in, pos), tgt);
    tuner.fine_tune(in, pos, tgt, 60);
    float final = eval_loss(model.forward(in, pos), tgt);

    printf("  loss %.4f -> %.4f (selected %.1f%% of %d blocks)\n",
           initial, final, tuner.stats().selected_fraction * 100.0f,
           tuner.stats().total_blocks);

    TEST_CHECK(final < initial - 0.005f, "selective fine-tune reduces loss");
    TEST_CHECK(tuner.stats().total_blocks > 0, "block saliency computed");
    TEST_CHECK(tuner.stats().selected_fraction > 0.0f &&
                   tuner.stats().selected_fraction <= 1.0f,
               "selection fraction inside (0, 1]");

    tuner.save("test_selective.oil");
    DenseModel loaded(make_cfg());
    loaded.load("test_selective.oil");
    TEST_CHECK(loaded.param_count() == model.param_count(),
               "quantized OIL save/load preserves parameter count");
    std::remove("test_selective.oil");
}

// ============================================================================
// Test 2 — RankAdapterEngine (native low-rank delta adapters)
// ============================================================================
static void test_rank_adapters() {
    TEST_SUITE("RankAdapterEngine (native OIL-rank)");
    AutogradEngine::instance().reset();

    DenseModel model(make_cfg());
    det_init_fan(model);
    Tensor a_in, a_pos, a_tgt, b_in, b_pos, b_tgt;
    // Full-vocab data: the base pretrains on "next token +1" over EVERY
    // token (all embeddings trained), and the adapter must learn "+2" — new
    // knowledge on already-seen tokens. (A band-restricted B would fail on
    // untrained embeddings, which no down-projection adapter can fix.)
    make_batch_shift(4, 32, model.vocab_size(), 1, a_in, a_pos, a_tgt);
    make_batch_shift(4, 32, model.vocab_size(), 2, b_in, b_pos, b_tgt);

    // Base knows pattern A only; B is genuinely new knowledge for it.
    pretrain_base(model, a_in, a_pos, a_tgt, 40, 0.05f);

    RankAdapterEngine engine(&model);
    RankAdapterConfig cfg;
    cfg.rank = 16;
    cfg.alpha = 8.0f;              // scale = alpha/rank = 0.5
    cfg.learning_rate = 2e-2f;
    cfg.warmup_steps = 5;
    cfg.factor_format = Format::OIL16;  // fp16-class store precision
    cfg.output_path = "test_adapters.nrad";
    engine.configure(cfg);
    engine.init_adapters();

    TEST_CHECK(engine.adapter_param_count() > 0, "adapter factors allocated");
    TEST_CHECK(engine.adapter_param_count() == 2 * (16 * 128 + 64 * 16),
               "adapter param count exact (2 layers x (A + B))");

    std::vector<std::vector<float>> snap;
    snapshot(model, snap);

    // B = 0 at init => zero delta => logits equal the plain base.
    Tensor f_base = model.forward(a_in, a_pos);
    Tensor f_adapt = engine.forward_with_adapters(a_in, a_pos);
    TEST_CHECK(f_base.numel() == f_adapt.numel(), "logits shapes match");
    bool zero_delta = f_base.numel() == f_adapt.numel();
    if (zero_delta) {
        const float* bd = f_base.data<float>();
        const float* ad = f_adapt.data<float>();
        for (int64_t i = 0; i < f_base.numel(); i++)
            if (std::fabs(bd[i] - ad[i]) > 1e-3f) { zero_delta = false; break; }
    }
    TEST_CHECK(zero_delta, "adapters start at zero delta (identity init)");

    float lossB0 = eval_loss(model.forward(b_in, b_pos), b_tgt);
    float best = lossB0;
    for (int s = 0; s < 800; s++) {
        engine.train_step(b_in, b_pos, b_tgt);
        best = std::min(best, engine.last_loss());
    }
    float final = engine.last_loss();
    printf("  loss on B %.4f -> %.4f (best %.4f, adapters: %lld params)\n",
           lossB0, final, best, (long long)engine.adapter_param_count());

    TEST_CHECK(best < lossB0 - 0.05f, "adapters learn new knowledge (loss on B drops)");
    TEST_CHECK(same_as_snapshot(model, snap),
               "base weights untouched during adapter training");

    // Save -> load roundtrip reproduces the forward.
    engine.save("test_adapters.nrad");
    RankAdapterEngine engine2(&model);
    engine2.configure(cfg);
    bool ok_load = engine2.load("test_adapters.nrad");
    TEST_CHECK(ok_load, "adapter file loads");
    TEST_CHECK(engine2.adapter_param_count() == engine.adapter_param_count(),
               "adapter params survive roundtrip");
    Tensor f1 = engine.forward_with_adapters(b_in, b_pos);
    Tensor f2 = engine2.forward_with_adapters(b_in, b_pos);
    // Quantized store => small reconstruction error; compare the loss the
    // loaded adapters produce rather than per-logit equality (robust to the
    // store precision and the model's random initialization).
    float l1 = eval_loss(f1, b_tgt);
    float l2 = eval_loss(f2, b_tgt);
    TEST_CHECK(std::fabs(l1 - l2) < 0.02f, "loaded adapters reproduce the forward");
    std::remove("test_adapters.nrad");

    // Capture the adapter-augmented forward BEFORE merging.
    float l_adapt = eval_loss(engine.forward_with_adapters(b_in, b_pos), b_tgt);

    engine2.merge_into_base();
    TEST_CHECK(!same_as_snapshot(model, snap),
               "merge_into_base folds deltas into base weights");
    // The merged PLAIN model forward must reproduce the adapter forward (the
    // deltas are now folded into the down-proj weights at the engine's actual
    // buffer positions) — this catches the classic merge-position bug.
    float l_merged = eval_loss(model.forward(b_in, b_pos), b_tgt);
    TEST_CHECK(std::fabs(l_adapt - l_merged) < 0.02f,
               "merged base reproduces the adapter forward");
}

// ============================================================================
// Test 3 — KnowledgeExpansionEngine (new weights, no forgetting, guard)
// ============================================================================
static void pretrain_base(DenseModel& m, const Tensor& in, const Tensor& pos,
                          const Tensor& tgt, int steps, float lr) {
    std::vector<Tensor*> params;
    m.get_parameters(params);
    for (auto* p : params) p->requires_grad(true);
    auto& engine = AutogradEngine::instance();
    SGD opt(lr, 0.9f);
    opt.add_param_group(params);
    for (int s = 0; s < steps; s++) {
        for (auto* p : params)
            if (p->has_grad()) p->zero_grad();
        AutogradEngine::set_enabled(true);
        for (auto* p : params) engine.register_parameter(p);
        Tensor logits = m.forward(in, pos);
        Tensor loss_t = AutogradEngine::cross_entropy_op(logits, tgt);
        engine.backward(loss_t);
        engine.clear();
        AutogradEngine::set_enabled(false);
        opt.step();
    }
}

static void test_knowledge_expansion() {
    TEST_SUITE("KnowledgeExpansionEngine (hallucination guard)");
    AutogradEngine::instance().reset();

    DenseModel model(make_cfg());
    det_init_fan(model);
    Tensor a_in, a_pos, a_tgt, b_in, b_pos, b_tgt, g_in, g_pos, g_tgt;
    // A: full-vocab "next token +1" pattern (base pretrains on it over EVERY
    // token, so all embeddings are trained and the base becomes confident).
    make_batch_shift(2, 16, model.vocab_size(), 1, a_in, a_pos, a_tgt);
    // B: full-vocab "next token +2" — genuinely NEW knowledge on already-seen
    // tokens: the slot must learn a different pattern the base does not know.
    make_batch_shift(2, 16, model.vocab_size(), 2, b_in, b_pos, b_tgt);
    // G: band-restricted data OUTSIDE the pretrain band. The base has never
    // seen those tokens, so it is genuinely UNSURE there (low max-softmax) —
    // exactly the condition under which the hallucination guard must consult
    // the slots.
    make_batch(2, 16, model.vocab_size(), 150, 200, g_in, g_pos, g_tgt);

    // Base learns pattern A only.
    pretrain_base(model, a_in, a_pos, a_tgt, 80, 0.05f);
    float lossA0 = eval_loss(model.forward(a_in, a_pos), a_tgt);
    float lossB0 = eval_loss(model.forward(b_in, b_pos), b_tgt);
    printf("  base on A: %.4f   on B: %.4f\n", lossA0, lossB0);
    TEST_CHECK(lossA0 < lossB0, "base knows A better than B");

    std::vector<std::vector<float>> snap;
    snapshot(model, snap);

    KnowledgeExpansionEngine kexp(&model);
    KnowledgeExpansionConfig kcfg;
    kcfg.slot_width = 128;
    kcfg.learning_rate = 2e-2f;
    kcfg.warmup_steps = 5;
    // Trust the base only when it is near-certain (max-softmax >= 0.95);
    // anything below that lets the slot weigh in. The slot's fixed gate
    // (|tanh(1)| ~ 0.76) is always open, so the consult decision is driven
    // purely by base confidence — an untrained slot contributes ~0 anyway.
    kcfg.confidence_threshold = 0.95f;
    kcfg.slot_gate_threshold = 0.05f;
    kexp.configure(kcfg);
    kexp.freeze_base(true);
    int64_t sid = kexp.add_slot("patternB", 128);
    TEST_CHECK(sid == 0, "knowledge slot added");
    TEST_CHECK(kexp.slot_param_count() > 0, "slot parameters allocated");

    float best = lossB0;
    for (int s = 0; s < 800; s++) {
        kexp.train_step(b_in, b_pos, b_tgt);
        best = std::min(best, kexp.last_loss());
    }
    float final = kexp.last_loss();
    printf("  slot on B: %.4f -> %.4f (best %.4f)\n", lossB0, final, best);
    TEST_CHECK(best < lossB0 - 0.02f, "slot learns the new knowledge");
    TEST_CHECK(same_as_snapshot(model, snap),
               "base weights bit-identical after slot training");

    float lossA1 = eval_loss(model.forward(a_in, a_pos), a_tgt);
    float lossA_guarded = eval_loss(kexp.forward_guarded(a_in, a_pos), a_tgt);
    printf("  A loss (base-only) %.4f   (guarded) %.4f\n", lossA1, lossA_guarded);
    TEST_CHECK(std::fabs(lossA1 - lossA0) < 1e-6f,
               "no forgetting: pure base loss on A unchanged");
    TEST_CHECK(lossA_guarded < lossA0 + 0.5f,
               "no catastrophic forgetting: guard keeps old knowledge intact");

    // Hallucination guard: base is unsure on G (unseen band) => slots consulted.
    Tensor conf, used;
    Tensor guarded = kexp.forward_guarded(g_in, g_pos, nullptr, &conf, &used);
    TEST_CHECK(conf.numel() == g_in.numel(), "confidence mask shape");
    TEST_CHECK(used.numel() == g_in.numel(), "slot-used mask shape");
    TEST_CHECK(guarded.numel() > 0, "guarded forward produces logits");
    int64_t nconsult = 0;
    const float* ud = used.data<float>();
    for (int64_t i = 0; i < used.numel(); i++)
        if (ud[i] > 0.5f) nconsult++;
    printf("  guard: %lld/%lld G tokens consult slots\n",
           (long long)nconsult, (long long)used.numel());
    TEST_CHECK(nconsult > 0, "guard consults slots on new knowledge");

    // Save -> load roundtrip.
    kexp.save("test_slots.kexp");
    KnowledgeExpansionEngine kexp2(&model);
    kexp2.configure(kcfg);
    TEST_CHECK(kexp2.load("test_slots.kexp"), "kexp file loads");
    TEST_CHECK(kexp2.slot_count() == 1, "slot count survives roundtrip");
    float lA_new = eval_loss(kexp2.forward_with_slots(a_in, a_pos), a_tgt);
    float lA_old = eval_loss(kexp.forward_with_slots(a_in, a_pos), a_tgt);
    TEST_CHECK(std::fabs(lA_new - lA_old) < 1e-4f,
               "loaded slots reproduce the forward");
    std::remove("test_slots.kexp");
}

int main(int argc, char** argv) {
    int only = (argc > 1) ? atoi(argv[1]) : 0;
    try {
        if (only == 0 || only == 1) test_selective();
        if (only == 0 || only == 2 || only == 4) test_rank_adapters();
        if (only == 0 || only == 3 || only == 4) test_knowledge_expansion();
    } catch (const std::exception& e) {
        fprintf(stderr, "\nEXCEPTION: %s\n", e.what());
        fflush(stderr);
        return 2;
    }
    return TEST_REPORT();
}
