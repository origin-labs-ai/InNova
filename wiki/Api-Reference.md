# API Reference

All APIs are in the `oil` namespace.

## Core Types (`types.h`)

```cpp
enum class Format : uint8_t {
    OIL1            = 0,  // 1.00 BPW, 1 centroid
    SPARK_Q0        = 1,  // 1.50 BPW, sign + FP16 scale (lossy)
    OIL2            = 2,  // 2.00 BPW, 4 centroids Lloyd-Max (lossy)
    OIL4            = 3,  // 4.00 BPW, 16 centroids Lloyd-Max (lossy)
    OIL8            = 4,  // 8.00 BPW, 256 centroids Lloyd-Max (lossy)
    OIL16           = 5,  // 16.00 BPW, FP16 storage (lossy)
    OIL32           = 6,  // 32.00 BPW, FP32 identity (lossless)
    OIL1_GRP        = 7,  // ~1.19 BPW, lossy grouped
    SPARK_Q0_GRP    = 8,  // ~1.69 BPW, lossy grouped
    OIL2_GRP        = 9,  // ~2.19 BPW, lossy grouped
    OIL4_GRP        = 10, // ~4.19 BPW, lossy grouped
    OIL8_GRP        = 11, // ~8.19 BPW, lossy grouped
    OIL16_GRP       = 12, // ~16.19 BPW, lossy grouped
    SPARK_SPARSE     = 13, // variable BPW, lossy sparse
    SPARK_SPARSE_GRP = 14, // variable BPW, lossy grouped sparse
};

enum class DType : uint8_t {
    I64, U8, U4, I2, I1, F16, F32
};

struct Shape {
    int64_t dims[4];
    int rank;
    int64_t numel() const;
};

struct TransformerConfig {
    int64_t vocab_size = 32000;
    int64_t hidden_size = 768;
    int64_t num_layers = 12;
    int64_t num_heads = 12;
    int64_t intermediate_size = 3072;
    int64_t max_seq_len = 2048;
    float norm_eps = 1e-5f;
    Format weight_format = Format::FP16;
    bool use_moe = false;
    int64_t num_experts = 0;
    int64_t top_k = 0;
};
```

## Tensor (`tensor.h`)

```cpp
class Tensor {
    // Construction
    Tensor();
    Tensor(Shape shape, DType dtype = DType::F32);
    Tensor(Shape shape, std::shared_ptr<Buffer> buffer, DType dtype);

    // Shape
    const Shape& shape() const;
    int64_t dim(int i) const;
    int rank() const;
    int64_t numel() const;
    int64_t size_bytes() const;

    // Data
    DType dtype() const;
    void* data();
    template<typename T> T* data();
    
    // Factories
    static Tensor zeros(Shape shape);
    static Tensor ones(Shape shape);
    static Tensor arange(int64_t n);

    // Operations
    Tensor view(Shape new_shape) const;
    Tensor reshape(Shape new_shape) const;
    Tensor slice(int dim, int64_t start, int64_t end) const;
    Tensor transpose(int dim1, int dim2) const;
    void fill(float value);
    Tensor clone() const;
    void copy_from(const Tensor& src);
    void copy_to(Tensor& dst) const;

    // Autograd
    bool requires_grad() const;
    void requires_grad(bool enabled);
    Tensor& grad() const;

    // Indexing
    std::variant<float, Tensor> operator[](int64_t index) const;
};
```

## Autograd (`autograd.h`)

```cpp
class AutogradEngine {
    static AutogradEngine& instance();
    Tensor forward(Tensor& input, GradientFn fn);
    void backward(Tensor& loss);
    void zero_grad();
};
```

## Model (`model.h`)

```cpp
class Model {
    TransformerConfig config;
    virtual Tensor forward(const Tensor& input_ids, 
                           const Tensor& positions,
                           KVCache* cache = nullptr) = 0;
    virtual void load(const std::string& path);
    virtual void save(const std::string& path) const;
    virtual int64_t param_count() const = 0;
    virtual int64_t vocab_size() const = 0;
};

class DenseModel : public Model { /* ... */ };
```

## Math (`math.h`)

```cpp
namespace oil::math {
    void matmul(Tensor& out, const Tensor& a, const Tensor& b);
    void rms_norm(Tensor& out, const Tensor& x, const Tensor& weight);
    void softmax(Tensor& out, const Tensor& x);
    void silu(Tensor& out, const Tensor& x);
    void gelu(Tensor& out, const Tensor& x);
    void rope(Tensor& out, const Tensor& x, int dim);
    void add(Tensor& out, const Tensor& a, const Tensor& b);
    void mul(Tensor& out, const Tensor& a, const Tensor& b);
    void cat(Tensor& out, const std::vector<Tensor>& tensors, int dim);
    void gather(Tensor& out, const Tensor& x, const Tensor& indices);
    void scatter_add(Tensor& out, const Tensor& x, 
                     const Tensor& indices, const Tensor& updates);
    float mean(const Tensor& x);
    float var(const Tensor& x);
}
```

## Trainer (`trainer.h`)

```cpp
struct TrainerConfig {
    int64_t batch_size = 8;
    int64_t epochs = 10;
    int64_t warmup_steps = 100;
    float learning_rate = 3e-4f;
    float weight_decay = 0.1f;
    float grad_clip = 1.0f;
};

class Trainer {
    Trainer(const TransformerConfig& model_cfg, 
            const TrainerConfig& train_cfg);
    void train();
    void train_step(const Tensor& batch);
    void save_checkpoint(const std::string& path);
    float get_loss() const;
};
```

## Tokenizer (`tokenizer.h`)

```cpp
class Tokenizer {
    Tokenizer(const std::string& vocab_path);
    Tokenizer(const std::vector<std::string>& vocab);
    std::vector<int64_t> encode(const std::string& text);
    std::string decode(const std::vector<int64_t>& ids);
    int vocab_size() const;
    int bos_token() const;
    int eos_token() const;
    int pad_token() const;
};
```

## Format I/O (`oil_format.h`)

```cpp
class OILReader {
    OILReader(const std::string& path);
    OILHeader read_header();
    TransformerConfig read_config();
    Tensor read_tensor(const std::string& name);
    std::vector<std::string> list_tensors();
};

class OILWriter {
    OILWriter(const std::string& path, const TransformerConfig& config);
    void write_tensor(const std::string& name, const Tensor& tensor);
    void write_config(const TransformerConfig& config);
    void finalize();
};
```

## Generator (`generator.h`)

```cpp
class Generator {
    Generator(std::unique_ptr<Model> model, 
              std::unique_ptr<Tokenizer> tokenizer);
    std::string generate(const std::string& prompt, int max_tokens = 256);
    std::vector<int64_t> generate_ids(const std::vector<int64_t>& input_ids,
                                       int max_tokens = 256);
    void stream_generate(const std::string& prompt,
                          std::function<void(const std::string&)> callback);
};
```

## Sampler (`sampler.h`)

```cpp
class Sampler {
    Sampler(float temperature = 1.0f, float top_p = 0.95f, 
            int top_k = 40, int64_t seed = -1);
    int64_t sample(const Tensor& logits);
    int64_t greedy(const Tensor& logits);
};
```

## Full Headers Reference

For complete API details, see:
- [File Documentation Index](files/_index) — per-file docs for all headers
- Header files: `include/oil/*.h` — inline documentation in source
