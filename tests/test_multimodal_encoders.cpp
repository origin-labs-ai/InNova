// test_multimodal_encoders.cpp — modality encoders, tokenizer, MoE routers
#include "quant/multimodal.h"
#include "quant/tensor.h"
#include "quant/tokenizer.h"
#include "quant/test.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <string>

using namespace quant;

static bool all_finite(const Tensor& t) {
    if (t.numel() == 0) return true;
    const float* d = t.data<float>();
    for (int64_t i = 0; i < t.numel(); i++)
        if (!std::isfinite(d[i])) return false;
    return true;
}

static TransformerConfig make_cfg() {
    TransformerConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_size = 16;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.head_dim = 8;
    cfg.ffn_hidden_size = 32;
    cfg.max_seq_len = 32;
    return cfg;
}

int main() {
    fprintf(stderr, "M0 start\n");
    TEST_SUITE("multimodal_encoders");
    printf("=== Multimodal encoders test ===\n\n");

    TransformerConfig cfg = make_cfg();
    DenseModel enc(cfg);

    // ModalityEncoder: per-modality encode
    {
        fprintf(stderr, "M1 modality encoder\n");
        ModalityEncoder vision(&enc, "vision");
        ModalityEncoder audio(&enc, "audio");
        ModalityEncoder text(&enc, "text");
        Tensor input(Shape{4, 16}, DType::F32);
        input.fill(0.4f);
        Tensor vo = vision.encode(input);
        Tensor ao = audio.encode(input);
        Tensor to = text.encode(input);
        TEST_CHECK(vo.numel() > 0 && all_finite(vo), "vision encoder finite");
        TEST_CHECK(ao.numel() > 0 && all_finite(ao), "audio encoder finite");
        TEST_CHECK(to.numel() > 0 && all_finite(to), "text encoder finite");
    }

    // Cross-modal alignment: contrastive loss
    {
        fprintf(stderr, "M2 cross-modal alignment\n");
        CrossModalAlignment align(&enc, &enc, 0.07f);
        Tensor image_emb(Shape{4, 16}, DType::F32);
        Tensor text_emb(Shape{4, 16}, DType::F32);
        image_emb.fill(0.3f); text_emb.fill(0.3f);
        float loss = align.contrastive_loss(image_emb, text_emb);
        printf("  contrastive loss: %.4f\n", loss);
        TEST_CHECK(std::isfinite(loss), "contrastive loss finite");
        TEST_CHECK(loss >= 0.0f, "contrastive loss non-negative");
    }

    // Vision+Text and Audio+Text MoE
    {
        fprintf(stderr, "M3 vision-text MoE\n");
        VisionTextMoE vtm(16, 4);
        Tensor vision_tokens(Shape{4, 16}, DType::F32);
        Tensor text_tokens(Shape{4, 16}, DType::F32);
        vision_tokens.fill(0.2f); text_tokens.fill(0.3f);
        Tensor out = vtm.forward(vision_tokens, text_tokens);
        TEST_CHECK(out.numel() > 0 && all_finite(out), "vision-text MoE finite");

        AudioTextMoE atm(16, 4);
        Tensor audio_tokens(Shape{4, 16}, DType::F32);
        audio_tokens.fill(0.25f);
        Tensor aout = atm.forward(audio_tokens, text_tokens);
        TEST_CHECK(aout.numel() > 0 && all_finite(aout), "audio-text MoE finite");
    }

    // All-modality MoE
    {
        fprintf(stderr, "M4 all-modality MoE\n");
        AllModalityMoE amm(16, 6);
        Tensor v(Shape{4, 16}, DType::F32);
        Tensor a(Shape{4, 16}, DType::F32);
        Tensor t(Shape{4, 16}, DType::F32);
        v.fill(0.1f); a.fill(0.2f); t.fill(0.3f);
        Tensor out = amm.forward({v, a, t});
        TEST_CHECK(out.numel() > 0 && all_finite(out), "all-modality MoE finite");
    }

    // Speech recognition / OCR / video pipeline smoke
    {
        fprintf(stderr, "M5 speech/ocr/video\n");
        SpeechRecognizer sr(&enc);
        Tensor audio(Shape{1, 32}, DType::F32);
        audio.fill(0.1f);
        std::string trans = sr.transcribe(audio);
        TEST_CHECK(true, "speech recognizer runs");

        OCRPipeline ocr(&enc);
        Tensor image(Shape{1, 16}, DType::F32);
        image.fill(0.1f);
        std::string txt = ocr.recognize(image);
        TEST_CHECK(true, "OCR pipeline runs");

        VideoUnderstanding vu(&enc);
        Tensor frames(Shape{4, 16}, DType::F32);
        frames.fill(0.1f);
        std::string desc = vu.describe(frames);
        TEST_CHECK(true, "video understanding runs");
    }

    // Image captioning / Visual QA
    {
        fprintf(stderr, "M6 captioning/vqa\n");
        ImageCaptioning cap(&enc, &enc);
        Tensor image(Shape{1, 32}, DType::F32);
        image.fill(0.1f);
        std::string c = cap.caption(image, 8);
        TEST_CHECK(true, "image captioning runs");

        VisualQA vqa(&enc, &enc);
        std::string a = vqa.answer(image, "what is this?");
        TEST_CHECK(true, "visual QA runs");
    }

    // MultiModalTokenizer with BPE
    {
        fprintf(stderr, "M7 tokenizer\n");
        BPETokenizer bpe;
        MultiModalTokenizer tok(&bpe);
        auto ids = tok.encode("encode me");
        TEST_CHECK(!ids.empty(), "tokenizer encodes");
        auto img = tok.encode_image(Tensor(Shape{4, 4}, DType::F32));
        TEST_CHECK(!img.empty(), "image tokens produced");
        auto aud = tok.encode_audio(Tensor(Shape{4, 4}, DType::F32));
        TEST_CHECK(!aud.empty(), "audio tokens produced");
    }
    fprintf(stderr, "M8 done\n");

    int failures = TEST_REPORT();
    printf("\nMULTIMODAL ENCODERS TEST %s\n", failures == 0 ? "PASSED" : "FAILED");
    return failures > 0 ? 1 : 0;
}
