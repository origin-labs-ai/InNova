#include "oil/multimodal.h"
#include "oil/model.h"
#include "oil/tensor.h"
#include "oil/math.h"
#include "oil/types.h"
#include "oil/transformer.h"
#include "oil/tokenizer.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include "oil/test.h"

using namespace oil;

// ViT patch embed simulation via ModalityEncoder
static void test_vision_encoder() {
    TEST_SUITE("Vision Encoder (ViT patch embed)");
    ModalityEncoder ve(nullptr, "vision");
    Tensor input({1, 196, 768});
    for (int64_t i = 0; i < input.numel(); i++)
        input.data<float>()[i] = std::sin((float)i * 0.01f);
    auto out = ve.encode(input);
    TEST_CHECK(out.numel() > 0, "vision encoder produces output");
    TEST_CHECK(out.dim(0) == 1, "vision encoder preserves batch dim");
    TEST_CHECK(out.dim(1) == 768, "vision encoder preserves feature dim");
    for (int64_t i = 0; i < out.numel(); i++)
        TEST_CHECK(std::isfinite(out.data<float>()[i]), "vision encoder outputs finite");

    // Test with random input
    Tensor input2({2, 50, 768});
    input2.fill(0.5f);
    auto out2 = ve.encode(input2);
    TEST_CHECK(out2.dim(0) == 2, "vision encoder batch=2 works");
    TEST_CHECK(out2.dim(1) == 768, "vision encoder features preserved for batch=2");
}

// Audio encoder via MelSpectrogram + ModalityEncoder
static void test_audio_encoder() {
    TEST_SUITE("Audio Encoder (mel spectrogram + encode)");
    MelSpectrogram ms(16000, 80);
    int64_t samples = 16000;
    Tensor waveform({samples});
    for (int64_t i = 0; i < samples; i++)
        waveform.data<float>()[i] = std::sin(2.0f * 3.14159f * 440.0f * i / 16000.0f)
                                  + 0.5f * std::sin(2.0f * 3.14159f * 880.0f * i / 16000.0f);
    auto mel = ms.compute(waveform);
    TEST_CHECK(mel.dim(0) == 80, "mel spectrogram has 80 bands");
    TEST_CHECK(mel.dim(1) > 0, "mel spectrogram has frames");
    for (int64_t i = 0; i < mel.numel(); i++)
        TEST_CHECK(std::isfinite(mel.data<float>()[i]), "mel values finite");

    // Test different sample rate
    MelSpectrogram ms2(44100, 128);
    Tensor waveform2({44100});
    for (int64_t i = 0; i < 44100; i++)
        waveform2.data<float>()[i] = std::sin(2.0f * 3.14159f * 1000.0f * i / 44100.0f);
    auto mel2 = ms2.compute(waveform2);
    TEST_CHECK(mel2.dim(0) == 128, "mel 44100/128 has 128 bands");
    TEST_CHECK(mel2.dim(1) > 0, "mel 44100/128 has frames");
    TEST_CHECK(std::isfinite(mel2.data<float>()[0]), "mel2 value finite");

    // ModalityEncoder for audio features
    ModalityEncoder ae(nullptr, "audio");
    Tensor audio_feat({1, 80, 50});
    audio_feat.fill(0.5f);
    auto encoded = ae.encode(audio_feat);
    TEST_CHECK(encoded.numel() > 0, "audio encoder produces output");

    // AudioFeatureExtractor
    AudioFeatureExtractor afe(nullptr);
    Tensor raw_audio({1, 1024});
    raw_audio.fill(0.3f);
    auto feat = afe.extract(raw_audio);
    TEST_CHECK(feat.numel() > 0, "audio feature extractor works");
}

// Image encoder via ModalityEncoder + ImageNetClassifier
static void test_image_encoder() {
    TEST_SUITE("Image Encoder (CNN frontend)");
    // ImageNetClassifier as ViT proxy
    ImageNetClassifier inc(nullptr, 1000);
    Tensor image({4, 768});
    for (int64_t i = 0; i < image.numel(); i++)
        image.data<float>()[i] = (float)(i % 10) / 10.0f;
    auto logits = inc.classify(image);
    TEST_CHECK(logits.dim(0) == 4, "image classifier preserves batch=4");
    TEST_CHECK(logits.dim(1) == 1000, "image classifier outputs 1000 classes");
    for (int64_t i = 0; i < logits.numel(); i++)
        TEST_CHECK(std::isfinite(logits.data<float>()[i]), "image classifier outputs finite");

    // Test with single image
    ImageNetClassifier inc2(nullptr, 10);
    Tensor single({1, 768});
    single.fill(0.5f);
    auto logits2 = inc2.classify(single);
    TEST_CHECK(logits2.dim(1) == 10, "image classifier 10 classes works");

    // ModalityEncoder for image
    ModalityEncoder ie(nullptr, "image");
    Tensor img_input({1, 224, 768});
    img_input.fill(0.5f);
    auto img_out = ie.encode(img_input);
    TEST_CHECK(img_out.numel() > 0, "image modality encoder works");
}

// Video encoder via VideoUnderstanding
static void test_video_encoder() {
    TEST_SUITE("Video Encoder (3D conv)");
    VideoUnderstanding vu(nullptr);
    Tensor frames({1, 16, 768});
    for (int64_t i = 0; i < frames.numel(); i++)
        frames.data<float>()[i] = std::sin((float)i * 0.05f);
    auto desc = vu.describe(frames);
    TEST_CHECK(!desc.empty(), "video description non-empty");

    // Test with longer sequence
    Tensor frames2({1, 32, 768});
    frames2.fill(0.3f);
    auto desc2 = vu.describe(frames2);
    TEST_CHECK(!desc2.empty(), "video description for longer seq works");

    // Test with single frame
    Tensor frames3({1, 1, 768});
    frames3.fill(0.1f);
    auto desc3 = vu.describe(frames3);
    TEST_CHECK(!desc3.empty(), "video description for single frame works");
}

// Text encoder via ModalityEncoder + BPETokenizer
static void test_text_encoder() {
    TEST_SUITE("Text Encoder");
    ModalityEncoder te(nullptr, "text");
    Tensor text({1, 32, 768});
    for (int64_t i = 0; i < text.numel(); i++)
        text.data<float>()[i] = (float)(i % 128) / 128.0f;
    auto out = te.encode(text);
    TEST_CHECK(out.numel() > 0, "text encoder produces output");
    TEST_CHECK(out.dim(0) == 1, "text encoder preserves batch");

    // Test with batch=2
    Tensor text2({2, 16, 768});
    text2.fill(0.5f);
    auto out2 = te.encode(text2);
    TEST_CHECK(out2.dim(0) == 2, "text encoder batch=2 works");

    // Test MultiModalTokenizer (text tokenization)
    MultiModalTokenizer mmt;
    auto tokens = mmt.encode("hello world this is a test");
    TEST_CHECK(!tokens.empty(), "multimodal tokenizer encodes text");
    TEST_CHECK(mmt.vocab_size() == 32000, "default vocab size");
    TEST_CHECK(mmt.image_token_id() == 30000, "image token id");
    TEST_CHECK(mmt.audio_token_id() == 31000, "audio token id");

    auto decoded = mmt.decode(tokens);
    TEST_CHECK(!decoded.empty(), "multimodal tokenizer decodes");

    // BPETokenizer integration
    BPETokenizer bpe;
    MultiModalTokenizer mmt2(&bpe);
    auto tok2 = mmt2.encode("test");
    TEST_CHECK(mmt2.vocab_size() == bpe.vocab_size(), "bpe-integrated vocab size");
}

// OCR encoder via OCRPipeline
static void test_ocr_encoder() {
    TEST_SUITE("OCR Encoder");
    OCRPipeline ocr(nullptr);
    Tensor image({1, 50, 256});
    for (int64_t i = 0; i < image.numel(); i++)
        image.data<float>()[i] = (float)(i % 100) / 100.0f;
    auto text = ocr.recognize(image);
    TEST_CHECK(!text.empty(), "OCR encoder returns text");

    // Test with wider image
    Tensor image2({1, 100, 512});
    image2.fill(0.3f);
    auto text2 = ocr.recognize(image2);
    TEST_CHECK(!text2.empty(), "OCR encoder handles wide image");
}

// Embeddings and contrastive loss via CrossModalAlignment
static void test_embeddings_contrastive() {
    TEST_SUITE("Embeddings (contrastive loss)");
    CrossModalAlignment cma(nullptr, nullptr, 0.07f);
    int64_t B = 4, D = 16;

    // Similar pairs should have lower loss
    Tensor img_emb({B, D}), txt_emb({B, D});
    for (int64_t i = 0; i < B; i++)
        for (int64_t d = 0; d < D; d++) {
            img_emb.data<float>()[i * D + d] = std::sin((float)(i * D + d) * 0.1f);
            txt_emb.data<float>()[i * D + d] = img_emb.data<float>()[i * D + d];
        }
    float loss_same = cma.contrastive_loss(img_emb, txt_emb);
    TEST_CHECK(loss_same > 0, "contrastive loss positive for identical");
    TEST_CHECK(std::isfinite(loss_same), "contrastive loss finite");
    printf("    loss (identical): %f\n", loss_same);

    // Different pairs should have higher loss
    Tensor img_emb2({B, D}), txt_emb2({B, D});
    for (int64_t i = 0; i < B; i++)
        for (int64_t d = 0; d < D; d++) {
            img_emb2.data<float>()[i * D + d] = (float)(i * D + d) / (float)(B * D);
            txt_emb2.data<float>()[i * D + d] = 1.0f - (float)(i * D + d) / (float)(B * D);
        }
    float loss_diff = cma.contrastive_loss(img_emb2, txt_emb2);
    TEST_CHECK(loss_diff > 0, "contrastive loss positive for different");
    TEST_CHECK(std::isfinite(loss_diff), "contrastive loss finite for different");

    // Alignment produces valid output
    auto aligned = cma.align(img_emb, txt_emb);
    TEST_CHECK(aligned.dim(0) == B, "aligned output batch dim");
    TEST_CHECK(aligned.dim(1) == D, "aligned output feature dim");
    for (int64_t i = 0; i < aligned.numel(); i++)
        TEST_CHECK(std::isfinite(aligned.data<float>()[i]), "aligned values finite");

    // Test with larger batch
    int64_t B2 = 8;
    Tensor img_big({B2, D}), txt_big({B2, D});
    img_big.fill(0.5f);
    txt_big.fill(0.5f);
    float loss_big = cma.contrastive_loss(img_big, txt_big);
    TEST_CHECK(std::isfinite(loss_big), "contrastive loss batch=8 finite");
}

// Cross-modal attention test
static void test_cross_modal_attention_detail() {
    TEST_SUITE("CrossModal Attention");
    CrossModalAttention cma(64);

    // Two modalities
    std::vector<Tensor> mods;
    mods.push_back(Tensor({10, 64}));
    mods.push_back(Tensor({5, 64}));
    mods[0].fill(1.0f);
    mods[1].fill(2.0f);
    auto out = cma.forward(mods);
    TEST_CHECK(out.numel() == 640, "cross-modal output shape {10, 64}");

    // Three modalities
    std::vector<Tensor> mods3;
    mods3.push_back(Tensor({8, 64}));
    mods3.push_back(Tensor({6, 64}));
    mods3.push_back(Tensor({4, 64}));
    mods3[0].fill(0.5f);
    mods3[1].fill(1.0f);
    mods3[2].fill(1.5f);
    auto out3 = cma.forward(mods3);
    TEST_CHECK(out3.numel() == 512, "cross-modal 3-mod output shape {8, 64}");

    // Edge: empty
    auto empty_out = cma.forward({});
    TEST_CHECK(empty_out.numel() == 64, "cross-modal empty returns default {64}");

    // Single modality
    auto single_out = cma.forward({mods[0]});
    TEST_CHECK(single_out.numel() == 640, "cross-modal single returns clone");
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("MYTHOS.cpp — Multi-Modal Encoder Test Suite\n");
    printf("============================================\n");

    test_vision_encoder();
    test_audio_encoder();
    test_image_encoder();
    test_video_encoder();
    test_text_encoder();
    test_ocr_encoder();
    test_embeddings_contrastive();
    test_cross_modal_attention_detail();

    printf("\n============================================\n");

    return TEST_REPORT() > 0 ? 1 : 0;
}
