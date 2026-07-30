#include "oil/multimodal.h"
#include "oil/model.h"
#include "oil/tensor.h"
#include "oil/math.h"
#include "oil/types.h"
#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>
#include <string>
#include "oil/test.h"

using namespace oil;

static void test_cross_modal_attention() {
    TEST_SUITE("H1: CrossModalAttention");
    CrossModalAttention cma(32);
    std::vector<Tensor> modalities;
    modalities.push_back(Tensor({32}));
    modalities.push_back(Tensor({32}));
    modalities.push_back(Tensor({32}));
    modalities[0].fill(1.0f);
    modalities[1].fill(2.0f);
    modalities[2].fill(3.0f);

    auto out = cma.forward(modalities);
    TEST_CHECK(out.numel() == 32, "output shape matches hidden dim");

    auto out_empty = cma.forward({});
    TEST_CHECK(out_empty.numel() == 32, "empty modalities returns default tensor");

    auto out_single = cma.forward({modalities[0]});
    TEST_CHECK(out_single.numel() == 32, "single modality works");
}

static void test_joint_multimodal_model() {
    TEST_SUITE("H2: JointMultimodalModel");
    JointMultimodalModel jmm(nullptr, 64);
    Tensor text({1, 10, 64}), image({1, 8, 64}), audio({1, 5, 64});
    text.fill(1.0f); image.fill(2.0f); audio.fill(3.0f);
    auto out = jmm.forward(text, image, audio);
    TEST_CHECK(out.numel() > 0, "joint model produces output");
    TEST_CHECK(std::isfinite(out.data<float>()[0]), "output values are finite");
}

static void test_imagenet_classifier() {
    TEST_SUITE("H3: ImageNetClassifier");
    ImageNetClassifier inc(nullptr, 100);
    Tensor image({2, 768});
    image.fill(0.5f);
    auto logits = inc.classify(image);
    TEST_CHECK(logits.dim(0) == 2, "classifier outputs batch dim");
    TEST_CHECK(logits.dim(1) == 100, "classifier outputs 100 classes");

    for (int64_t i = 0; i < logits.numel(); i++)
        TEST_CHECK(std::isfinite(logits.data<float>()[i]), "classifier outputs are finite");
}

static void test_speech_recognizer() {
    TEST_SUITE("H4: SpeechRecognizer");
    SpeechRecognizer sr(nullptr);
    Tensor audio({1, 100, 40});
    audio.fill(0.5f);
    auto text = sr.transcribe(audio);
    TEST_CHECK(!text.empty(), "transcribe returns non-empty text");
}

static void test_ocr_pipeline() {
    TEST_SUITE("H5: OCRPipeline");
    OCRPipeline ocr(nullptr);
    Tensor image({1, 50, 256});
    image.fill(0.3f);
    auto text = ocr.recognize(image);
    TEST_CHECK(!text.empty(), "OCR returns non-empty text");
}

static void test_video_understanding() {
    TEST_SUITE("H6: VideoUnderstanding");
    VideoUnderstanding vu(nullptr);
    Tensor frames({1, 16, 768});
    frames.fill(0.5f);
    auto desc = vu.describe(frames);
    TEST_CHECK(!desc.empty(), "video description non-empty");
    TEST_CHECK(desc.find("[video") != std::string::npos, "description contains marker");
}

static void test_image_captioning() {
    TEST_SUITE("H7: ImageCaptioning");
    ImageCaptioning ic(nullptr, nullptr);
    Tensor image({1, 768});
    image.fill(1.0f);
    auto cap = ic.caption(image, 5);
    TEST_CHECK(!cap.empty(), "caption returns text");
}

static void test_visual_qa() {
    TEST_SUITE("H8: VisualQA");
    VisualQA vqa(nullptr, nullptr);
    Tensor image({1, 768});
    image.fill(1.0f);
    auto ans = vqa.answer(image, "what is this?");
    TEST_CHECK(!ans.empty(), "VQA returns answer");
}

static void test_text_to_image() {
    TEST_SUITE("H9: TextToImage");
    TextToImage t2i(64, 256);
    auto img = t2i.generate("a cat", 10);
    TEST_CHECK(img.numel() > 0, "text-to-image produces output");
    TEST_CHECK(img.dim(0) == 3, "image has 3 channels");
    TEST_CHECK(img.dim(1) == 256, "image height");
    TEST_CHECK(img.dim(2) == 256, "image width");
}

static void test_audio_synthesizer() {
    TEST_SUITE("H10: AudioSynthesizer");
    AudioSynthesizer as;
    Tensor mel({80, 50});
    mel.fill(0.5f);
    auto wav = as.synthesize(mel);
    TEST_CHECK(wav.numel() > 0, "synthesizer produces waveform");
    for (int64_t i = 0; i < wav.numel(); i++)
        TEST_CHECK(std::isfinite(wav.data<float>()[i]), "waveform values finite");
}

static void test_mel_spectrogram() {
    TEST_SUITE("H11: MelSpectrogram");
    MelSpectrogram ms(22050, 80);
    Tensor waveform({22050}); // 1 second
    for (int64_t i = 0; i < 22050; i++)
        waveform.data<float>()[i] = std::sin(2.0f * 3.14159f * 440.0f * i / 22050.0f);
    auto mel = ms.compute(waveform);
    TEST_CHECK(mel.dim(0) == 80, "mel spectrogram has 80 bands");
    TEST_CHECK(mel.dim(1) > 0, "mel spectrogram has frames");
    for (int64_t i = 0; i < mel.numel(); i++)
        TEST_CHECK(std::isfinite(mel.data<float>()[i]), "mel values finite");
}

static void test_audio_feature_extractor() {
    TEST_SUITE("H12: AudioFeatureExtractor");
    AudioFeatureExtractor afe(nullptr);
    Tensor audio({1, 1024});
    audio.fill(0.5f);
    auto feat = afe.extract(audio);
    TEST_CHECK(feat.numel() > 0, "extractor produces features");
}

static void test_multimodal_tokenizer() {
    TEST_SUITE("H13: MultiModalTokenizer");
    MultiModalTokenizer mmt;
    auto tokens = mmt.encode("hello world");
    TEST_CHECK(!tokens.empty(), "tokenizer encodes text");
    TEST_CHECK(mmt.vocab_size() == 32000, "default vocab size");
    TEST_CHECK(mmt.image_token_id() == 30000, "image token id");
    TEST_CHECK(mmt.audio_token_id() == 31000, "audio token id");

    auto decoded = mmt.decode(tokens);
    TEST_CHECK(!decoded.empty(), "tokenizer decodes text");

    auto img_tokens = mmt.encode_image(Tensor({10}));
    TEST_CHECK(!img_tokens.empty(), "image encoding works");
    TEST_CHECK(img_tokens[0] == 30000, "image encoding returns image token");

    auto aud_tokens = mmt.encode_audio(Tensor({10}));
    TEST_CHECK(!aud_tokens.empty(), "audio encoding works");
    TEST_CHECK(aud_tokens[0] == 31000, "audio encoding returns audio token");
}

static void test_modality_encoder() {
    TEST_SUITE("H14: ModalityEncoder");
    ModalityEncoder me(nullptr, "text");
    Tensor inp({1, 10});
    inp.fill(1.0f);
    auto encoded = me.encode(inp);
    TEST_CHECK(encoded.numel() > 0, "modality encoder works");
}

static void test_cross_modal_alignment() {
    TEST_SUITE("H15: CrossModalAlignment");
    CrossModalAlignment cma(nullptr, nullptr, 0.07f);
    int64_t B=4, D=16;
    Tensor img_emb({B, D}), txt_emb({B, D});
    img_emb.fill(1.0f);
    txt_emb.fill(1.0f);
    float loss = cma.contrastive_loss(img_emb, txt_emb);
    TEST_CHECK(loss > 0, "contrastive loss positive");
    TEST_CHECK(std::isfinite(loss), "contrastive loss finite");

    auto aligned = cma.align(img_emb, txt_emb);
    TEST_CHECK(aligned.dim(0) == B, "aligned output batch dim");
    TEST_CHECK(aligned.dim(1) == D, "aligned output feature dim");
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("InNova — Multi-Modal (H1-H15) Test Suite\n");
    printf("============================================\n");

    test_cross_modal_attention();
    test_joint_multimodal_model();
    test_imagenet_classifier();
    test_speech_recognizer();
    test_ocr_pipeline();
    test_video_understanding();
    test_image_captioning();
    test_visual_qa();
    test_text_to_image();
    test_audio_synthesizer();
    test_mel_spectrogram();
    test_audio_feature_extractor();
    test_multimodal_tokenizer();
    test_modality_encoder();
    test_cross_modal_alignment();

    printf("\n============================================\n");

    return TEST_REPORT() > 0 ? 1 : 0;
}
