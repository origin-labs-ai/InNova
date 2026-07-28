cmake_minimum_required(VERSION 3.20)
project(MYTHOS.cpp VERSION 0.1.02 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CTest)
include(arch)
include(compiler)

option(OIL_BUILD_TESTS "Build unit tests" ON)
option(OIL_BUILD_BENCHMARKS "Build benchmarks" ON)
option(OIL_BUILD_TOOLS "Build CLI tools" ON)
option(OIL_SANITIZE "Enable address sanitizer" OFF)
option(OIL_COVERAGE "Enable code coverage (GCC/Clang only)" OFF)

if(OIL_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    message(STATUS "Code coverage enabled (--coverage)")
  else()
    message(WARNING "OIL_COVERAGE only supported with GCC or Clang")
  endif()
endif()

set(OIL_COVERAGE_LCOV_DIR "${CMAKE_BINARY_DIR}/coverage" CACHE PATH
  "Directory for lcov coverage reports")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/oil_config.h.in"
  "${CMAKE_BINARY_DIR}/oil_config.h"
)

add_library(oil_config INTERFACE)
target_include_directories(oil_config INTERFACE "${CMAKE_BINARY_DIR}")

add_library(oil_core
  src/tensor.cpp
  src/memory.cpp
  src/random.cpp
  src/tensor_view.cpp
  src/data_gen.cpp
  src/metrics.cpp
  src/parquet_streamer.cpp
  src/telemetry.cpp
)
target_include_directories(oil_core PUBLIC include)
target_link_libraries(oil_core PUBLIC oil_config)

add_library(oil_math
  src/math.cpp
  src/math_avx2.cpp
  src/math_avx2_tiled.cpp
  src/math_avx2_tensor.cpp
  src/math_avx2_vec.cpp
  src/simd_math.cpp
)
if(OIL_AVX512)
  target_sources(oil_math PRIVATE src/math_avx512.cpp)
endif()
target_include_directories(oil_math PUBLIC include)
target_link_libraries(oil_math PUBLIC oil_core)

add_library(oil_format
  src/oil_format.cpp
  src/codebook.cpp
  src/format_planner.cpp
  src/format_registry.cpp
  src/vector_quantizer.cpp
  src/quant_store.cpp
  src/oil_engines_core.cpp
  src/oil_engines_fp.cpp
  src/oil_engines_oil1.cpp
  src/oil_engines_oil2.cpp
  src/oil_engines_oil4.cpp
  src/oil_engines_oil8.cpp
  src/oil_engines_quant.cpp
  src/oil_engines_spark.cpp
)
target_include_directories(oil_format PUBLIC include)
target_link_libraries(oil_format PUBLIC oil_math)

add_library(oil_kernel
  src/kernel_i2s.cpp
  src/kernel_tl.cpp
  src/kernel_oil8.cpp
  src/kernel_oil4.cpp
  src/int8_quant.cpp
  src/kernel_production.cpp
)
target_include_directories(oil_kernel PUBLIC include)
target_link_libraries(oil_kernel PUBLIC oil_format)

add_library(oil_model
  src/transformer.cpp
  src/model.cpp
  src/kv_cache.cpp
  src/kv_cache_oil4.cpp
  src/autograd_engine.cpp
  src/autograd_functions.cpp
  src/autograd_grad.cpp
  src/flash_attention.cpp
)
target_include_directories(oil_model PUBLIC include)
target_link_libraries(oil_model PUBLIC oil_kernel)

add_library(oil_inference
  src/sampler.cpp
  src/generator.cpp
)
target_include_directories(oil_inference PUBLIC include)
target_link_libraries(oil_inference PUBLIC oil_model)

add_library(oil_tokenizer
  src/bpe_tokenizer_advanced.cpp
  src/bpe_tokenizer_bpe.cpp
  src/bpe_tokenizer_unicode.cpp
  src/dataset.cpp
)
target_include_directories(oil_tokenizer PUBLIC include)
target_link_libraries(oil_tokenizer PUBLIC oil_core)

add_library(oil_trainer
  src/optimizer.cpp
  src/scheduler.cpp
  src/eval.cpp
  src/ste_quantizer.cpp
  src/finetune.cpp
  src/log_writer.cpp
  src/distributed.cpp
  src/native_weight.cpp
  src/native_trainer.cpp
  src/hf_streamer.cpp
  src/moe_trainer.cpp
  src/hpo_nas.cpp
  src/ddp.cpp
  src/pipeline.cpp
  src/reward.cpp
  src/trainer_core.cpp
  src/trainer_data.cpp
  src/trainer_rl.cpp
  src/trainer_rl_ops.cpp
  src/training_utils.cpp
  src/zero_offload.cpp
  src/zero_optimizer_core.cpp
  src/zero_optimizer_mem.cpp
)
target_include_directories(oil_trainer PUBLIC include)
target_link_libraries(oil_trainer PUBLIC oil_model oil_inference oil_tokenizer)

add_library(oil_engine
  engines/inference/inference.cpp
  engines/inference/stream.cpp
  engines/oil/quantize.cpp
  engines/oil/codec.cpp
  src/code_gen.cpp
)
target_include_directories(oil_engine PUBLIC include engines/inference engines/oil)
target_link_libraries(oil_engine PUBLIC oil_inference oil_tokenizer)



add_library(oil_dense
  engines/trainer/dense/trainer.cpp
  engines/trainer/dense/dataloader.cpp
  engines/trainer/dense/checkpoint.cpp
)
target_include_directories(oil_dense PUBLIC include engines/trainer/dense)
target_link_libraries(oil_dense PUBLIC oil_trainer)

add_library(oil_moe
  engines/trainer/moe/moe.cpp
  engines/trainer/moe/vision/vision.cpp
  engines/trainer/moe/audio/audio.cpp
  engines/trainer/moe/embeddings/embeddings.cpp
  engines/trainer/moe/image/vision.cpp
  engines/trainer/moe/ocr/ocr.cpp
  engines/trainer/moe/text/multimodal_text.cpp
  engines/trainer/moe/video/video.cpp
)
target_include_directories(oil_moe PUBLIC include
  engines/trainer/moe
  engines/trainer/moe/vision
  engines/trainer/moe/audio
  engines/trainer/moe/embeddings
  engines/trainer/moe/image
  engines/trainer/moe/ocr
  engines/trainer/moe/text
  engines/trainer/moe/video
)
target_link_libraries(oil_moe PUBLIC oil_model)

# DirectX GPU compute library (Windows-only, needs D3D12)
if(WIN32)
  add_library(oil_gpu src/gpu_compute.cpp src/gpu_compute_full.cpp src/gpu_compute_vulkan.cpp src/igpu_zero_copy.cpp)
  target_include_directories(oil_gpu PUBLIC include)
  target_link_libraries(oil_gpu PUBLIC oil_core)
  target_link_libraries(oil_gpu PUBLIC d3d12 dxgi d3dcompiler)
  target_compile_definitions(oil_gpu PRIVATE OIL_USE_DIRECTX)
  target_sources(oil_gpu PRIVATE src/gpu_extras.cpp)
else()
  add_library(oil_gpu INTERFACE)
  target_include_directories(oil_gpu INTERFACE include)
endif()

# Hardware backend abstraction
add_library(oil_backend
  src/backend.cpp
)
target_include_directories(oil_backend PUBLIC include)
if(WIN32)
  target_link_libraries(oil_backend PUBLIC oil_math oil_gpu)
else()
  target_link_libraries(oil_backend PUBLIC oil_math)
endif()

# All MoE variants
add_library(oil_moe_variants
  src/moe_variants.cpp
  src/moe_model.cpp
  src/expert_parallel.cpp
  src/moe_advanced_support.cpp
  src/moe_quantized_test_core.cpp
  src/moe_quantized_test_extra.cpp
  src/mommoe_block.cpp
  src/native_oil_moe.cpp
)
target_include_directories(oil_moe_variants PUBLIC include)
target_link_libraries(oil_moe_variants PUBLIC oil_model oil_backend)

# Distributed training (tensor parallelism, FSDP)
add_library(oil_distributed
  src/tensor_parallelism.cpp
  src/fsdp.cpp
)
target_include_directories(oil_distributed PUBLIC include)
target_link_libraries(oil_distributed PUBLIC oil_model oil_kernel)

# Cat D: Inference optimization
add_library(oil_inference_opt
  src/inference_opt.cpp
)
target_include_directories(oil_inference_opt PUBLIC include)
target_link_libraries(oil_inference_opt PUBLIC oil_model oil_inference)

# Cat F: MoE enhancement
add_library(oil_moe_enhance
  src/moe_enhance.cpp
)
target_include_directories(oil_moe_enhance PUBLIC include)
target_link_libraries(oil_moe_enhance PUBLIC oil_model oil_moe_variants)

# Cat G: ASI Pipeline
add_library(oil_asi
  src/asi.cpp
  src/asi_flywheel.cpp
  src/asi_extended.cpp
  src/meta_cognition.cpp
  src/self_eval.cpp
  src/world_model.cpp
  src/multi_agent.cpp
)
target_include_directories(oil_asi PUBLIC include)
target_link_libraries(oil_asi PUBLIC oil_trainer oil_model)

# Cat H: Multi-Modal
add_library(oil_multimodal
  src/multimodal.cpp
  src/multimodal_cross_attn.cpp
  src/multimodal_fusion.cpp
  engines/trainer/multimodal/vision/vision.cpp
  engines/trainer/multimodal/image/image.cpp
  engines/trainer/multimodal/video/video.cpp
  engines/trainer/multimodal/text/text.cpp
  engines/trainer/multimodal/ocr/ocr.cpp
  engines/trainer/multimodal/embeddings/embeddings.cpp
  engines/trainer/multimodal/audio/audio.cpp
)
target_include_directories(oil_multimodal PUBLIC include
  engines/trainer/multimodal
  engines/trainer/multimodal/vision
  engines/trainer/multimodal/image
  engines/trainer/multimodal/video
  engines/trainer/multimodal/text
  engines/trainer/multimodal/ocr
  engines/trainer/multimodal/embeddings
  engines/trainer/multimodal/audio
)
target_link_libraries(oil_multimodal PUBLIC oil_model)
target_link_libraries(oil_multimodal PUBLIC oil_tokenizer oil_trainer)

# Cat I: Production (HTTP server, logging, config, C API, plugins)
add_library(oil_production
  src/production_core.cpp
  src/production_gen.cpp
  src/production_api.cpp
  src/production_gen_ops.cpp
  src/http_server.cpp
  src/json_parser.cpp
)
target_include_directories(oil_production PUBLIC include)
target_link_libraries(oil_production PUBLIC oil_model oil_inference oil_tokenizer)
if(WIN32)
  target_link_libraries(oil_production PUBLIC ws2_32)
endif()

# Vulkan compute backend (dynamically loaded at runtime, no SDK required)
option(OIL_VULKAN "Enable Vulkan compute backend (dynamically loaded)" ON)
if(OIL_VULKAN AND WIN32)
  target_compile_definitions(oil_gpu PRIVATE OIL_USE_VULKAN)
endif()

if(OIL_BUILD_TOOLS)
  add_executable(oil_infer tools/infer.cpp)
  target_include_directories(oil_infer PRIVATE engines/inference)
  target_link_libraries(oil_infer PRIVATE oil_engine oil_tokenizer)

  add_executable(oil_train tools/train.cpp)
  target_link_libraries(oil_train PRIVATE oil_dense oil_tokenizer)

  add_executable(train_64m tools/train_64m.cpp)
  target_link_libraries(train_64m PRIVATE oil_dense oil_trainer oil_tokenizer)

  add_executable(oil_finetune tools/finetune.cpp)
  target_link_libraries(oil_finetune PRIVATE oil_trainer)

  add_executable(oil_convert tools/convert.cpp)
  target_link_libraries(oil_convert PRIVATE oil_format)

  add_executable(oil_bench tools/bench.cpp)
  target_link_libraries(oil_bench PRIVATE oil_kernel oil_inference oil_tokenizer)

  add_executable(oil_info tools/info.cpp)
  target_link_libraries(oil_info PRIVATE oil_format)

  add_executable(oil_format_list tools/format_list.cpp)
  target_link_libraries(oil_format_list PRIVATE oil_format)

  add_executable(oil_serve tools/serve.cpp)
  target_link_libraries(oil_serve PRIVATE oil_production oil_model oil_tokenizer)

  add_executable(oil_quantize tools/quantize.cpp)
  target_link_libraries(oil_quantize PRIVATE oil_format oil_kernel)

  add_executable(oil_evaluate tools/evaluate.cpp)
  target_link_libraries(oil_evaluate PRIVATE oil_model oil_inference oil_tokenizer oil_trainer)
endif()

if(OIL_BUILD_TESTS)
  function(add_oil_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE include)
    target_link_libraries(${name} PRIVATE ${ARGN})
    add_test(NAME ${name} COMMAND $<TARGET_FILE:${name}>)
    set_tests_properties(${name} PROPERTIES WORKING_DIRECTORY $<TARGET_FILE_DIR:${name}>)
  endfunction()

  include(CTest)
  add_subdirectory(tests)

  if(OIL_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    find_program(LCOV_PATH lcov)
    find_program(GENHTML_PATH genhtml)

    if(LCOV_PATH AND GENHTML_PATH)
      add_custom_target(coverage
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OIL_COVERAGE_LCOV_DIR}
        COMMAND ${LCOV_PATH} --directory ${CMAKE_BINARY_DIR} --zerocounters
        COMMAND ${CMAKE_COMMAND} -E echo "-- Running all tests for coverage..."
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -j $ENV{NPROC}
        COMMAND ${LCOV_PATH} --directory ${CMAKE_BINARY_DIR} --capture
                 --output-file ${OIL_COVERAGE_LCOV_DIR}/coverage.info
        COMMAND ${LCOV_PATH} --remove ${OIL_COVERAGE_LCOV_DIR}/coverage.info
                 '/usr/*' '/opt/*' '${CMAKE_BINARY_DIR}/_deps/*'
                 '*/tests/*' '*/bench/*' '*/tools/*'
                 --output-file ${OIL_COVERAGE_LCOV_DIR}/coverage_filtered.info
        COMMAND ${GENHTML_PATH} ${OIL_COVERAGE_LCOV_DIR}/coverage_filtered.info
                 --output-directory ${OIL_COVERAGE_LCOV_DIR}/html
                 --title "MYTHOS.cpp Coverage"
                 --legend
        COMMENT "Generating code coverage report in ${OIL_COVERAGE_LCOV_DIR}/html"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      )
      message(STATUS "Coverage target available: cmake --build . --target coverage")
    else()
      message(WARNING "lcov/genhtml not found; coverage target not available")
    endif()
  endif()
endif()

if(OIL_BUILD_BENCHMARKS)
  macro(add_oil_bench name src deps)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE include)
    target_link_libraries(${name} PRIVATE ${deps})
  endmacro()
  add_oil_bench(bench_kernels   bench/bench_kernels.cpp   oil_kernel oil_math)
  add_oil_bench(bench_inference bench/bench_inference.cpp oil_inference oil_model oil_math oil_core)
  add_executable(bench_quality bench/bench_quality.cpp)
  target_include_directories(bench_quality PRIVATE include)
  target_link_libraries(bench_quality PRIVATE oil_inference oil_tokenizer oil_model oil_format oil_math oil_core)

  # J1-J15: All benchmarks
  add_oil_bench(bench_all bench/bench_all.cpp oil_model oil_math oil_format)

  # Hardware-probe / auto-select benchmark
  add_oil_bench(bench_hardware tests/test_bench.cpp oil_backend oil_math)

  # OIL quantized GEMM benchmarks (Gap #2)
  add_oil_bench(bench_oil_quant bench/bench_oil_quant.cpp oil_kernel oil_math oil_core)

  # AWQ/GPTQ comparison benchmarks
  add_oil_bench(bench_awq_gptq bench/bench_awq_gptq.cpp oil_format oil_core)

  # GPT-2 real inference quality test
  add_oil_bench(bench_gpt2_inference bench/bench_gpt2_inference.cpp oil_format oil_core)

  # Training benchmarks
  add_oil_bench(bench_training bench/bench_training.cpp oil_trainer oil_model oil_tokenizer)
  add_oil_bench(bench_multimodal bench/bench_multimodal.cpp oil_multimodal oil_model oil_tokenizer)

  # Training speed benchmarks
  add_oil_bench(bench_speed tools/bench_speed.cpp oil_trainer oil_model)
endif()

# SOPS tools + integration
add_library(oil_sops STATIC
  sops/sops_scheduler.cpp
  src/sops_integration.cpp
)
target_include_directories(oil_sops PUBLIC include sops)
target_link_libraries(oil_sops PUBLIC oil_core)
set_target_properties(oil_sops PROPERTIES CXX_STANDARD 20)

# Thread safety primitives
add_library(oil_thread_safety STATIC
  src/thread_safety.cpp
)
target_include_directories(oil_thread_safety PUBLIC include)
target_link_libraries(oil_thread_safety PUBLIC oil_core)

# Continual learning engine
add_library(oil_continual STATIC
  src/continual_engine.cpp
)
target_include_directories(oil_continual PUBLIC include)
target_link_libraries(oil_continual PUBLIC oil_format oil_sops)

# Speculative decoding engine
add_library(oil_speculative STATIC
  src/speculative_decoder.cpp
)
target_include_directories(oil_speculative PUBLIC include)
target_link_libraries(oil_speculative PUBLIC oil_model oil_sops)

add_executable(bench_sops sops/sops_bench.cpp)
target_include_directories(bench_sops PRIVATE include)
target_link_libraries(bench_sops PRIVATE oil_core)
set_target_properties(bench_sops PROPERTIES CXX_STANDARD 20)

add_executable(sops_info sops/sops_format_info.cpp)
target_include_directories(sops_info PRIVATE include)
set_target_properties(sops_info PROPERTIES CXX_STANDARD 20)

add_executable(sops_train_sim sops/sops_training_sim.cpp)
target_include_directories(sops_train_sim PRIVATE include)
target_link_libraries(sops_train_sim PRIVATE oil_core)
set_target_properties(sops_train_sim PROPERTIES CXX_STANDARD 20)
