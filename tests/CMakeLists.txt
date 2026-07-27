# tests/CMakeLists.txt — CTest integration with labels and timeouts
# Requires parent CMakeLists.txt to call add_subdirectory(tests)

# ── helper ────────────────────────────────────────────────────────
function(add_oil_test_full name src)
  set(_all_args ${ARGN})
  list(LENGTH _all_args _arg_count)
  math(EXPR _timeout_idx "${_arg_count} - 1")
  list(GET _all_args ${_timeout_idx} _timeout_val)
  list(REMOVE_AT _all_args ${_timeout_idx})
  set(deps ${_all_args})

  add_executable(${name} ${src})
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(${name} PRIVATE ${deps})
  add_test(NAME ${name} COMMAND $<TARGET_FILE:${name}>)
  set_tests_properties(${name} PROPERTIES
    WORKING_DIRECTORY $<TARGET_FILE_DIR:${name}>
    TIMEOUT ${_timeout_val}
  )
endfunction()

set(_default_timeout 60)
set(_long_timeout    300)

# ── core ──────────────────────────────────────────────────────────
add_oil_test_full(test_tensor     test_tensor.cpp     oil_core                              ${_default_timeout})
set_tests_properties(test_tensor     PROPERTIES LABELS "core")

add_oil_test_full(test_math       test_math.cpp       oil_math                              ${_default_timeout})
set_tests_properties(test_math       PROPERTIES LABELS "core")

add_oil_test_full(test_format     test_format.cpp     oil_format                            ${_default_timeout})
set_tests_properties(test_format     PROPERTIES LABELS "core")

add_oil_test_full(test_kernel     test_kernel.cpp     oil_kernel                            ${_default_timeout})
set_tests_properties(test_kernel     PROPERTIES LABELS "core")

add_oil_test_full(test_model      test_model.cpp      oil_model                             ${_default_timeout})
set_tests_properties(test_model      PROPERTIES LABELS "core")

add_oil_test_full(test_tokenizer  test_tokenizer.cpp  oil_tokenizer                         ${_default_timeout})
set_tests_properties(test_tokenizer  PROPERTIES LABELS "core")

add_oil_test_full(test_debug      test_debug.cpp      oil_model                             ${_default_timeout})
set_tests_properties(test_debug      PROPERTIES LABELS "core")

add_oil_test_full(test_all        test_all.cpp        oil_moe_variants oil_backend          ${_long_timeout})
set_tests_properties(test_all        PROPERTIES LABELS "core;distributed")

# ── training ──────────────────────────────────────────────────────
add_oil_test_full(test_trainer    test_trainer.cpp    oil_trainer oil_inference oil_tokenizer ${_long_timeout})
set_tests_properties(test_trainer    PROPERTIES LABELS "training")

add_oil_test_full(test_training   test_training.cpp   oil_trainer oil_model oil_engine oil_math oil_core ${_long_timeout})
set_tests_properties(test_training   PROPERTIES LABELS "training")

add_oil_test_full(test_gradient_check test_gradient_check.cpp oil_trainer oil_model oil_math oil_core ${_long_timeout})
set_tests_properties(test_gradient_check PROPERTIES LABELS "training")

add_oil_test_full(test_expert_parallel test_expert_parallel.cpp oil_moe_variants oil_trainer oil_model oil_math oil_kernel ${_long_timeout})
set_tests_properties(test_expert_parallel PROPERTIES LABELS "training;distributed")

add_oil_test_full(test_mixed_precision_proof test_mixed_precision_proof.cpp oil_trainer oil_model oil_math oil_core ${_long_timeout})
set_tests_properties(test_mixed_precision_proof PROPERTIES LABELS "training")

add_oil_test_full(test_quant_engines   test_quant_engines.cpp   oil_format oil_kernel oil_math ${_long_timeout})
set_tests_properties(test_quant_engines   PROPERTIES LABELS "training")

add_oil_test_full(test_quant_convergence test_quant_convergence.cpp oil_format oil_kernel oil_math oil_core oil_model oil_trainer oil_inference ${_long_timeout})
set_tests_properties(test_quant_convergence PROPERTIES LABELS "training")

add_oil_test_full(test_native_oil   test_native_oil.cpp   oil_trainer oil_model oil_kernel oil_format ${_long_timeout})
set_tests_properties(test_native_oil   PROPERTIES LABELS "training")

add_oil_test_full(test_native_oil_moe test_native_oil_moe.cpp oil_trainer oil_model oil_moe_variants oil_kernel oil_format ${_long_timeout})
set_tests_properties(test_native_oil_moe PROPERTIES LABELS "training;distributed")

add_oil_test_full(test_moe_training test_moe_training.cpp oil_trainer oil_model oil_moe_variants oil_kernel ${_long_timeout})
set_tests_properties(test_moe_training PROPERTIES LABELS "training;distributed")

add_oil_test_full(test_training_features test_training_features.cpp oil_trainer oil_model oil_math oil_core ${_long_timeout})
set_tests_properties(test_training_features PROPERTIES LABELS "training")

add_oil_test_full(test_native_weight test_native_weight.cpp oil_trainer oil_model oil_kernel oil_format ${_long_timeout})
set_tests_properties(test_native_weight PROPERTIES LABELS "training")

# ── inference ─────────────────────────────────────────────────────
add_oil_test_full(test_inference_opt test_inference_opt.cpp oil_inference_opt oil_model oil_inference ${_default_timeout})
set_tests_properties(test_inference_opt PROPERTIES LABELS "inference")

add_oil_test_full(test_inference_batch test_inference_batch.cpp oil_inference oil_inference_opt oil_model oil_math oil_core ${_default_timeout})
set_tests_properties(test_inference_batch PROPERTIES LABELS "inference")

add_oil_test_full(test_ops          test_ops.cpp          oil_kernel oil_math oil_core oil_model oil_inference oil_backend oil_format ${_default_timeout})
set_tests_properties(test_ops          PROPERTIES LABELS "inference")

# ── GPU ───────────────────────────────────────────────────────────
if(WIN32)
  target_link_libraries(test_all PRIVATE oil_gpu)
  add_executable(test_gpu test_gpu.cpp)
  target_include_directories(test_gpu PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(test_gpu PRIVATE oil_math oil_gpu oil_backend)
  add_test(NAME test_gpu COMMAND test_gpu)
  set_tests_properties(test_gpu PROPERTIES
    TIMEOUT ${_long_timeout}
    LABELS "gpu"
  )
endif()

# ── distributed / MoE ────────────────────────────────────────────
add_oil_test_full(test_moe_crash_repro test_moe_crash_repro.cpp oil_moe_variants oil_model ${_long_timeout})
set_tests_properties(test_moe_crash_repro PROPERTIES LABELS "distributed")

# ── ASI / multimodal / production ─────────────────────────────────
add_oil_test_full(test_asi          test_asi.cpp          oil_asi oil_trainer oil_model oil_tokenizer ${_long_timeout})
set_tests_properties(test_asi          PROPERTIES LABELS "inference")

add_oil_test_full(test_asi_flywheel  test_asi_flywheel.cpp oil_asi oil_trainer oil_model         ${_long_timeout})
set_tests_properties(test_asi_flywheel  PROPERTIES LABELS "inference")

add_oil_test_full(test_multimodal   test_multimodal.cpp   oil_multimodal oil_model oil_tokenizer oil_trainer oil_inference ${_long_timeout})
set_tests_properties(test_multimodal   PROPERTIES LABELS "inference")

add_oil_test_full(test_multimodal_encoders test_multimodal_encoders.cpp oil_multimodal oil_model oil_tokenizer ${_long_timeout})
set_tests_properties(test_multimodal_encoders PROPERTIES LABELS "inference")

add_oil_test_full(test_production   test_production.cpp   oil_production oil_model oil_tokenizer ${_long_timeout})
set_tests_properties(test_production   PROPERTIES LABELS "inference")

# ── protection / integrity ────────────────────────────────────────
add_oil_test_full(test_protected   test_protected.cpp   oil_moe_variants oil_backend oil_trainer oil_tokenizer oil_model oil_kernel oil_format oil_math oil_inference oil_engine ${_long_timeout})
set_tests_properties(test_protected   PROPERTIES LABELS "core;training")

add_oil_test_full(test_sha256_corrupt test_sha256_corrupt.cpp oil_core oil_format               ${_default_timeout})
set_tests_properties(test_sha256_corrupt PROPERTIES LABELS "core")

# ── paged KV cache ───────────────────────────────────────────────
add_oil_test_full(test_paged_kv_4m  paged_kv_4m_test.cpp  oil_model oil_kernel               ${_long_timeout})
set_tests_properties(test_paged_kv_4m  PROPERTIES LABELS "inference")

# ── parallel execution ────────────────────────────────────────────
include(ProcessorCount)
ProcessorCount(NPROC)
if(NPROC EQUAL 0)
  set(NPROC 4)
endif()
set_tests_properties(
  test_tensor test_math test_format test_kernel test_model test_tokenizer
  test_ops test_sha256_corrupt
  PROPERTIES COST 1.0
)
set_tests_properties(
  test_trainer test_training test_gradient_check test_expert_parallel
  test_all test_protected test_inference_opt test_asi test_multimodal
  test_production test_mixed_precision_proof test_native_oil test_moe_crash_repro
  test_quant_engines test_quant_convergence test_native_weight
  test_moe_training test_native_oil_moe test_multimodal_encoders
  test_asi_flywheel test_inference_batch test_paged_kv_4m
  test_training_features
  PROPERTIES COST 2.0
)
