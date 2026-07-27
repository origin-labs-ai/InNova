# ============================================================================
# ADAPTER-EDITION — Subdirectory of main MYTHOS.cpp project
# ----------------------------------------------------------------------------
# Links against the parent project's already-built libraries (oil_core,
# oil_format, oil_kernel, oil_model, oil_math).  Does NOT recompile parent
# sources — avoids ODR violations when built as a subdirectory.
#
# Scope: format converters (GGUF, Safetensors, PTQ) + native trainer
# (scratch + continual modes) + delta-adapter host + MTP-head trainer.
# ============================================================================
cmake_minimum_required(VERSION 3.20)
project(MYTHOS_Adapter_Edition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Options ────────────────────────────────────────────────────────────────
option(OIL_ADAPTER_BUILD_TOOLS "Build adapter edition CLI tools" ON)
option(OIL_ADAPTER_BUILD_TESTS "Build adapter edition tests" ON)

# ── Library: oil_adapter ───────────────────────────────────────────────────
# Adapter bridge implementations + continual trainer + delta adapter + MTP heads
set(ADAPTER_SOURCES
    "src/adapter_core.cpp"
    "src/ptq_bridge.cpp"
    "src/gguf_bridge.cpp"
    "src/safetensors_bridge.cpp"
    "src/continual_trainer.cpp"
    "src/delta_adapter.cpp"
    "src/mtp_head_trainer.cpp"
)

add_library(oil_adapter STATIC ${ADAPTER_SOURCES})
target_include_directories(oil_adapter PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
)
target_link_libraries(oil_adapter PUBLIC
    oil_core
    oil_math
    oil_format
    oil_kernel
    oil_model
)
if(MSVC)
    target_compile_options(oil_adapter PRIVATE /W3 /utf-8 /DNOMINMAX)
else()
    target_compile_options(oil_adapter PRIVATE -Wall -Wextra)
endif()

# ── CLI tools ──────────────────────────────────────────────────────────────
if(OIL_ADAPTER_BUILD_TOOLS)
    function(add_adapter_tool name src)
        add_executable(${name} ${src})
        target_include_directories(${name} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include"
        )
        target_link_libraries(${name} PRIVATE oil_adapter)
        if(MSVC)
            target_compile_options(${name} PRIVATE /W3 /utf-8 /DNOMINMAX)
        endif()
    endfunction()

    add_adapter_tool(oil_ptq    "tools/oil_ptq.cpp")
    add_adapter_tool(oil_adapt  "tools/oil_adapt.cpp")
    add_adapter_tool(oil_import "tools/oil_import.cpp")
    add_adapter_tool(bench_compare "tools/bench_compare.cpp")
    add_adapter_tool(bench_full "tools/bench_full.cpp")
    add_adapter_tool(bench_ollama_comparison "tools/bench_ollama_comparison.cpp")
    add_adapter_tool(bench_bitnet_comparison "tools/bench_bitnet_comparison.cpp")
endif()

# ── Tests ──────────────────────────────────────────────────────────────────
if(OIL_ADAPTER_BUILD_TESTS)
    enable_testing()

    add_executable(test_adapter_bridges "tests/test_adapter_bridges.cpp")
    target_include_directories(test_adapter_bridges PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    )
    target_link_libraries(test_adapter_bridges PRIVATE oil_adapter)
    if(MSVC)
        target_compile_options(test_adapter_bridges PRIVATE /W3 /utf-8 /DNOMINMAX)
    endif()
    add_test(NAME test_adapter_bridges COMMAND test_adapter_bridges)
    set_tests_properties(test_adapter_bridges PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()
