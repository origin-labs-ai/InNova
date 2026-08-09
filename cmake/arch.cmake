include(CheckCXXSourceCompiles)

set(HAVE_AVX2 FALSE)
set(HAVE_AVX512 FALSE)
set(HAVE_NEON FALSE)

# Save current flags and add arch-specific test flags
function(_try_avx2)
  set(CMAKE_REQUIRED_FLAGS_SAVE "${CMAKE_REQUIRED_FLAGS}")
  if(MSVC)
    set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
  else()
    set(CMAKE_REQUIRED_FLAGS "-mavx2 -mavx -mbmi2 -mfma")
  endif()
  message(STATUS "ARCH: Testing AVX2 with flags: ${CMAKE_REQUIRED_FLAGS}")
  check_cxx_source_compiles("
    #include <immintrin.h>
    int main() {
      __m256i a = _mm256_setzero_si256();
      __m256i b = _mm256_shuffle_epi8(a, a);
      __m256 c = _mm256_set1_ps(1.0f);
      __m256 d = _mm256_add_ps(c, c);
      (void)b; (void)d;
      return 0;
    }
  " HAVE_AVX2)
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS_SAVE}")
  message(STATUS "ARCH: AVX2 check result: ${HAVE_AVX2}")
  if(HAVE_AVX2)
    set(QUANT_AVX2 ON CACHE INTERNAL "AVX2 support")
    message(STATUS "ARCH: AVX2 detected")
  else()
    message(STATUS "ARCH: AVX2 NOT detected (trying __AVX2__ definition fallback)")
  endif()
  # On MSVC x64, force-enable AVX2 since it is always available since Haswell (2013)
  # MSVC does not define __AVX2__ without /arch:AVX2 flag even when CPU supports it
  if(MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)")
    set(QUANT_AVX2 ON CACHE INTERNAL "AVX2 support (forced on MSVC x64)")
    message(STATUS "ARCH: AVX2 forced ON for MSVC x64")
  endif()
endfunction()

function(_try_avx512)
  set(CMAKE_REQUIRED_FLAGS_SAVE "${CMAKE_REQUIRED_FLAGS}")
  if(MSVC)
    set(CMAKE_REQUIRED_FLAGS "/arch:AVX512")
  else()
    set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512bw -mavx512vl")
  endif()
  check_cxx_source_compiles("
    #include <immintrin.h>
    int main() {
      __m512i a = _mm512_setzero_si512();
      (void)a;
      return 0;
    }
  " HAVE_AVX512)
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS_SAVE}")
  if(HAVE_AVX512)
    set(QUANT_AVX512 ON CACHE INTERNAL "AVX-512 support")
    message(STATUS "ARCH: AVX-512 detected")
  endif()
endfunction()

function(_try_neon)
  check_cxx_source_compiles("
    #include <arm_neon.h>
    int main() {
      int8x16_t a = vdupq_n_s8(0);
      float32x4_t b = vdupq_n_f32(0.0f);
      (void)a; (void)b;
      return 0;
    }
  " HAVE_NEON)
  if(HAVE_NEON)
    set(QUANT_NEON ON CACHE INTERNAL "NEON support")
    message(STATUS "ARCH: NEON detected")
  endif()
endfunction()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i686|i386)")
  _try_avx2()
  _try_avx512()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|ARM64|arm64)")
  _try_neon()
endif()
