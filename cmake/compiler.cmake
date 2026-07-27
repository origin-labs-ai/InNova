if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(STATUS "COMPILER: Clang ${CMAKE_CXX_COMPILER_VERSION}")

  add_compile_options(-Wall -Wextra -Wpedantic -fstack-protector-strong
    -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function
    -Wno-missing-field-initializers -Wno-sign-compare -Wno-float-equal
    -Wno-cast-qual -Wno-unknown-argument -Wno-unused-command-line-argument)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)

  if(OIL_SANITIZE)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
  endif()

  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -DNDEBUG -D_FORTIFY_SOURCE=2)
  elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O0 -g -DOIL_DEBUG)
  else()
    add_compile_options(-O2)
  endif()

  if(OIL_AVX2)
    add_compile_options(-mavx2 -mavx -mbmi2 -mfma)
  endif()
  if(OIL_AVX512)
    add_compile_options(-mavx512f -mavx512bw -mavx512vl)
  endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  message(STATUS "COMPILER: GCC ${CMAKE_CXX_COMPILER_VERSION}")
  add_compile_options(-Wall -Wextra -Wpedantic -fstack-protector-strong -Wno-unused-parameter)

  if(OIL_SANITIZE)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
  endif()

  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -DNDEBUG -march=native -D_FORTIFY_SOURCE=2)
  elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O0 -g -DOIL_DEBUG)
  endif()

  if(OIL_AVX2)
    add_compile_options(-mavx2 -mavx -mbmi2 -mfma)
  endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  message(STATUS "COMPILER: MSVC")
  add_compile_options(/W3 /utf-8)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)

  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(/O2 /DNDEBUG)
  elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(/Od /Zi /DOIL_DEBUG)
  endif()

  if(OIL_AVX2)
    add_compile_options(/arch:AVX2)
    add_definitions(-D__AVX2__)
  endif()
  if(OIL_AVX512)
    add_compile_options(/arch:AVX512)
    add_definitions(-D__AVX512F__)
  endif()
endif()
