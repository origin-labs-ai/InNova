@echo off
rem ASAN/UBSAN build for Windows (Clang-cl)
set PATH=%USERPROFILE%\AppData\Roaming\Python\Python314\Scripts;%PATH%
set CFLAGS=/fsanitize=address,undefined /fno-sanitize-recover=all
set CXXFLAGS=/fsanitize=address,undefined /fno-sanitize-recover=all
cmake -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug ^
    -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl ^
    -DCMAKE_C_FLAGS="/fsanitize=address,undefined /fno-sanitize-recover=all" ^
    -DCMAKE_CXX_FLAGS="/fsanitize=address,undefined /fno-sanitize-recover=all" ^
    -DOIL_BUILD_TESTS=ON
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
cmake --build build-asan --parallel
if %ERRORLEVEL% equ 0 (
    echo.
    echo === ASAN/UBSAN build SUCCESS ===
    echo Run: .\build-asan\tests\test_ops.exe
)
