@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set CFLAGS=/nologo /Iinclude /Ibuild_verify_new /DWIN32 /D_WINDOWS /DNDEBUG /D_CRT_SECURE_NO_WARNINGS /D__AVX2__ /O2 /std:c++20 /EHsc /MD
set OUT=build_verify_new\Release\manual

if not exist %OUT% mkdir %OUT%

echo ============================================================
echo Compiling all sources for test_native_quant_moe
echo ============================================================

echo --- quant_core ---
cl.exe %CFLAGS% /c src\tensor.cpp /Fo%OUT%\tensor.obj || exit /b 1
cl.exe %CFLAGS% /c src\memory.cpp /Fo%OUT%\memory.obj || exit /b 1
cl.exe %CFLAGS% /c src\random.cpp /Fo%OUT%\random.obj || exit /b 1
cl.exe %CFLAGS% /c src\tensor_view.cpp /Fo%OUT%\tensor_view.obj || exit /b 1
cl.exe %CFLAGS% /c src\data_gen.cpp /Fo%OUT%\data_gen.obj || exit /b 1

echo --- quant_math ---
cl.exe %CFLAGS% /c src\math.cpp /Fo%OUT%\math.obj || exit /b 1
cl.exe %CFLAGS% /c src\math_avx2_tensor.cpp /Fo%OUT%\math_avx2_tensor.obj || exit /b 1
cl.exe %CFLAGS% /c src\math_avx2_vec.cpp /Fo%OUT%\math_avx2_vec.obj || exit /b 1
cl.exe %CFLAGS% /c src\simd_math.cpp /Fo%OUT%\simd_math.obj || exit /b 1

echo --- quant_model ---
cl.exe %CFLAGS% /c src\transformer.cpp /Fo%OUT%\transformer.obj || exit /b 1
cl.exe %CFLAGS% /c src\model.cpp /Fo%OUT%\model.obj || exit /b 1
cl.exe %CFLAGS% /c src\kv_cache.cpp /Fo%OUT%\kv_cache.obj || exit /b 1
cl.exe %CFLAGS% /c src\autograd_engine.cpp /Fo%OUT%\autograd_engine.obj || exit /b 1
cl.exe %CFLAGS% /c src\autograd_functions.cpp /Fo%OUT%\autograd_functions.obj || exit /b 1
cl.exe %CFLAGS% /c src\autograd_grad.cpp /Fo%OUT%\autograd_grad.obj || exit /b 1
cl.exe %CFLAGS% /c src\flash_attention.cpp /Fo%OUT%\flash_attention.obj || exit /b 1
cl.exe %CFLAGS% /c src\dataloader.cpp /Fo%OUT%\dataloader.obj || exit /b 1
cl.exe %CFLAGS% /c src\training_utils.cpp /Fo%OUT%\training_utils.obj || exit /b 1
cl.exe %CFLAGS% /c src\metrics.cpp /Fo%OUT%\metrics.obj || exit /b 1

echo --- quant_native ---
cl.exe %CFLAGS% /c src\native_quant_moe.cpp /Fo%OUT%\native_quant_moe.obj || exit /b 1
cl.exe %CFLAGS% /c src\native_weight.cpp /Fo%OUT%\native_weight.obj || exit /b 1
cl.exe %CFLAGS% /c src\native_trainer.cpp /Fo%OUT%\native_trainer.obj || exit /b 1

echo --- test ---
cl.exe %CFLAGS% /c tests\test_native_quant_moe.cpp /Fo%OUT%\test_native_quant_moe.obj || exit /b 1

echo ============================================================
echo Linking...
echo ============================================================

link.exe /nologo /OUT:build_verify_new\Release\test_native_quant_moe_gs.exe /MACHINE:x64 ^
    %OUT%\test_native_quant_moe.obj ^
    %OUT%\native_quant_moe.obj %OUT%\native_weight.obj %OUT%\native_trainer.obj ^
    %OUT%\tensor.obj %OUT%\memory.obj %OUT%\random.obj %OUT%\tensor_view.obj %OUT%\data_gen.obj ^
    %OUT%\math.obj %OUT%\math_avx2_tensor.obj %OUT%\math_avx2_vec.obj %OUT%\simd_math.obj ^
    %OUT%\transformer.obj %OUT%\model.obj %OUT%\kv_cache.obj ^
    %OUT%\autograd_engine.obj %OUT%\autograd_functions.obj %OUT%\autograd_grad.obj ^
    %OUT%\flash_attention.obj %OUT%\dataloader.obj %OUT%\training_utils.obj %OUT%\metrics.obj ^
    build_verify_new\Release\quant_moe_model.lib ^
    build_verify_new\Release\quant_moe_variants.lib ^
    build_verify_new\Release\quant_backend.lib ^
    build_verify_new\Release\quant_gpu.lib ^
    build_verify_new\Release\quant_kernel.lib ^
    build_verify_new\Release\quant_format.lib ^
    build_verify_new\Release\quant_inference.lib ^
    build_verify_new\Release\quant_trainer.lib ^
    build_verify_new\Release\quant_tokenizer.lib ^
    build_verify_new\Release\quant_distributed.lib ^
    d3d12.lib dxgi.lib d3dcompiler.lib ^
    kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ^
    ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib ^
    || exit /b 1

echo ============================================================
echo SUCCESS
echo ============================================================
dir /Q build_verify_new\Release\test_native_quant_moe_gs.exe