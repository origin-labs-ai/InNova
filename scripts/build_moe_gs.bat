@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set CFLAGS=/nologo /Iinclude /Ibuild_verify_new /DWIN32 /D_WINDOWS /DNDEBUG /D_CRT_SECURE_NO_WARNINGS /D__AVX2__ /O2 /std:c++20 /EHsc /MD
set OUT=build_verify_new\Release\manual

if not exist %OUT% mkdir %OUT%

echo Compiling native_quant_moe.cpp...
cl.exe %CFLAGS% /c src\native_quant_moe.cpp /Fo%OUT%\native_quant_moe.obj || exit /b 1

echo Compiling native_weight.cpp...
cl.exe %CFLAGS% /c src\native_weight.cpp /Fo%OUT%\native_weight.obj || exit /b 1

echo Compiling native_trainer.cpp...
cl.exe %CFLAGS% /c src\native_trainer.cpp /Fo%OUT%\native_trainer.obj || exit /b 1

echo Compiling test_native_quant_moe.cpp...
cl.exe %CFLAGS% /c tests\test_native_quant_moe.cpp /Fo%OUT%\test_native_quant_moe.obj || exit /b 1

echo Linking...
link.exe /nologo /OUT:build_verify_new\Release\test_native_quant_moe_gs.exe /MACHINE:x64 ^
    %OUT%\test_native_quant_moe.obj ^
    %OUT%\native_quant_moe.obj ^
    %OUT%\native_weight.obj ^
    %OUT%\native_trainer.obj ^
    build_verify_new\Release\quant_model.lib ^
    build_verify_new\Release\quant_math.lib ^
    build_verify_new\Release\quant_core.lib ^
    build_verify_new\Release\quant_inference.lib ^
    build_verify_new\Release\quant_backend.lib ^
    build_verify_new\Release\quant_gpu.lib ^
    build_verify_new\Release\quant_kernel.lib ^
    build_verify_new\Release\quant_format.lib ^
    build_verify_new\Release\quant_moe_model.lib ^
    build_verify_new\Release\quant_moe_variants.lib ^
    build_verify_new\Release\quant_trainer.lib ^
    build_verify_new\Release\quant_tokenizer.lib ^
    build_verify_new\Release\quant_distributed.lib ^
    d3d12.lib dxgi.lib d3dcompiler.lib ^
    kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ^
    ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib || exit /b 1

echo SUCCESS: build_verify_new\Release\test_native_quant_moe_gs.exe created
dir /Q build_verify_new\Release\test_native_quant_moe_gs.exe