param(
    [string]$BuildDir = "build_ci",
    [switch]$Rebuild,
    [switch]$FullTest,
    [switch]$Coverage
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
$StartTime = Get-Date

Write-Host "=== InNova Local CI ===" -ForegroundColor Cyan
Write-Host "Time: $($StartTime.ToString('yyyy-MM-dd HH:mm'))"
Write-Host "Repo: $RepoRoot"
Write-Host "Build: $BuildDir"

# ── 1. Find VS compiler ─────────────────────────────────────────────
$vcvars = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) {
    Write-Host "ERROR: vcvars64.bat not found at $vcvars" -ForegroundColor Red
    exit 1
}

Write-Host "`n[1/6] Compiler: MSVC 2026 x64" -ForegroundColor Yellow

# ── 2. Configure ────────────────────────────────────────────────────
Write-Host "`n[2/6] Configuring CMake..." -ForegroundColor Yellow
if ($Rebuild -and (Test-Path $BuildDir)) {
    Remove-Item -Path $BuildDir -Recurse -Force
}
if (-not (Test-Path $BuildDir)) {
    cmd.exe /c "call `"$vcvars`" && cmake -B $BuildDir -G Ninja -DCMAKE_BUILD_TYPE=Release -DOIL_BUILD_TESTS=ON -DOIL_BUILD_BENCHMARKS=ON -DOIL_BUILD_TOOLS=ON 2>&1"
    if ($LASTEXITCODE -ne 0) { Write-Host "CONFIGURE FAILED" -ForegroundColor Red; exit 1 }
}

# ── 3. Build ────────────────────────────────────────────────────────
Write-Host "`n[3/6] Building..." -ForegroundColor Yellow
cmd.exe /c "call `"$vcvars`" && cmake --build $BuildDir --parallel 2>&1"
if ($LASTEXITCODE -ne 0) { Write-Host "BUILD FAILED" -ForegroundColor Red; exit 1 }

# ── 4. Quick tests ──────────────────────────────────────────────────
Write-Host "`n[4/6] Running quick tests (excl. GPU/training/protected)..." -ForegroundColor Yellow
cmd.exe /c "call `"$vcvars`" && ctest --test-dir $BuildDir --output-on-failure --timeout 120 --exclude-regex `"test_protected|test_gpu|test_training|test_native_oil_moe|paged_kv_4m_test`" 2>&1"
if ($LASTEXITCODE -ne 0) { Write-Host "QUICK TESTS FAILED" -ForegroundColor Red; exit 1 }

# ── 5. Full tests (optional) ────────────────────────────────────────
if ($FullTest) {
    Write-Host "`n[5/6] Running ALL tests..." -ForegroundColor Yellow
    cmd.exe /c "call `"$vcvars`" && ctest --test-dir $BuildDir --output-on-failure --timeout 300 2>&1"
    if ($LASTEXITCODE -ne 0) { Write-Host "SOME TESTS FAILED" -ForegroundColor DarkYellow }
}

# ── 6. Coverage (optional) ──────────────────────────────────────────
if ($Coverage) {
    Write-Host "`n[6/6] Code coverage..." -ForegroundColor Yellow
    if (Get-Command "OpenCppCoverage" -ErrorAction SilentlyContinue) {
        OpenCppCoverage --sources $RepoRoot\src --export_type html:coverage_report `
            -- $BuildDir\test_all.exe 2>&1
        Write-Host "Coverage report: coverage_report/index.html" -ForegroundColor Green
    } else {
        Write-Host "OpenCppCoverage not found — install via: winget install OpenCppCoverage" -ForegroundColor DarkYellow
    }
}

$Duration = (Get-Date) - $StartTime
Write-Host "`n=== CI DONE in $($Duration.TotalMinutes.ToString('F1')) minutes ===" -ForegroundColor Cyan