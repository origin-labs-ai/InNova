$testDir = "C:\Users\thaku\Downloads\InNova\build\Release"
$tests = Get-ChildItem "$testDir\test_*.exe" | Sort-Object Name
Write-Host "Found $($tests.Count) tests"
$passed = 0
$failed = @()

foreach ($t in $tests) {
    Start-Sleep -Milliseconds 300
    try {
        $p = Start-Process -FilePath $t.FullName -WorkingDirectory $testDir -NoNewWindow -Wait -PassThru
        if ($p.ExitCode -eq 0) {
            Write-Host "  PASS: $($t.BaseName)" -ForegroundColor Green
            $passed++
        } else {
            Write-Host "  FAIL: $($t.BaseName) (exit=$($p.ExitCode))" -ForegroundColor Red
            $failed += $t.BaseName
        }
    } catch {
        Write-Host "  ERROR: $($t.BaseName) - $($_.Exception.Message)" -ForegroundColor Red
        $failed += $t.BaseName
    }
}

Write-Host ""
Write-Host "Results: $passed/$($tests.Count) tests passed"
if ($failed.Count -gt 0) {
    Write-Host "Failed: $($failed -join ', ')" -ForegroundColor Red
    exit 1
} else {
    Write-Host "ALL TESTS PASSED!" -ForegroundColor Green
    exit 0
}
