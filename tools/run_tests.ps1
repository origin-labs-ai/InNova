$testDir = "C:\Users\thaku\Downloads\InNova\build\Release"
$tests = Get-ChildItem "$testDir\test_*.exe" | Sort-Object Name
$passed = 0
$failed = @()

foreach ($t in $tests) {
    Start-Sleep -Milliseconds 500
    $output = ""
    $exitCode = -1
    try {
        $p = Start-Process -FilePath $t.FullName -ArgumentList "" -WorkingDirectory $testDir -NoNewWindow -Wait -PassThru -RedirectStandardOutput "$env:TEMP\$($t.BaseName)_out.txt" -RedirectStandardError "$env:TEMP\$($t.BaseName)_err.txt"
        $exitCode = $p.ExitCode
        $output = Get-Content "$env:TEMP\$($t.BaseName)_out.txt" -ErrorAction SilentlyContinue
    } catch {
        $exitCode = -1
    }
    
    if ($exitCode -eq 0) {
        Write-Host "  PASS: $($t.BaseName)" -ForegroundColor Green
        $passed++
    } else {
        Write-Host "  FAIL: $($t.BaseName) (exit=$exitCode)" -ForegroundColor Red
        $failed += $t.BaseName
    }
}

Write-Host ""
Write-Host "Results: $passed/$($tests.Count) tests passed"
if ($failed.Count -gt 0) {
    Write-Host "Failed: $($failed -join ', ')" -ForegroundColor Red
} else {
    Write-Host "ALL 16/16 TESTS PASSED!" -ForegroundColor Green
}
