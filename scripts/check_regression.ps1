# check_regression.ps1 — bench regression tracker (zero-dependency, pwsh 7+)
# Usage: pwsh scripts/check_regression.ps1 -Baseline <csv> -Current <csv> [-ThresholdPct 5.0]
# Exit codes: 0 = green (no regression beyond threshold), 1 = red (regression found),
#             2 = usage/input error.
# Rule (TRANSCRIPT.md L007): any format whose encode_us or decode_us regresses
# more than ThresholdPct vs baseline fails the gate.

param(
    [Parameter(Mandatory = $true)][string]$Baseline,
    [Parameter(Mandatory = $true)][string]$Current,
    [double]$ThresholdPct = 5.0
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Baseline)) { Write-Error "Baseline CSV not found: $Baseline"; exit 2 }
if (-not (Test-Path $Current))  { Write-Error "Current CSV not found: $Current";   exit 2 }

function Read-BenchCsv([string]$Path) {
    $map = @{}
    $lines = Get-Content -LiteralPath $Path
    if ($lines.Count -lt 2) { return $map }
    foreach ($line in $lines[1..($lines.Count - 1)]) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $f = $line -split ','
        if ($f.Count -lt 7) { continue }
        $key = "$($f[0])|$($f[1])"
        $entry = @{
            EncodeUs = [double]$f[5]
            DecodeUs = [double]$f[6]
        }
        if ($f.Count -ge 9) {
            $entry.EncodeStd = [double]$f[7]
            $entry.DecodeStd = [double]$f[8]
        }
        $map[$key] = $entry
    }
    return $map
}

$base = Read-BenchCsv $Baseline
$curr = Read-BenchCsv $Current

if ($base.Count -eq 0) { Write-Host "REGRESSION CHECK: baseline empty => nothing to compare, GREEN"; exit 0 }

$regressions = @()
$compared = 0
$worstKey = ""
$worstDelta = -1000.0

foreach ($key in $curr.Keys) {
    if (-not $base.ContainsKey($key)) { continue }  # new format, no baseline
    $b = $base[$key]; $c = $curr[$key]
    foreach ($metric in @('EncodeUs', 'DecodeUs')) {
        $bv = [double]$b[$metric]
        $cv = [double]$c[$metric]
        if ($bv -le 0) { continue }
        $deltaPct = (($cv - $bv) / $bv) * 100.0
        if ($deltaPct -gt $worstDelta) { $worstDelta = $deltaPct; $worstKey = "$($key):$metric" }
        if ($deltaPct -gt $ThresholdPct) {
            $regressions += [pscustomobject]@{
                Key       = $key
                Metric    = $metric
                Baseline  = $bv
                Current   = $cv
                DeltaPct  = [math]::Round($deltaPct, 2)
            }
        }
    }
    $compared++
}

Write-Host "== REGRESSION CHECK =="
Write-Host "Baseline : $Baseline"
Write-Host "Current  : $Current"
Write-Host "Compared : $compared rows x 2 metrics (threshold +$ThresholdPct%)"
Write-Host ("Worst delta: {0}% on {1}" -f [math]::Round($worstDelta, 2), $worstKey)

if ($regressions.Count -gt 0) {
    Write-Host "STATUS: RED - $($regressions.Count) regression(s):" -ForegroundColor Red
    $regressions | Format-Table -AutoSize | Out-String | Write-Host
    exit 1
}

Write-Host "STATUS: GREEN" -ForegroundColor Green
exit 0
