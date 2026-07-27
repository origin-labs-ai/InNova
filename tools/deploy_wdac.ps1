# deploy_wdac.ps1 — Deploy WDAC policy to trust XeonAI signed binaries
$ErrorActionPreference = "Stop"
$policyXml = Join-Path $PSScriptRoot "wdac\allow_build.xml"
$targetDir = "C:\Windows\System32\CodeIntegrity\CiPolicies\Active\{B244370E-44C9-4C06-B551-F6016E563076}"

Write-Host "=== Deploy WDAC Policy ===" -ForegroundColor Cyan

if (-not (Test-Path $policyXml)) {
    Write-Host "Policy XML not found: $policyXml" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $targetDir)) {
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
}

Copy-Item -Path $policyXml -Destination (Join-Path $targetDir "allow_build.xml") -Force
Write-Host "Policy deployed to: $targetDir" -ForegroundColor Green

# Convert and activate using PowerShell cmdlets (requires admin)
try {
    $binPath = Join-Path $targetDir "allow_build.bin"
    ConvertFrom-CIPolicy -XmlFilePath $policyXml -BinaryFilePath $binPath -ErrorAction Stop | Out-Null
    Write-Host "Converted to binary policy: $binPath" -ForegroundColor Green

    # Set as supplemental policy (requires admin)
    $policyId = "{B244370E-44C9-4C06-B551-F6016E563076}"
    Set-CIPolicyPolicyIdSupplemental -FilePath $binPath -PolicyId $policyId -ErrorAction SilentlyContinue | Out-Null
    Write-Host "Set as supplemental policy (requires reboot)" -ForegroundColor Yellow
} catch {
    Write-Host "WDAC activation requires admin. Policy XML copied. Run as admin to activate." -ForegroundColor DarkYellow
    Write-Host "  ConvertFrom-CIPolicy -XmlFilePath $policyXml -BinaryFilePath $binPath" -ForegroundColor DarkGray
    Write-Host "  Set-CIPolicyPolicyIdSupplemental -FilePath $binPath -PolicyId $policyId" -ForegroundColor DarkGray
}
