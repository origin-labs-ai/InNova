# disable_smartscreen.ps1 — Disable SmartScreen reputation-based protection
$ErrorActionPreference = "Stop"
$regPath = "HKLM:\SYSTEM\CurrentControlSet\Control\CI"
$regName = "VerifiedAndReputablePolicyState"

Write-Host "=== Disable SmartScreen Reputation Policy ===" -ForegroundColor Cyan

try {
    $current = Get-ItemProperty -Path $regPath -Name $regName -ErrorAction Stop
    Write-Host "Current value: $($current.$regName)" -ForegroundColor Yellow
} catch {
    Write-Host "Registry key not found, will create it." -ForegroundColor DarkYellow
}

Set-ItemProperty -Path $regPath -Name $regName -Value 0 -Force -ErrorAction Stop
Write-Host "SmartScreen reputation policy disabled (value=0)" -ForegroundColor Green
Write-Host "Reboot required for changes to take effect." -ForegroundColor Yellow
