# sign_adapter.ps1 — Sign ADAPTER-EDITION binaries with XeonAI cert
param([string]$BuildDir)

$ErrorActionPreference = "Stop"
$CertStore = "Cert:\CurrentUser\My"
$CertSubject = "CN=XeonAI"

Write-Host "=== XeonAI Code Signing ===" -ForegroundColor Cyan
Write-Host "Build dir: $BuildDir"

$cert = Get-ChildItem $CertStore -CodeSigningCert |
        Where-Object { $_.Subject -eq $CertSubject } |
        Select-Object -First 1

if (-not $cert) {
    Write-Host "`n[1/2] Creating self-signed code-signing certificate: $CertSubject" -ForegroundColor Yellow
    $cert = New-SelfSignedCertificate `
        -Subject $CertSubject `
        -Type CodeSigningCert `
        -CertStoreLocation $CertStore `
        -NotAfter (Get-Date).AddYears(5) `
        -HashAlgorithm SHA256 `
        -KeyUsage DigitalSignature `
        -TextExtension @("2.5.29.37={text}1.3.6.1.5.5.7.3.3")
    Write-Host "  Created cert: $($cert.Thumbprint)" -ForegroundColor Green
} else {
    Write-Host "`n[1/2] Reusing existing cert: $($cert.Thumbprint)" -ForegroundColor Green
}

Write-Host "`n[2/2] Signing executables in $BuildDir..." -ForegroundColor Yellow
$exes = Get-ChildItem -LiteralPath $BuildDir -Filter "*.exe"
$signed = 0
foreach ($exe in $exes) {
    try {
        Set-AuthenticodeSignature -FilePath $exe.FullName -Certificate $cert -HashAlgorithm SHA256 -ErrorAction Stop | Out-Null
        $signed++
        Write-Host "  Signed: $($exe.Name)" -ForegroundColor Green
    } catch {
        Write-Host "  WARN: Failed to sign $($exe.Name): $_" -ForegroundColor DarkYellow
    }
}
Write-Host "`nSigned $signed / $($exes.Count) executables" -ForegroundColor Cyan
Write-Host "Publisher: XeonAI" -ForegroundColor Cyan
