# sign.ps1 — Code-sign all MYTHOS binaries + deploy WDAC policy
# Run as Administrator from repo root: .\tools\sign.ps1
#
# What it does:
#   1. Creates a self-signed code-signing cert (if not cached)
#   2. Signs every .exe in build_windows/Release/
#   3. Generates a WDAC supplemental policy trusting the publisher
#   4. Deploys the policy (requires admin)
#
# After this, SAC/WDAC will trust all signed binaries — even after rebuilds.

param(
    [string]$BuildDir = "build_windows\Release",
    [string]$CertStore = "Cert:\CurrentUser\My",
    [string]$CertSubject = "CN=XeonAI",
    [string]$PolicyName = "XeonAI Dev Allowlist"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
$ExeDir = Join-Path $RepoRoot $BuildDir

Write-Host "=== MYTHOS.cpp Code Signing ===" -ForegroundColor Cyan
Write-Host "Build dir: $ExeDir"

# ── 1. Create / reuse self-signed cert ──────────────────────────────
$cert = Get-ChildItem $CertStore -CodeSigningCert |
        Where-Object { $_.Subject -eq $CertSubject } |
        Select-Object -First 1

if (-not $cert) {
    Write-Host "`n[1/4] Creating self-signed code-signing certificate..." -ForegroundColor Yellow
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
    Write-Host "`n[1/4] Reusing existing cert: $($cert.Thumbprint)" -ForegroundColor Green
}

# ── 2. Sign every .exe ─────────────────────────────────────────────
Write-Host "`n[2/4] Signing executables..." -ForegroundColor Yellow
$exes = Get-ChildItem $ExeDir -Filter "*.exe"
$signed = 0
foreach ($exe in $exes) {
    try {
        Set-AuthenticodeSignature -FilePath $exe.FullName -Certificate $cert -HashAlgorithm SHA256 -ErrorAction Stop | Out-Null
        $signed++
    } catch {
        Write-Host "  WARN: Failed to sign $($exe.Name): $_" -ForegroundColor DarkYellow
    }
}
Write-Host "  Signed $signed / $($exes.Count) executables" -ForegroundColor Green

# ── 3. Generate publisher-based WDAC supplemental policy ────────────
Write-Host "`n[3/4] Generating WDAC policy (publisher trust)..." -ForegroundColor Yellow

$policyDir = Join-Path $RepoRoot "tools\wdac"
if (-not (Test-Path $policyDir)) { New-Item -ItemType Directory -Path $policyDir -Force | Out-Null }

$certHash = $cert.Thumbprint
$issuerHash = (Get-AuthenticodeSignature ($exes[0].FullName)).SignerCertificate.Issuer

# Publisher-based rule: trust anything signed by this cert
$policyXml = @"
<?xml version="1.0" encoding="utf-8"?>
<SiPolicy xmlns="urn:schemas-microsoft-com:sipolicy">
  <VersionEx>10.0.0.0</VersionEx>
  <PlatformID>{2E07F7E4-194C-4D20-B7C9-6F44A6C5A234}</PlatformID>
  <PolicyID>{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}</PolicyID>
  <Rules>
    <Rule>
      <Option>Enabled:Unsigned System Integrity Policy</Option>
    </Rule>
    <Rule>
      <Option>Enabled:Audit Mode</Option>
    </Rule>
  </Rules>
  <EKUs />
  <FileRules>
    <Allow ID="ID_ALLOW_SIGNER_1" FriendlyName="XeonAI Dev Signing Cert" />
  </FileRules>
  <Signers>
    <Signer ID="ID_SIGNER_1" Name="XeonAI Dev Signing">
      <CertPublisher Value="$certHash" />
    </Signer>
  </Signers>
  <SigningScenarios>
    <SigningScenario Value="131" ID="ID_SIGNINGSCENARIO_DRIVERS_1" FriendlyName="Drivers">
      <ProductSigners />
    </SigningScenario>
    <SigningScenario Value="12" ID="ID_SIGNINGSCENARIO_WINDOWS" FriendlyName="User-mode">
      <ProductSigners>
        <AllowedSigners>
          <AllowedSigner RefID="ID_SIGNER_1" />
        </AllowedSigners>
      </ProductSigners>
    </SigningScenario>
  </SigningScenarios>
  <UpdatePolicySigners />
  <CiSigners />
  <HvciOptions>0</HvciOptions>
  <PolicyTypeID>{A244370E-44C9-4C06-B551-F6016E563076}</PolicyTypeID>
</SiPolicy>
"@

$policyXmlPath = Join-Path $policyDir "mythos_publisher_policy.xml"
$policyBinPath = Join-Path $policyDir "mythos_publisher_policy.bin"

Set-Content -Path $policyXmlPath -Value $policyXml -Encoding UTF8
Write-Host "  Policy XML: $policyXmlPath" -ForegroundColor Green

# ── 4. Deploy policy ────────────────────────────────────────────────
Write-Host "`n[4/4] Deploying WDAC policy..." -ForegroundColor Yellow

# Also update hash-based supplemental policy with current hashes
$hashPolicy = '<?xml version="1.0" encoding="utf-8"?>' + "`n"
$hashPolicy += '<SIPolicy xmlns="urn:schemas-microsoft-com:sipolicy" SchemaVersion="1.0" siPolicyType="Supplemental">' + "`n"
$hashPolicy += '  <VersionEx>10.0.26100.1</VersionEx>' + "`n"
$hashPolicy += '  <PolicyID>{B244370E-44C9-4C06-B551-F6016E563076}</PolicyID>' + "`n"
$hashPolicy += '  <PlatformID>{2E07F7E4-194C-4D20-B7C9-6F44A6C5A234}</PlatformID>' + "`n"
$hashPolicy += '  <Rules><Rule><Option>Enabled:Unsigned System Integrity Policy</Option></Rule></Rules>' + "`n"
$hashPolicy += '  <FileRules>' + "`n"

foreach ($exe in $exes) {
    $sha256 = (Get-FileHash $exe.FullName -Algorithm SHA256).Hash
    $hashPolicy += "    <Allow Hash=`"$sha256`" />`n"
}

$hashPolicy += '  </FileRules>' + "`n"
$hashPolicy += '</SIPolicy>'

$allowBuildPath = Join-Path $policyDir "allow_build.xml"
Set-Content -Path $allowBuildPath -Value $hashPolicy -Encoding UTF8
Write-Host "  Updated allow_build.xml with $($exes.Count) current hashes" -ForegroundColor Green

# Deploy: copy policy to CiPolicies\Active and set as supplemental
$activePolDir = "C:\Windows\System32\CodeIntegrity\CiPolicies\Active"
if (Test-Path $activePolDir) {
    try {
        # Convert XML to binary (requires TrustedPlatformModule or convert from policy XML)
        # For supplemental policies, we can use the XML directly with WDAC
        $targetDir = Join-Path $activePolDir "{B244370E-44C9-4C06-B551-F6016E563076}"
        if (-not (Test-Path $targetDir)) { New-Item -ItemType Directory -Path $targetDir -Force | Out-Null }
        Copy-Item -Path $allowBuildPath -Destination (Join-Path $targetDir "allow_build.xml") -Force
        Write-Host "  Deployed supplemental policy to CI policy store" -ForegroundColor Green
    } catch {
        Write-Host "  WARN: Could not deploy to CI store (run as admin): $_" -ForegroundColor DarkYellow
    }
} else {
    Write-Host "  INFO: CI policy store not found — policy saved for manual deployment" -ForegroundColor DarkYellow
}

# ── Summary ─────────────────────────────────────────────────────────
Write-Host "`n=== Done ===" -ForegroundColor Cyan
Write-Host "Signed: $signed executables"
Write-Host "Cert thumbprint: $certHash"
Write-Host "Policy: $allowBuildPath"
Write-Host ""
Write-Host "To deploy policy (run as admin):" -ForegroundColor Yellow
Write-Host '  1. ConvertTo-CIPolicy ... (or use the allow_build.xml directly)'
Write-Host '  2. Add-CIPolicy -FilePath allow_build.xml -Supplemental'
Write-Host '  3. Set-CIPolicy -FilePath ... -PolicyId ...'
Write-Host ""
Write-Host "If SAC still blocks, disable it:" -ForegroundColor Yellow
Write-Host '  Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\CI" -Name "VerifiedAndReputablePolicyState" -Value 0 -Force'
Write-Host '  # Requires reboot'
