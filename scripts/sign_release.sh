#!/usr/bin/env bash
#
# sign_release.sh — MYTHOS.cpp v0.1.02 Release Signing Script
#
# Steps:
#   1. Generate SHA-256 and MD5 checksums for all build artifacts
#   2. Authenticode sign Windows binaries (placeholder — requires signing cert)
#   3. GPG sign the checksum file (placeholder — requires GPG key)
#   4. Verify all signatures
#   5. Create release archive
#
# Usage:
#   bash scripts/sign_release.sh [build_dir] [output_dir]
#
# Defaults:
#   build_dir  = build/Release
#   output_dir = release/v0.1.02

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

BUILD_DIR="${1:-${PROJECT_ROOT}/build/Release}"
OUTPUT_DIR="${2:-${PROJECT_ROOT}/release/v0.1.02}"
VERSION="0.1.02"

# Authenticode signing certificate path (TODO: set when cert is available)
# Windows: use signtool.exe from Windows SDK
# Linux/Mac: use osslsigncode or skip
AUTHENTICODE_CERT="${AUTHENTICODE_CERT:-}"
AUTHENTICODE_PASSWORD="${AUTHENTICODE_PASSWORD:-}"

# GPG key ID for signing (TODO: set when GPG key is generated)
GPG_KEY_ID="${GPG_KEY_ID:-}"
GPG_PASSPHRASE="${GPG_PASSPHRASE:-}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()   { echo -e "${GREEN}[SIGN]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ─── Validate build directory ───────────────────────────────────────

if [[ ! -d "$BUILD_DIR" ]]; then
    error "Build directory not found: $BUILD_DIR"
fi

log "Build directory: $BUILD_DIR"
log "Output directory: $OUTPUT_DIR"
log "Version: $VERSION"

# ─── Step 1: Collect artifacts ──────────────────────────────────────

log "Step 1: Collecting build artifacts..."

mkdir -p "$OUTPUT_DIR"

ARTIFACTS=()
while IFS= read -r -d '' file; do
    ARTIFACTS+=("$file")
done < <(find "$BUILD_DIR" -maxdepth 2 \( -name "*.exe" -o -name "*.dll" -o -name "*.lib" -o -name "*.so" -o -name "*.a" \) -print0 2>/dev/null || true)

if [[ ${#ARTIFACTS[@]} -eq 0 ]]; then
    warn "No artifacts found in $BUILD_DIR — will generate checksums for available files"
    # Fall back to all files in build dir
    while IFS= read -r -d '' file; do
        ARTIFACTS+=("$file")
    done < <(find "$BUILD_DIR" -maxdepth 2 -type f -print0 2>/dev/null || true)
fi

log "Found ${#ARTIFACTS[@]} artifacts"

# ─── Step 2: Authenticode signing (Windows) ─────────────────────────

log "Step 2: Authenticode signing..."

# TODO: Replace with actual signing when certificate is available
# Example with signtool.exe:
#   signtool sign /f "$AUTHENTICODE_CERT" /p "$AUTHENTICODE_PASSWORD" \
#     /tr http://timestamp.digicert.com /td sha256 /fd sha256 "$file"

AUTHENTICODE_DONE=false
if [[ -n "$AUTHENTICODE_CERT" ]] && [[ -f "$AUTHENTICODE_CERT" ]]; then
    for file in "${ARTIFACTS[@]}"; do
        if [[ "$file" == *.exe ]] || [[ "$file" == *.dll ]]; then
            log "  Signing: $(basename "$file")"
            # TODO: Uncomment when signtool is available in PATH
            # signtool sign /f "$AUTHENTICODE_CERT" /p "$AUTHENTICODE_PASSWORD" \
            #   /tr http://timestamp.digicert.com /td sha256 /fd sha256 "$file"
            echo "  [TODO] Authenticode sign: $file — no signing certificate configured"
        fi
    done
    AUTHENTICODE_DONE=true
else
    warn "  Authenticode certificate not configured (set AUTHENTICODE_CERT env var)"
    warn "  Skipping Authenticode signing — binaries will be unsigned"
    warn "  TODO: Obtain signing certificate from CA (DigiCert, Sectigo, etc.)"
fi

# ─── Step 3: Generate checksums ─────────────────────────────────────

log "Step 3: Generating checksums..."

CHECKSUMS_FILE="${OUTPUT_DIR}/checksums_${VERSION}.txt"

echo "# MYTHOS.cpp v${VERSION} Release Checksums" > "$CHECKSUMS_FILE"
echo "# Generated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$CHECKSUMS_FILE"
echo "# " >> "$CHECKSUMS_FILE"
echo "# SHA-256 checksums:" >> "$CHECKSUMS_FILE"
echo "" >> "$CHECKSUMS_FILE"

for file in "${ARTIFACTS[@]}"; do
    rel_path="${file#${BUILD_DIR}/}"
    if command -v sha256sum &> /dev/null; then
        sha256sum "$file" | sed "s|${BUILD_DIR}/||" >> "$CHECKSUMS_FILE"
    elif command -v shasum &> /dev/null; then
        shasum -a 256 "$file" | sed "s|${BUILD_DIR}/||" >> "$CHECKSUMS_FILE"
    elif command -v powershell &> /dev/null; then
        hash=$(powershell -Command "(Get-FileHash -Algorithm SHA256 '$file').Hash.ToLower()" 2>/dev/null || echo "HASH_ERROR")
        echo "$hash  $rel_path" >> "$CHECKSUMS_FILE"
    else
        warn "  No SHA-256 tool found — recording file sizes only"
        echo "SIZE=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo 0)  $rel_path" >> "$CHECKSUMS_FILE"
    fi
done

echo "" >> "$CHECKSUMS_FILE"
echo "# MD5 checksums:" >> "$CHECKSUMS_FILE"
echo "" >> "$CHECKSUMS_FILE"

for file in "${ARTIFACTS[@]}"; do
    rel_path="${file#${BUILD_DIR}/}"
    if command -v md5sum &> /dev/null; then
        md5sum "$file" | sed "s|${BUILD_DIR}/||" >> "$CHECKSUMS_FILE"
    elif command -v md5 &> /dev/null; then
        md5 "$file" | awk '{print $4, $NF}' | sed "s|${BUILD_DIR}/||" >> "$CHECKSUMS_FILE"
    elif command -v powershell &> /dev/null; then
        hash=$(powershell -Command "(Get-FileHash -Algorithm MD5 '$file').Hash.ToLower()" 2>/dev/null || echo "HASH_ERROR")
        echo "$hash  $rel_path" >> "$CHECKSUMS_FILE"
    fi
done

log "  Checksums written to: $CHECKSUMS_FILE"

# ─── Step 4: GPG signing ────────────────────────────────────────────

log "Step 4: GPG signing..."

# TODO: Replace with actual GPG signing when key is generated
# Example:
#   gpg --batch --yes --passphrase "$GPG_PASSPHRASE" \
#     --local-user "$GPG_KEY_ID" --detach-sign "$CHECKSUMS_FILE"

SIGNATURE_FILE="${CHECKSUMS_FILE}.sig"

if [[ -n "$GPG_KEY_ID" ]] && command -v gpg &> /dev/null; then
    log "  Signing checksums with GPG key: $GPG_KEY_ID"
    # TODO: Uncomment when GPG key is configured
    # gpg --batch --yes --passphrase "$GPG_PASSPHRASE" \
    #   --local-user "$GPG_KEY_ID" --detach-sign "$CHECKSUMS_FILE"
    echo "  [TODO] GPG sign: $CHECKSUMS_FILE — placeholder, not yet signed"
else
    warn "  GPG key not configured (set GPG_KEY_ID env var)"
    warn "  Skipping GPG signing"
    warn "  TODO: Generate GPG key pair:"
    warn "    gpg --full-generate-key"
    warn "    gpg --export --armor YOUR_KEY_ID > mytheros_signing_key.pub"
    echo "  [TODO] GPG signature placeholder — not yet signed" > "$SIGNATURE_FILE"
fi

# ─── Step 5: Verify signatures ──────────────────────────────────────

log "Step 5: Verifying signatures..."

# Verify checksums file exists and is non-empty
if [[ ! -s "$CHECKSUMS_FILE" ]]; then
    error "Checksums file is empty: $CHECKSUMS_FILE"
fi
log "  Checksums file verified: $(wc -l < "$CHECKSUMS_FILE") lines"

# Verify signature file exists
if [[ -f "$SIGNATURE_FILE" ]]; then
    if [[ -n "$GPG_KEY_ID" ]] && command -v gpg &> /dev/null; then
        # TODO: Uncomment when GPG signing is active
        # gpg --verify "$SIGNATURE_FILE" "$CHECKSUMS_FILE"
        # if [[ $? -eq 0 ]]; then
        #     log "  GPG signature: VALID"
        # else
        #     error "  GPG signature: INVALID"
        # fi
        log "  GPG signature file present (not yet cryptographically signed)"
    else
        log "  GPG signature placeholder present"
    fi
else
    warn "  No signature file found"
fi

# Verify SHA-256 checksums
log "  Verifying SHA-256 checksums..."
VERIFY_OK=0
VERIFY_FAIL=0

while IFS= read -r line; do
    # Skip comments and empty lines
    [[ "$line" =~ ^# ]] && continue
    [[ -z "$line" ]] && continue
    # Skip MD5 section
    [[ "$line" =~ ^# ]] && continue

    expected_hash=$(echo "$line" | awk '{print $1}')
    rel_file=$(echo "$line" | awk '{print $2}')
    full_file="${BUILD_DIR}/${rel_file}"

    if [[ ! -f "$full_file" ]]; then
        warn "  File not found: $rel_file"
        ((VERIFY_FAIL++)) || true
        continue
    fi

    if command -v sha256sum &> /dev/null; then
        actual_hash=$(sha256sum "$full_file" | awk '{print $1}')
    elif command -v powershell &> /dev/null; then
        actual_hash=$(powershell -Command "(Get-FileHash -Algorithm SHA256 '$full_file').Hash.ToLower()" 2>/dev/null || echo "")
    else
        warn "  Cannot verify — no SHA-256 tool available"
        break
    fi

    if [[ "$actual_hash" == "$expected_hash" ]]; then
        ((VERIFY_OK++)) || true
    else
        warn "  MISMATCH: $rel_file"
        warn "    Expected: $expected_hash"
        warn "    Actual:   $actual_hash"
        ((VERIFY_FAIL++)) || true
    fi
done < <(sed -n '/^# SHA-256 checksums/,/^#/p' "$CHECKSUMS_FILE" | grep -v '^#')

log "  Verification: $VERIFY_OK passed, $VERIFY_FAIL failed"

# ─── Step 6: Create release archive ─────────────────────────────────

log "Step 6: Creating release archive..."

ARCHIVE_NAME="MYTHOS.cpp-v${VERSION}-release"
ARCHIVE_DIR="${OUTPUT_DIR}"

# Copy checksums to archive dir
cp "$CHECKSUMS_FILE" "$ARCHIVE_DIR/" 2>/dev/null || true
[[ -f "$SIGNATURE_FILE" ]] && cp "$SIGNATURE_FILE" "$ARCHIVE_DIR/" 2>/dev/null || true

# Create archive
if command -v 7z &> /dev/null; then
    ARCHIVE_PATH="${OUTPUT_DIR}/${ARCHIVE_NAME}.zip"
    7z a -tzip "$ARCHIVE_PATH" "$ARCHIVE_DIR" -xr!*.sig 2>/dev/null
    log "  Archive created (7z): $ARCHIVE_PATH"
elif command -v tar &> /dev/null; then
    ARCHIVE_PATH="${OUTPUT_DIR}/${ARCHIVE_NAME}.tar.gz"
    tar -czf "$ARCHIVE_PATH" -C "$OUTPUT_DIR" . 2>/dev/null
    log "  Archive created (tar): $ARCHIVE_PATH"
elif command -v powershell &> /dev/null; then
    ARCHIVE_PATH="${OUTPUT_DIR}/${ARCHIVE_NAME}.zip"
    powershell -Command "Compress-Archive -Path '$ARCHIVE_DIR\*' -DestinationPath '$ARCHIVE_PATH' -Force" 2>/dev/null
    log "  Archive created (PowerShell): $ARCHIVE_PATH"
else
    warn "  No archive tool found — skipping archive creation"
fi

# ─── Summary ────────────────────────────────────────────────────────

echo ""
log "============================================"
log "  MYTHOS.cpp v${VERSION} Release Signing Complete"
log "============================================"
log "  Artifacts:    ${#ARTIFACTS[@]} files"
log "  Checksums:    $CHECKSUMS_FILE"
log "  Signature:    ${SIGNATURE_FILE} (TODO: replace with real GPG sig)"
log "  Authenticode: $([ "$AUTHENTICODE_DONE" = true ] && echo 'configured' || echo 'TODO: requires signing certificate')"
log "  Verify:       $VERIFY_OK passed, $VERIFY_FAIL failed"
[[ -n "${ARCHIVE_PATH:-}" ]] && log "  Archive:      $ARCHIVE_PATH"
log ""
log "  TODO before shipping:"
log "    1. Obtain Authenticode signing certificate (DigiCert / Sectigo)"
log "    2. Generate GPG signing key pair"
log "    3. Set AUTHENTICODE_CERT and GPG_KEY_ID environment variables"
log "    4. Re-run this script with certificates configured"
log "============================================"
