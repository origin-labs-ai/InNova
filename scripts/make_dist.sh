#!/usr/bin/env bash
# MYTHOS.cpp distribution builder
# Usage: bash scripts/make_dist.sh [version]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION="${1:-v0.1.0-engine-prod}"
OUTDIR="dist"
mkdir -p "$OUTDIR/source"

echo "=== Building distribution: $VERSION ==="

# --- Platform detection ---
case "$(uname -s)" in
    Linux*)  PLAT="linux-x86_64" ;;
    Darwin*) PLAT="macos-arm64"  ;;
    CYGWIN*|MINGW*|MSYS*) PLAT="windows-x64" ;;
    *)       echo "Unknown OS"; exit 1 ;;
esac

echo "Platform: $PLAT"

# --- Build ---
cmake -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DOIL_BUILD_TESTS=ON -DOIL_BUILD_BENCHMARKS=ON
cmake --build build-release --parallel

# --- Copy binaries ---
mkdir -p "$OUTDIR/$PLAT"
if [ "$PLAT" = "windows-x64" ]; then
    find build-release -maxdepth 2 -name "*.exe" -exec cp {} "$OUTDIR/$PLAT/" \;
else
    find build-release -maxdepth 2 -type f -executable -exec cp {} "$OUTDIR/$PLAT/" \;
fi

# --- SHA256SUMS ---
cd "$OUTDIR/$PLAT"
rm -f SHA256SUMS
sha256sum * > SHA256SUMS
cd "$ROOT"

# --- Source tarball ---
tar --exclude='.git' --exclude='build*' --exclude='dist' --exclude='.kilo' \
    --exclude='.research' --exclude='.github' \
    -czf "$OUTDIR/mythos-$VERSION-source.tar.gz" \
    CMakeLists.txt LICENSE README.md AGENTS.md \
    src/ include/ engines/ tests/ bench/ tools/

cd "$OUTDIR"
sha256sum "mythos-$VERSION-source.tar.gz" > "mythos-$VERSION-source.tar.gz.sha256"
cd "$ROOT"

echo "=== Distribution built at $OUTDIR/ ==="
echo "Binaries: $OUTDIR/$PLAT/"
echo "Source:   $OUTDIR/mythos-$VERSION-source.tar.gz"
