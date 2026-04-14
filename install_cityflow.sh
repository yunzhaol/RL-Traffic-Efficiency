#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# install_cityflow.sh  —  Build and install CityFlow from source (macOS)
#
# Fixes applied automatically:
#   - Installs cmake via Homebrew if missing
#   - Replaces CityFlow's bundled pybind11 v2.3.0 with v2.13.6
#     (v2.3.0 is incompatible with Python 3.12 — PyFrameObject removed)
#
# Usage:
#   bash install_cityflow.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "▶  Checking dependencies..."

# cmake
if ! command -v cmake &>/dev/null; then
    echo "   cmake not found — installing via Homebrew..."
    brew install cmake
fi
echo "   cmake: $(cmake --version | head -1)"

# Xcode CLT
if ! xcode-select -p &>/dev/null; then
    echo "   Xcode Command Line Tools not found — installing..."
    xcode-select --install
    echo "   Re-run this script after the Xcode CLT install finishes."
    exit 0
fi
echo "   Xcode CLT: $(xcode-select -p)"
echo ""

# ── Clone CityFlow ────────────────────────────────────────────────────────────
if [[ -d "CityFlow" ]]; then
    echo "▶  CityFlow directory already exists, pulling latest..."
    git -C CityFlow pull
else
    echo "▶  Cloning CityFlow..."
    git clone https://github.com/cityflow-project/CityFlow.git
fi
echo ""

cd CityFlow

# ── Patch: replace bundled pybind11 2.3.0 with 2.13.6 ────────────────────────
# pybind11 < 2.11 is incompatible with Python 3.12 (PyFrameObject removed).
echo "▶  Patching pybind11 (bundled v2.3.0 → v2.13.6 for Python 3.12 support)..."
rm -rf extern/pybind11
git clone --branch v2.13.6 --depth 1 \
    https://github.com/pybind/pybind11.git extern/pybind11
echo ""

# ── Configure ─────────────────────────────────────────────────────────────────
echo "▶  Running cmake..."
rm -rf CMakeCache.txt CMakeFiles
cmake . \
    -DPYTHON_EXECUTABLE="$(which python3)" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
echo ""

# ── Build ─────────────────────────────────────────────────────────────────────
echo "▶  Building CityFlow (this takes ~1 minute)..."
NPROC=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
make -j"$NPROC"
echo ""

# ── Install .so into site-packages ───────────────────────────────────────────
SO_FILE=$(ls cityflow.cpython-*.so 2>/dev/null | head -1)
if [[ -z "$SO_FILE" ]]; then
    echo "ERROR: .so file not found after build. Check errors above."
    exit 1
fi

SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
echo "▶  Installing $SO_FILE → $SITE_PACKAGES"
cp "$SO_FILE" "$SITE_PACKAGES/"

cd ..

# ── Verify ────────────────────────────────────────────────────────────────────
python3 -c "import cityflow; print('CityFlow installed successfully:', cityflow.Engine)" || {
    echo "ERROR: CityFlow import failed after build. Check errors above."
    exit 1
}
echo ""
echo "════════════════════════════════════════════════════════"
echo "  CityFlow ready.  Run: bash run_cityflow.sh"
echo "════════════════════════════════════════════════════════"
