#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_sumo.sh  —  Setup SUMO network (if needed), train DQN + PPO + Fixed, plot
#
# Prerequisites:
#   1. Download and install SUMO from https://sumo.dlr.de/docs/Downloads.php
#   2. Add to ~/.zshrc:
#        export SUMO_HOME="/usr/local/share/sumo"
#        export PATH="$PATH:$SUMO_HOME/bin"
#
# Usage:
#   bash run_sumo.sh
#   bash run_sumo.sh --episodes 500
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EPISODES=300
STEPS=200

while [[ $# -gt 0 ]]; do
    case $1 in
        --episodes) EPISODES="$2"; shift 2 ;;
        --steps)    STEPS="$2";    shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "════════════════════════════════════════════════════════"
echo "  SUMO Training  |  episodes=$EPISODES  steps=$STEPS"
echo "════════════════════════════════════════════════════════"
echo ""

# ── Dependency checks ─────────────────────────────────────────────────────────
python3 -c "import traci" 2>/dev/null || {
    echo "ERROR: traci not found.  Run: pip3 install traci sumolib"; exit 1
}

# Add SUMO to PATH — check common install locations
SUMO_CANDIDATES=(
    "$HOME/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/bin"
    "/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/bin"
    "/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/bin"
    "${SUMO_HOME:-}/bin"
    "/usr/local/share/sumo/bin"
    "/opt/homebrew/bin"
    "/usr/local/bin"
)
for candidate in "${SUMO_CANDIDATES[@]}"; do
    if [[ -f "$candidate/sumo" ]]; then
        export PATH="$PATH:$candidate"
        export SUMO_HOME="$(dirname "$candidate")"
        break
    fi
done

command -v sumo &>/dev/null || {
    echo "ERROR: sumo binary not found."
    echo ""
    echo "Install SUMO:"
    echo "  1. Download the macOS .pkg from https://sumo.dlr.de/docs/Downloads.php"
    echo "  2. Double-click to install"
    echo "  3. Add to ~/.zshrc:"
    echo '       export SUMO_HOME="/usr/local/share/sumo"'
    echo '       export PATH="$PATH:$SUMO_HOME/bin"'
    echo "  4. Restart terminal and re-run: bash run_sumo.sh"
    exit 1
}

echo "sumo binary: $(which sumo)"
echo ""

# ── Generate SUMO network (once) ──────────────────────────────────────────────
NET_FILE="data/sumo/cross.net.xml"
if [[ ! -f "$NET_FILE" ]]; then
    echo "▶  Generating SUMO network (one-time setup)..."
    bash data/sumo/setup_sumo.sh
    echo ""
else
    echo "▶  SUMO network already exists, skipping setup."
    echo ""
fi

# Filter out noisy traci/SUMO connection messages
_run() { python3 "$@" 2> >(grep -v "Retrying in" | grep -v "SUMO_HOME" >&2); }

# ── Fixed-time baseline ───────────────────────────────────────────────────────
echo "▶  [1/3] Fixed-time baseline"
_run training/run_fixed.py --sim sumo --episodes 50 --steps "$STEPS"
echo ""

# ── DQN ───────────────────────────────────────────────────────────────────────
echo "▶  [2/3] DQN (Double + Dueling + Huber)"
_run training/train_dqn.py --sim sumo --episodes "$EPISODES" --steps "$STEPS"
echo ""

# ── PPO ───────────────────────────────────────────────────────────────────────
echo "▶  [3/3] PPO (GAE + clipped surrogate)"
_run training/train_ppo.py --sim sumo --episodes "$EPISODES" --steps "$STEPS"
echo ""

# ── Plots ─────────────────────────────────────────────────────────────────────
echo "▶  Generating comparison plots"
python3 training/compare_results.py
echo ""

echo "════════════════════════════════════════════════════════"
echo "  Done.  Results in: results/"
echo "  Plots:  results/comparison_sumo.png"
echo "          results/comparison_rewards.png"
echo "════════════════════════════════════════════════════════"
