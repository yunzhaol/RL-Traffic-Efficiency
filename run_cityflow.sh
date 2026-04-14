#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_cityflow.sh  —  Train DQN + PPO + Fixed baseline on CityFlow, then plot
#
# Usage:
#   bash run_cityflow.sh
#   bash run_cityflow.sh --episodes 500   # override episode count
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

# Always run from project root
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "════════════════════════════════════════════════════════"
echo "  CityFlow Training  |  episodes=$EPISODES  steps=$STEPS"
echo "════════════════════════════════════════════════════════"
echo ""

# ── Dependency check ──────────────────────────────────────────────────────────
python3 -c "import cityflow" 2>/dev/null || {
    echo "ERROR: CityFlow not installed."
    echo ""
    echo "Build from source:"
    echo "  git clone https://github.com/cityflow-project/CityFlow.git"
    echo "  cd CityFlow && pip3 install ."
    exit 1
}

# ── Fixed-time baseline ───────────────────────────────────────────────────────
echo "▶  [1/3] Fixed-time baseline"
python3 training/run_fixed.py --sim cityflow --episodes 50 --steps "$STEPS"
echo ""

# ── DQN ───────────────────────────────────────────────────────────────────────
echo "▶  [2/3] DQN (Double + Dueling + Huber)"
python3 training/train_dqn.py --sim cityflow --episodes "$EPISODES" --steps "$STEPS"
echo ""

# ── PPO ───────────────────────────────────────────────────────────────────────
echo "▶  [3/3] PPO (GAE + clipped surrogate)"
python3 training/train_ppo.py --sim cityflow --episodes "$EPISODES" --steps "$STEPS"
echo ""

# ── Plots ─────────────────────────────────────────────────────────────────────
echo "▶  Generating comparison plots"
python3 training/compare_results.py
echo ""

echo "════════════════════════════════════════════════════════"
echo "  Done.  Results in: results/"
echo "  Plots:  results/comparison_cityflow.png"
echo "          results/comparison_rewards.png"
echo "════════════════════════════════════════════════════════"
