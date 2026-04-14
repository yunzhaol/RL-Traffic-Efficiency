#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all.sh  —  Run full experiment: CityFlow + SUMO, then compare everything
#
# Usage:
#   bash run_all.sh                   # both simulators, 300 episodes each
#   bash run_all.sh --episodes 500    # override episode count for both
#   bash run_all.sh --skip-cityflow   # SUMO only
#   bash run_all.sh --skip-sumo       # CityFlow only
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EPISODES=300
STEPS=200
SKIP_CITYFLOW=0
SKIP_SUMO=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --episodes)      EPISODES="$2";   shift 2 ;;
        --steps)         STEPS="$2";      shift 2 ;;
        --skip-cityflow) SKIP_CITYFLOW=1; shift   ;;
        --skip-sumo)     SKIP_SUMO=1;     shift   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "════════════════════════════════════════════════════════════════"
echo "  Full Experiment  |  episodes=$EPISODES  steps=$STEPS"
echo "════════════════════════════════════════════════════════════════"
echo ""

# ── CityFlow ──────────────────────────────────────────────────────────────────
if [[ $SKIP_CITYFLOW -eq 0 ]]; then
    if python3 -c "import cityflow" 2>/dev/null; then
        echo "══ CityFlow ══════════════════════════════════════════════════"
        bash run_cityflow.sh --episodes "$EPISODES" --steps "$STEPS"
        echo ""
    else
        echo "⚠  CityFlow not installed — skipping CityFlow runs."
        echo "   To install: git clone https://github.com/cityflow-project/CityFlow.git && cd CityFlow && pip3 install ."
        echo ""
    fi
fi

# ── SUMO ──────────────────────────────────────────────────────────────────────
if [[ $SKIP_SUMO -eq 0 ]]; then
    for prefix in /opt/homebrew /usr/local; do
        [[ -f "$prefix/bin/sumo" ]] && export PATH="$prefix/bin:$PATH" && break
    done

    if command -v sumo &>/dev/null; then
        echo "══ SUMO ══════════════════════════════════════════════════════"
        bash run_sumo.sh --episodes "$EPISODES" --steps "$STEPS"
        echo ""
    else
        echo "⚠  SUMO not installed — skipping SUMO runs."
        echo "   To install: brew install sumo"
        echo ""
    fi
fi

# ── Final cross-simulator comparison ─────────────────────────────────────────
echo "══ Cross-simulator comparison ════════════════════════════════"
python3 training/compare_results.py
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "  All done.  Open results/ for plots and reward/loss logs."
echo "════════════════════════════════════════════════════════════════"
