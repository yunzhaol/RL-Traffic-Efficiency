#!/usr/bin/env bash
set -euo pipefail

SIM="sumo"
EPISODES=10
DEMO_METHOD="dqn"
DEMO_STEPS=200
DEMO_DELAY=200
GUI_SETTINGS="data/sumo/gui.settings.xml"
RUN_DEMO=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --sim)         SIM="$2"; shift 2 ;;
        --episodes)    EPISODES="$2"; shift 2 ;;
        --demo-method) DEMO_METHOD="$2"; shift 2 ;;
        --demo-steps)  DEMO_STEPS="$2"; shift 2 ;;
        --demo-delay)  DEMO_DELAY="$2"; shift 2 ;;
        --gui-settings) GUI_SETTINGS="$2"; shift 2 ;;
        --no-demo)     RUN_DEMO=0; shift 1 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "════════════════════════════════════════════════════════"
echo "  SHOWCASE  |  sim=$SIM  eval_episodes=$EPISODES"
echo "════════════════════════════════════════════════════════"
echo ""

echo "▶  [1/3] Quantitative evaluation metrics"
python3 training/evaluate.py --sim "$SIM" --episodes "$EPISODES"
echo ""

echo "▶  [2/3] Signal phase timeline visualization"
python3 training/visualize_phases.py --sim "$SIM"
echo ""

if [[ "$RUN_DEMO" -eq 1 ]]; then
    echo "▶  [3/3] Interactive GUI demo (Ctrl+C to stop)"
    if [[ "$SIM" == "sumo" ]]; then
        # Clean up any stale SUMO processes from earlier stages.
        killall sumo-gui sumo 2>/dev/null || true
        # Match the local Terminal setup that can display SUMO-GUI on macOS.
        if [[ "$(uname)" == "Darwin" ]] && [[ -z "${DISPLAY:-}" ]]; then
            export DISPLAY=:0
        fi
    fi
    python3 training/demo.py --sim "$SIM" --method "$DEMO_METHOD" --steps "$DEMO_STEPS" --delay "$DEMO_DELAY" --gui-settings "$GUI_SETTINGS"
    echo ""
else
    echo "▶  [3/3] Interactive GUI demo skipped (--no-demo)"
    echo ""
fi

echo "════════════════════════════════════════════════════════"
echo "  Showcase outputs:"
echo "    results/evaluation_${SIM}.json"
echo "    results/evaluation_${SIM}.png"
echo "    results/phase_timeline_${SIM}.png"
echo "════════════════════════════════════════════════════════"
