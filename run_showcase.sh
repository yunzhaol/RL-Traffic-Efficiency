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

# ── Resolve SUMO_HOME + PATH (same logic as run_sumo.sh) ──────────────────────
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

# Filter out noisy traci/SUMO connection messages
_run() { python3 "$@" 2> >(grep -v "Retrying in" | grep -v "SUMO_HOME" >&2); }

# Auto-detect Cursor / VS Code integrated terminal — GUI cannot open there
IN_CURSOR=0
if [[ "${TERM_PROGRAM:-}" == "vscode" ]]; then
    IN_CURSOR=1
fi

echo "════════════════════════════════════════════════════════"
echo "  SHOWCASE  |  sim=$SIM  eval_episodes=$EPISODES"
echo "════════════════════════════════════════════════════════"
echo ""

echo "▶  [1/3] Quantitative evaluation metrics"
_run training/evaluate.py --sim "$SIM" --episodes "$EPISODES"
echo ""

echo "▶  [2/3] Signal phase timeline visualization"
_run training/visualize_phases.py --sim "$SIM"
echo ""

DEMO_CMD="python3 training/demo.py --sim $SIM --method $DEMO_METHOD --steps $DEMO_STEPS --delay $DEMO_DELAY --gui-settings $GUI_SETTINGS"

if [[ "$RUN_DEMO" -eq 0 ]]; then
    echo "▶  [3/3] GUI demo skipped (--no-demo)"
    echo "   To run manually in Terminal.app:"
    echo "   cd $(pwd) && $DEMO_CMD"
    echo ""
elif [[ "$IN_CURSOR" -eq 1 ]]; then
    echo "▶  [3/3] GUI demo — opening Terminal.app window automatically..."
    echo ""
    if [[ "$OSTYPE" == "darwin"* ]] && command -v osascript &>/dev/null; then
        osascript -e "tell application \"Terminal\"
            activate
            do script \"cd $(pwd) && $DEMO_CMD\"
        end tell"
        echo "   SUMO-GUI demo launched in Terminal.app."
    else
        echo "   Run this command in your terminal to launch the demo:"
        echo "   cd $(pwd) && $DEMO_CMD"
    fi
    echo ""
else
    echo "▶  [3/3] Interactive GUI demo (Ctrl+C to stop)"
    eval "$DEMO_CMD" 2> >(grep -v "Retrying in" | grep -v "SUMO_HOME" >&2)
    echo ""
fi

echo "════════════════════════════════════════════════════════"
echo "  Showcase outputs:"
echo "    results/evaluation_${SIM}.json"
echo "    results/evaluation_${SIM}.png"
echo "    results/phase_timeline_${SIM}.png"
echo "════════════════════════════════════════════════════════"
