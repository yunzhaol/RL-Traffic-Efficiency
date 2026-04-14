#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_sumo.sh  —  Generate SUMO network from source XML files
#
# Run ONCE after installing SUMO:
#   1. Download from https://sumo.dlr.de/docs/Downloads.php  (macOS .pkg)
#   2. Add to ~/.zshrc:
#        export SUMO_HOME="/usr/local/share/sumo"
#        export PATH="$PATH:$SUMO_HOME/bin"
#   3. cd data/sumo && bash setup_sumo.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Locate netconvert ─────────────────────────────────────────────────────────
if command -v netconvert &>/dev/null; then
    NETCONVERT="netconvert"
elif [[ -f "/opt/homebrew/bin/netconvert" ]]; then
    NETCONVERT="/opt/homebrew/bin/netconvert"
elif [[ -f "/usr/local/bin/netconvert" ]]; then
    NETCONVERT="/usr/local/bin/netconvert"
elif [[ -n "${SUMO_HOME:-}" && -f "$SUMO_HOME/bin/netconvert" ]]; then
    NETCONVERT="$SUMO_HOME/bin/netconvert"
else
    echo "Error: netconvert not found."
    echo ""
    echo "Install SUMO from: https://sumo.dlr.de/docs/Downloads.php"
    echo "Then add to ~/.zshrc:"
    echo '  export SUMO_HOME="/usr/local/share/sumo"'
    echo '  export PATH="$PATH:$SUMO_HOME/bin"'
    exit 1
fi

echo "Using netconvert: $NETCONVERT"
echo ""

# ── Generate network ──────────────────────────────────────────────────────────
echo "Generating cross.net.xml from node/edge definitions..."
"$NETCONVERT" \
    --node-files=cross.nod.xml \
    --edge-files=cross.edg.xml \
    --output-file=cross.net.xml \
    --no-turnarounds true \
    --tls.default-type static \
    --no-warnings

echo ""
echo "Done. Files in data/sumo/:"
ls -lh cross.net.xml cross.rou.xml cross.sumocfg
echo ""
echo "You can now run SUMO training:"
echo "  python training/train_dqn.py --sim sumo"
echo "  python training/train_ppo.py --sim sumo"
