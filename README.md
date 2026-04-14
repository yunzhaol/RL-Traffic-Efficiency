# CS9170: Optimizing Urban Traffic Efficiency with Reinforcement Learning

## Problem Definition:
Traditional fixed-time traffic signals are inefficient. The goal is to use RL to dynamically control traffic light timing at intersections based on real-time queue lengths and waiting times, minimizing overall vehicle delay and congestion.

- **State**: Traffic conditions (vehicle counts, waiting vehicles, direction, signal phase)
- **Action**: Traffic light phase (e.g., North-South vs East-West green)
- **Reward**: Negative congestion (based on vehicle count and waiting time)

---

## Quick Start

### CityFlow
```bash
# Step 1 — Install CityFlow (one-time, handles cmake + build automatically, ~1 min)
bash install_cityflow.sh

# Step 2 — Run full experiment (Fixed baseline + DQN + PPO + plots)
bash run_cityflow.sh
```

### SUMO
```bash
# Step 1 — Download and install the macOS package (one-time, ~5–15 min)
# https://sumo.dlr.de/docs/Downloads.php  →  macOS  →  download .pkg  →  double-click to install

# Step 2 — Add SUMO to your shell (add to ~/.zshrc, then restart terminal)
export SUMO_HOME="/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO"
export PATH="$PATH:$SUMO_HOME/bin"

# Step 3 (only if GUI cannot open) — install and start XQuartz
brew install --cask xquartz
open -a XQuartz
export DISPLAY=:0
xhost +localhost

# Quick test
sumo-gui -c data/sumo/cross.sumocfg

# Step 4 — Run full experiment (auto-generates network on first run)
bash run_sumo.sh
```

### Both simulators
```bash
bash run_all.sh
```

Optional flags (work on all three scripts):
```bash
bash run_all.sh --episodes 500   # more training
bash run_all.sh --skip-sumo      # CityFlow only
bash run_all.sh --skip-cityflow  # SUMO only
```

Results and plots are saved to `results/`.

### Viewing Results

View results for a single method after training:
```bash
python3 training/plot_results.py --method dqn --sim cityflow
python3 training/plot_results.py --method ppo --sim cityflow
python3 training/plot_results.py --method dqn --sim sumo
python3 training/plot_results.py --method ppo --sim sumo
```

Compare all methods on one plot:
```bash
python3 training/compare_results.py
```

### Showcase (Best for Demo / Presentation)

Run all visualization outputs in one go (metrics + phase timeline + optional live GUI demo):
```bash
bash run_showcase.sh                     # default: sumo
bash run_showcase.sh --sim cityflow      # cityflow version
bash run_showcase.sh --episodes 20       # more stable evaluation metrics
bash run_showcase.sh --no-demo           # only generate plots, no GUI
bash run_showcase.sh --gui-settings data/sumo/gui.settings.xml
```

Note: if SUMO-GUI does not pop up inside Cursor terminal, run demo commands in macOS Terminal.app / iTerm.

Direct commands (if you want each artifact separately):
```bash
# 1) Quantitative metrics comparison (reward / queue / throughput)
python3 training/evaluate.py --sim sumo --episodes 10

# 2) Signal phase decision timeline (fixed vs dqn vs ppo)
python3 training/visualize_phases.py --sim sumo

# 3) Live RL control in SUMO-GUI (most intuitive)
python3 training/demo.py --method dqn --sim sumo --delay 200
python3 training/demo.py --method ppo --sim sumo --delay 200
python3 training/demo.py --method dqn --sim sumo --delay 200 --gui-settings data/sumo/gui.settings.xml
```

Save camera once (so you do not manually Locate/zoom every time):
```bash
# 1) Open SUMO-GUI once, move view to center intersection, set zoom
sumo-gui -c data/sumo/cross.sumocfg

# 2) In GUI menu: File -> Save GUI Settings As...
#    Save to: data/sumo/gui.settings.xml

# 3) Reuse saved camera automatically in demo/showcase
python3 training/demo.py --method dqn --sim sumo --delay 200 --gui-settings data/sumo/gui.settings.xml
bash run_showcase.sh --gui-settings data/sumo/gui.settings.xml
```

Generated files:
- `results/evaluation_sumo.json`
- `results/evaluation_sumo.png`
- `results/phase_timeline_sumo.png`

---

## Methodology

### Simulation Environment
- **CityFlow** 
- **SUMO (Simulation of Urban MObility)**

### RL Model
- **Deep Q-Network (DQN)**
- **Proximal Policy Optimization(PPO)**

### Training
- Epsilon-greedy exploration
- Gradient clipping for stability
- Reward scaling for better convergence

---

