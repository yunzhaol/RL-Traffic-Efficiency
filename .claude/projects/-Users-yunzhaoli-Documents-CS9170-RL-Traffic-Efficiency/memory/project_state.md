---
name: Project State
description: CS9170 RL traffic project — what's built, what's pending, key design decisions
type: project
---

CS9170 RL Traffic Efficiency — group project with Jialin Cai, Zhiyan Chen, Yunzhao Li, An Zhou.

Goal: DQN + PPO for single-intersection traffic signal control. No MARL (dropped from plan).
Simulators: CityFlow + SUMO (both).

**Status as of 2026-04-14:**
- DQN: refactored with Double DQN, Dueling network (128-128), Huber loss, fixed epsilon decay (0.99, was 0.998 — major bug)
- PPO: implemented from scratch with GAE, actor-critic, orthogonal init
- SUMO env: written (env/sumo_traffic_env.py); needs `brew install sumo` + `cd data/sumo && bash setup_sumo.sh`
- CityFlow: pip wheel missing for Python 3.12/macOS; needs source build

**Key fix applied:** epsilon_decay was 0.998 (epsilon=0.82 after 100 eps, agent stayed near-random). Changed to 0.99 (epsilon reaches min 0.05 by ep 300).

**Run order:**
1. `python training/train_dqn.py --sim cityflow` (300 eps)
2. `python training/train_ppo.py --sim cityflow` (300 eps)
3. `brew install sumo` → `cd data/sumo && bash setup_sumo.sh` → repeat with --sim sumo
4. `python training/run_fixed.py` (baseline)
5. `python training/compare_results.py` (plots)

**Why:** Saves to results/{method}_{sim}_rewards/losses.txt + checkpoints/

**How to apply:** New training scripts accept --sim and --episodes args; state is normalized before agent input (STATE_NORM array in train scripts).
