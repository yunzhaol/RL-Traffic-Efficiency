"""
Visualise signal phase decisions over one episode.

Shows two things side by side for Fixed-time, DQN, and PPO:
  • Top row:    phase timeline  (blue = NS green,  orange = EW green)
  • Bottom row: total waiting vehicles over time

This makes it easy to see that the RL agent adapts its phase duration
to traffic conditions, while fixed-time blindly alternates.

Output: results/phase_timeline.png

Usage:
    python training/visualize_phases.py --sim sumo
    python training/visualize_phases.py --sim cityflow
    python training/visualize_phases.py --sim sumo --show
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

STATE_NORM     = np.array([50, 50, 50, 50, 50, 50, 50, 50, 200, 200, 1], dtype=np.float32)
STEPS          = 200
RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "..", "results")
PHASE_COLORS   = {0: "#4c96d7", 1: "#f5a623"}   # blue=NS, orange=EW
PHASE_LABELS   = {0: "NS green", 1: "EW green"}


def normalize(state):
    return np.array(state, dtype=np.float32) / (STATE_NORM + 1e-8)


def make_env(sim: str):
    if sim == "cityflow":
        from env.traffic_env import TrafficEnv
        return TrafficEnv()
    from env.sumo_traffic_env import SumoTrafficEnv
    return SumoTrafficEnv()


def load_agent(method: str, sim: str):
    if method == "fixed":
        return None
    ckpt = f"checkpoints/{method}_{sim}_best.pt"
    if not os.path.exists(ckpt):
        return None
    if method == "dqn":
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent(state_dim=11, action_dim=2, hidden_dim=128, dueling=True)
        agent.load(ckpt)
        agent.epsilon = 0.0
        return agent
    if method == "ppo":
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(state_dim=11, action_dim=2, hidden_dim=128)
        agent.load(ckpt)
        return agent


def get_action(agent, method: str, state, step: int) -> int:
    if method == "fixed":
        return (step // 3) % 2
    if method == "dqn":
        return agent.choose_action(state)
    action, _, _ = agent.choose_action(state)
    return action


def collect_episode(method: str, sim: str):
    """Run one episode, return (phases list, waiting list)."""
    agent = load_agent(method, sim)
    if agent is None and method != "fixed":
        return None, None

    env   = make_env(sim)
    state = normalize(env.reset())
    phases, waiting = [], []

    for step in range(STEPS):
        action                   = get_action(agent, method, state, step)
        next_state, _, done      = env.step(action)
        state                    = normalize(next_state)
        phases.append(action)
        waiting.append(next_state[9])   # total_waiting
        if done:
            break

    if hasattr(env, "close"):
        env.close()
    return phases, waiting


def draw_phase_bar(ax, phases, title):
    """Draw a horizontal color bar where each step is colored by phase."""
    n = len(phases)
    for i, phase in enumerate(phases):
        ax.barh(0, 1, left=i, height=1,
                color=PHASE_COLORS[phase], linewidth=0)
    ax.set_xlim(0, n)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Step")
    ax.set_title(title, fontweight="bold", fontsize=11)

    ns_patch  = mpatches.Patch(color=PHASE_COLORS[0], label="NS green")
    ew_patch  = mpatches.Patch(color=PHASE_COLORS[1], label="EW green")
    ax.legend(handles=[ns_patch, ew_patch], loc="upper right",
              fontsize=8, framealpha=0.8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", choices=["cityflow", "sumo"], default="sumo")
    parser.add_argument("--show", action="store_true",
                        help="Display matplotlib window after saving chart")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    methods = ["fixed", "dqn", "ppo"]
    titles  = {"fixed": "Fixed-time (baseline)", "dqn": "DQN (trained)", "ppo": "PPO (trained)"}

    data = {}
    for method in methods:
        print(f"Collecting episode: {method.upper()} on {args.sim} ...")
        phases, waiting = collect_episode(method, args.sim)
        if phases is None:
            print(f"  No checkpoint for {method} — skipping.")
            continue
        data[method] = (phases, waiting)

    if not data:
        print("No data collected. Run training first.")
        return

    n_cols  = len(data)
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 6),
                              gridspec_kw={"height_ratios": [1, 3]})
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(f"Signal Phase Decisions — {args.sim.upper()}",
                 fontsize=13, fontweight="bold")

    for col, (method, (phases, waiting)) in enumerate(data.items()):
        # Phase timeline bar
        draw_phase_bar(axes[0][col], phases, titles[method])

        # Waiting vehicles line
        ax = axes[1][col]
        x  = list(range(1, len(waiting) + 1))
        ax.plot(x, waiting, color="#2c7bb6", linewidth=1.5, alpha=0.85)
        ax.fill_between(x, waiting, alpha=0.15, color="#2c7bb6")
        ax.set_xlabel("Step")
        ax.set_ylabel("Waiting vehicles" if col == 0 else "")
        ax.set_title("Queue Length Over Time", fontsize=10)
        ax.grid(True, alpha=0.25)

        avg_q = np.mean(waiting)
        ax.axhline(avg_q, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(len(waiting) * 0.02, avg_q * 1.02, f"avg={avg_q:.1f}",
                color="red", fontsize=8)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"phase_timeline_{args.sim}.png")
    plt.savefig(out, dpi=150)
    print(f"\nSaved: {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
