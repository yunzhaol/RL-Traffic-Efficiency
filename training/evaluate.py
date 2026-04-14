"""
Evaluate all trained methods and produce a metrics comparison bar chart.

Metrics reported per method:
  • Average episode reward
  • Average queue length  (waiting vehicles per step)
  • Throughput            (vehicles completing route, SUMO only)
  • % improvement in queue vs fixed-time baseline

Output:
  results/evaluation_metrics.json   — raw numbers
  results/evaluation_bar_chart.png  — comparison bar chart

Usage:
    python training/evaluate.py --sim sumo
    python training/evaluate.py --sim cityflow
    python training/evaluate.py --sim sumo --episodes 20
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

STATE_NORM = np.array([50, 50, 50, 50, 50, 50, 50, 50, 200, 200, 1], dtype=np.float32)
STEPS_PER_EPISODE = 200
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


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
        return None   # skip missing checkpoints gracefully
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
    if method == "ppo":
        action, _, _ = agent.choose_action(state)
        return action


def run_episodes(method: str, sim: str, n_episodes: int):
    agent = load_agent(method, sim)
    if agent is None and method != "fixed":
        print(f"  [{method}/{sim}] No checkpoint found — skipping.")
        return None

    env = make_env(sim)
    rewards, queues, throughputs = [], [], []

    for ep in range(n_episodes):
        state        = normalize(env.reset())
        total_reward = 0.0
        step_queues  = []

        for step in range(STEPS_PER_EPISODE):
            action                   = get_action(agent, method, state, step)
            next_state, reward, done = env.step(action)
            state                    = normalize(next_state)
            total_reward            += reward
            step_queues.append(next_state[9])   # total_waiting from state
            if done:
                break

        rewards.append(total_reward)
        queues.append(float(np.mean(step_queues)))

        # Throughput only available for SUMO
        if sim == "sumo" and hasattr(env, "get_throughput"):
            throughputs.append(env.get_throughput())

        print(f"  [{method}/{sim}] Ep {ep + 1}/{n_episodes} | "
              f"Reward {total_reward:>8.2f} | Avg queue {queues[-1]:.1f}")

    if hasattr(env, "close"):
        env.close()

    return {
        "avg_reward":     float(np.mean(rewards)),
        "std_reward":     float(np.std(rewards)),
        "avg_queue":      float(np.mean(queues)),
        "std_queue":      float(np.std(queues)),
        "avg_throughput": float(np.mean(throughputs)) if throughputs else None,
    }


def plot_bar_chart(all_results: dict, sim: str):
    methods = [m for m, r in all_results.items() if r is not None]
    if not methods:
        print("No results to plot.")
        return

    colors = {"fixed": "tab:gray", "dqn": "tab:blue", "ppo": "tab:orange"}
    labels = {"fixed": "Fixed-time", "dqn": "DQN", "ppo": "PPO"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"Performance Comparison — {sim.upper()}", fontsize=13, fontweight="bold")

    # ── Avg reward ────────────────────────────────────────────────────────
    ax = axes[0]
    vals = [all_results[m]["avg_reward"] for m in methods]
    errs = [all_results[m]["std_reward"] for m in methods]
    bars = ax.bar([labels[m] for m in methods], vals, yerr=errs,
                  color=[colors[m] for m in methods], capsize=5, alpha=0.85)
    ax.set_title("Avg Episode Reward", fontweight="bold")
    ax.set_ylabel("Total Reward")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    # ── Avg queue length ──────────────────────────────────────────────────
    ax = axes[1]
    vals = [all_results[m]["avg_queue"] for m in methods]
    errs = [all_results[m]["std_queue"] for m in methods]

    # Compute % improvement vs fixed-time
    fixed_queue = all_results.get("fixed", {})
    fixed_q     = fixed_queue.get("avg_queue", None) if fixed_queue else None

    bars = ax.bar([labels[m] for m in methods], vals, yerr=errs,
                  color=[colors[m] for m in methods], capsize=5, alpha=0.85)
    ax.set_title("Avg Queue Length (waiting vehicles)", fontweight="bold")
    ax.set_ylabel("Vehicles")
    ax.grid(axis="y", alpha=0.3)
    for bar, val, m in zip(bars, vals, methods):
        label = f"{val:.1f}"
        if fixed_q and m != "fixed" and fixed_q > 0:
            pct = (fixed_q - val) / fixed_q * 100
            label += f"\n({pct:+.1f}%)"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                label, ha="center", va="bottom", fontsize=8)

    # ── Throughput (SUMO only) ────────────────────────────────────────────
    ax = axes[2]
    tp_methods = [m for m in methods if all_results[m].get("avg_throughput") is not None]
    if tp_methods:
        tp_vals = [all_results[m]["avg_throughput"] for m in tp_methods]
        bars = ax.bar([labels[m] for m in tp_methods], tp_vals,
                      color=[colors[m] for m in tp_methods], alpha=0.85)
        ax.set_title("Avg Throughput (vehicles/episode)", fontweight="bold")
        ax.set_ylabel("Vehicles completed")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, tp_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    else:
        ax.text(0.5, 0.5, "Throughput available\nfor SUMO only",
                ha="center", va="center", transform=ax.transAxes, fontsize=10,
                color="gray")
        ax.set_title("Throughput", fontweight="bold")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"evaluation_{sim}.png")
    plt.savefig(out, dpi=150)
    print(f"\nBar chart saved: {out}")
    plt.show()


def print_summary(all_results: dict, sim: str):
    fixed = all_results.get("fixed")
    print("\n" + "═" * 60)
    print(f"  EVALUATION SUMMARY — {sim.upper()}")
    print("═" * 60)
    print(f"{'Method':<12} {'Avg Reward':>12} {'Avg Queue':>12} {'Throughput':>12}")
    print("─" * 60)
    for method, res in all_results.items():
        if res is None:
            continue
        tp = f"{res['avg_throughput']:.0f}" if res.get("avg_throughput") else "N/A"
        imp = ""
        if fixed and method != "fixed" and fixed["avg_queue"] > 0:
            pct = (fixed["avg_queue"] - res["avg_queue"]) / fixed["avg_queue"] * 100
            imp = f"  ({pct:+.1f}% queue)"
        label = {"fixed": "Fixed-time", "dqn": "DQN", "ppo": "PPO"}.get(method, method)
        print(f"{label:<12} {res['avg_reward']:>12.2f} {res['avg_queue']:>12.1f} {tp:>12}{imp}")
    print("═" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim",      choices=["cityflow", "sumo"], default="sumo")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Evaluation episodes per method (10 recommended)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Evaluating all methods on {args.sim}  ({args.episodes} episodes each)\n")

    all_results = {}
    for method in ["fixed", "dqn", "ppo"]:
        print(f"── {method.upper()} ──────────────────────────────────")
        try:
            all_results[method] = run_episodes(method, args.sim, args.episodes)
        except Exception as e:
            print(f"  Error: {e}")
            all_results[method] = None
        print()

    # Save raw metrics
    out_json = os.path.join(RESULTS_DIR, f"evaluation_{args.sim}.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Raw metrics saved: {out_json}")

    print_summary(all_results, args.sim)
    plot_bar_chart(all_results, args.sim)


if __name__ == "__main__":
    main()
