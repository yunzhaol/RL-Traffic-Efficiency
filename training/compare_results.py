"""
Compare training results across all methods and simulators.

Reads reward/loss files from results/ and produces:
  results/comparison_rewards.png  — reward curves (raw + smoothed)
  results/comparison_losses.png   — loss curves (DQN and PPO)

Usage:
    python training/compare_results.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ── File registry ─────────────────────────────────────────────────────────────
# (label, reward_file, loss_file, color, linestyle)
METHODS = [
    ("DQN – CityFlow",  "dqn_cityflow_rewards.txt",   "dqn_cityflow_losses.txt",  "tab:blue",   "-"),
    ("PPO – CityFlow",  "ppo_cityflow_rewards.txt",   "ppo_cityflow_losses.txt",  "tab:orange", "-"),
    ("DQN – SUMO",      "dqn_sumo_rewards.txt",       "dqn_sumo_losses.txt",      "tab:green",  "--"),
    ("PPO – SUMO",      "ppo_sumo_rewards.txt",       "ppo_sumo_losses.txt",      "tab:red",    "--"),
    ("Fixed – CityFlow","fixed_cityflow_rewards.txt",  None,                       "tab:gray",   ":"),
    ("Fixed – SUMO",    "fixed_sumo_rewards.txt",      None,                       "black",      ":"),
]


def load(filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    return [float(x) for x in lines]


def smooth(values, window: int = 15):
    if len(values) < window:
        return np.array(values), list(range(len(values)))
    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
    x = list(range(window - 1, len(values)))
    return smoothed, x


def plot_panel(ax, datasets, title, ylabel):
    any_plotted = False
    for label, values, color, ls in datasets:
        if values is None:
            continue
        x_raw = list(range(1, len(values) + 1))
        ax.plot(x_raw, values, color=color, alpha=0.18, linewidth=0.8)
        s, sx = smooth(values)
        ax.plot([v + 1 for v in sx], s,
                label=label, color=color, linestyle=ls, linewidth=2.0)
        any_plotted = True

    if not any_plotted:
        ax.text(0.5, 0.5, "No data yet.\nRun training first.",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    reward_data = []
    loss_data   = []

    for label, r_file, l_file, color, ls in METHODS:
        rewards = load(r_file)
        losses  = load(l_file) if l_file else None
        reward_data.append((label, rewards,  color, ls))
        if losses is not None:
            loss_data.append((label, losses, color, ls))

    # ── Reward comparison ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_panel(ax, reward_data, "Episode Rewards — All Methods", "Total Reward per Episode")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "comparison_rewards.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()

    # ── Loss comparison ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_panel(ax, loss_data, "Training Loss — DQN & PPO", "Average Loss per Episode")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "comparison_losses.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()

    # ── Per-simulator side-by-side ─────────────────────────────────────────
    for sim_name in ["CityFlow", "SUMO"]:
        sim_rewards = [(lbl, v, c, ls)
                       for lbl, v, c, ls in reward_data
                       if sim_name.lower() in lbl.lower()]
        if not any(v is not None for _, v, _, _ in sim_rewards):
            continue
        fig, ax = plt.subplots(figsize=(9, 4))
        plot_panel(ax, sim_rewards,
                   f"Reward Comparison — {sim_name}",
                   "Total Reward per Episode")
        plt.tight_layout()
        out = os.path.join(RESULTS_DIR, f"comparison_{sim_name.lower()}.png")
        plt.savefig(out, dpi=150)
        print(f"Saved: {out}")
        plt.close()


if __name__ == "__main__":
    main()
