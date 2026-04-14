"""
Plot rewards and losses for a single method.

Usage:
    python training/plot_results.py --method dqn --sim cityflow
    python training/plot_results.py --method ppo --sim cityflow
    python training/plot_results.py --method dqn --sim sumo
    python training/plot_results.py --method ppo --sim sumo
    python training/plot_results.py --method fixed --sim cityflow
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [float(l.strip()) for l in f if l.strip()]


def smooth(values, window=15):
    if len(values) < window:
        return values, list(range(len(values)))
    s = np.convolve(values, np.ones(window) / window, mode="valid")
    return s, list(range(window - 1, len(values)))


def plot(method, sim):
    prefix = f"{method}_{sim}"
    rewards = load(f"{prefix}_rewards.txt")
    losses  = load(f"{prefix}_losses.txt")

    if rewards is None:
        print(f"No reward file found: results/{prefix}_rewards.txt")
        print("Run training first.")
        return

    n_plots = 2 if losses else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    # Reward plot
    ax = axes[0]
    x = list(range(1, len(rewards) + 1))
    ax.plot(x, rewards, alpha=0.3, color="tab:blue", label="raw")
    s, sx = smooth(rewards)
    ax.plot([v + 1 for v in sx], s, color="tab:blue", linewidth=2, label="smoothed")
    ax.set_title(f"{method.upper()} / {sim}  —  Episode Rewards", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True, alpha=0.25)

    # Loss plot
    if losses:
        ax = axes[1]
        x = list(range(1, len(losses) + 1))
        ax.plot(x, losses, alpha=0.3, color="tab:orange", label="raw")
        s, sx = smooth(losses)
        ax.plot([v + 1 for v in sx], s, color="tab:orange", linewidth=2, label="smoothed")
        ax.set_title(f"{method.upper()} / {sim}  —  Training Loss", fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average Loss")
        ax.legend()
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"{prefix}_plot.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["dqn", "ppo", "fixed"], required=True)
    parser.add_argument("--sim",    choices=["cityflow", "sumo"],     required=True)
    args = parser.parse_args()
    plot(args.method, args.sim)
