"""
Fixed-time signal baseline — alternates NS/EW every 3 steps (30 sim-seconds each).
Serves as the lower-bound comparison for DQN and PPO.

Usage:
    python training/run_fixed.py                # CityFlow (default)
    python training/run_fixed.py --sim sumo     # SUMO
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_fixed(sim: str, episodes: int, steps_per_episode: int, phase_hold: int = 3):
    if sim == "cityflow":
        from env.traffic_env import TrafficEnv
        env = TrafficEnv()
    elif sim == "sumo":
        from env.sumo_traffic_env import SumoTrafficEnv
        env = SumoTrafficEnv()
    else:
        raise ValueError(f"Unknown simulator '{sim}'")

    rewards = []

    try:
        for ep in range(episodes):
            env.reset()
            total_reward = 0.0

            for step in range(steps_per_episode):
                action = (step // phase_hold) % 2
                _, reward, done = env.step(action)
                total_reward += reward
                if done:
                    break

            rewards.append(total_reward)
            print(f"[Fixed/{sim}] Ep {ep + 1:>3}/{episodes} | Reward {total_reward:.2f}")

    except KeyboardInterrupt:
        print(f"\n\nInterrupted at episode {len(rewards)}. Saving results so far...")

    finally:
        if hasattr(env, "close"):
            env.close()

    if rewards:
        avg = sum(rewards) / len(rewards)
        print(f"\nAverage reward: {avg:.2f}")

    os.makedirs("results", exist_ok=True)
    out_path = f"results/fixed_{sim}_rewards.txt"
    with open(out_path, "w") as f:
        f.writelines(f"{r}\n" for r in rewards)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim",      choices=["cityflow", "sumo"], default="cityflow")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps",    type=int, default=200)
    args = parser.parse_args()
    run_fixed(args.sim, args.episodes, args.steps)
