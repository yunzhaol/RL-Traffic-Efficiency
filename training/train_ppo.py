"""
PPO training script — supports both CityFlow and SUMO simulators.

Usage:
    python training/train_ppo.py                  # CityFlow (default)
    python training/train_ppo.py --sim sumo        # SUMO (requires: brew install sumo)
    python training/train_ppo.py --episodes 500    # longer run

PPO is on-policy: each episode's transitions are used once for the update
then discarded. The 'loss' logged is the combined PPO objective
(policy + value + entropy); unlike DQN it does not monotonically decrease —
use the reward curve to assess convergence.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.ppo_agent import PPOAgent

STATE_NORM = np.array(
    [50, 50, 50, 50, 50, 50, 50, 50, 200, 200, 1], dtype=np.float32
)


def normalize(state):
    return np.array(state, dtype=np.float32) / (STATE_NORM + 1e-8)


def train(sim: str, episodes: int, steps_per_episode: int):
    if sim == "cityflow":
        from env.traffic_env import TrafficEnv
        env = TrafficEnv()
    elif sim == "sumo":
        from env.sumo_traffic_env import SumoTrafficEnv
        env = SumoTrafficEnv()
    else:
        raise ValueError(f"Unknown simulator '{sim}'. Choose: cityflow | sumo")

    agent = PPOAgent(
        state_dim=11,
        action_dim=2,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        n_epochs=10,
        mini_batch_size=64,
        hidden_dim=128,
    )

    os.makedirs("results",     exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    episode_rewards: list[float] = []
    episode_losses:  list[float] = []
    best_reward = float("-inf")

    try:
        for ep in range(episodes):
            state        = normalize(env.reset())
            total_reward = 0.0

            for _ in range(steps_per_episode):
                action, log_prob, value       = agent.choose_action(state)
                next_state, reward, done      = env.step(action)
                next_state                    = normalize(next_state)

                agent.store_transition(state, action, log_prob, reward, value, float(done))

                state        = next_state
                total_reward += reward
                if done:
                    break

            # Bootstrap value of the last state for GAE
            with torch.no_grad():
                last_val = agent.network.get_value(
                    torch.FloatTensor(state).unsqueeze(0)
                ).item()

            avg_loss = agent.update(last_val)

            episode_rewards.append(total_reward)
            episode_losses.append(avg_loss)

            if total_reward > best_reward:
                best_reward = total_reward
                agent.save(f"checkpoints/ppo_{sim}_best.pt")

            print(
                f"[PPO/{sim}] Ep {ep + 1:>3}/{episodes} | "
                f"Reward {total_reward:>8.2f} | "
                f"Loss {avg_loss:>8.4f}"
            )

    except KeyboardInterrupt:
        print(f"\n\nInterrupted at episode {len(episode_rewards)}. Saving results so far...")

    finally:
        if hasattr(env, "close"):
            env.close()

    prefix = f"ppo_{sim}"
    with open(f"results/{prefix}_rewards.txt", "w") as f:
        f.writelines(f"{r}\n" for r in episode_rewards)
    with open(f"results/{prefix}_losses.txt", "w") as f:
        f.writelines(f"{l}\n" for l in episode_losses)

    print(f"\nDone. Best reward: {best_reward:.2f}  →  checkpoints/ppo_{sim}_best.pt")
    print(f"Results saved to results/{prefix}_rewards.txt / _losses.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim",      choices=["cityflow", "sumo"], default="cityflow")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--steps",    type=int, default=200)
    args = parser.parse_args()
    train(args.sim, args.episodes, args.steps)
