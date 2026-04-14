"""
DQN training script — supports both CityFlow and SUMO simulators.

Usage:
    python training/train_dqn.py                  # CityFlow (default)
    python training/train_dqn.py --sim sumo        # SUMO (requires: brew install sumo)
    python training/train_dqn.py --episodes 500    # longer run
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.dqn_agent import DQNAgent

# ── State normalisation ──────────────────────────────────────────────────────
# State layout (11 dims):
#   [veh_W, veh_S, veh_E, veh_N,   (vehicle counts per approach)
#    wait_W, wait_S, wait_E, wait_N, (waiting vehicle counts)
#    total_veh, total_wait, phase]
STATE_NORM = np.array(
    [50, 50, 50, 50, 50, 50, 50, 50, 200, 200, 1], dtype=np.float32
)


def normalize(state):
    return np.array(state, dtype=np.float32) / (STATE_NORM + 1e-8)


# ── Training loop ─────────────────────────────────────────────────────────────
def train(sim: str, episodes: int, steps_per_episode: int):
    if sim == "cityflow":
        from env.traffic_env import TrafficEnv
        env = TrafficEnv()
    elif sim == "sumo":
        from env.sumo_traffic_env import SumoTrafficEnv
        env = SumoTrafficEnv()
    else:
        raise ValueError(f"Unknown simulator '{sim}'. Choose: cityflow | sumo")

    agent = DQNAgent(
        state_dim=11,
        action_dim=2,
        lr=1e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.99,      # ε reaches ~0.05 by episode 300
        epsilon_min=0.05,
        buffer_capacity=20000,
        target_update_freq=50,
        hidden_dim=128,
        dueling=True,
    )

    batch_size      = 64
    min_buffer_size = 500

    os.makedirs("results",     exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    episode_rewards: list[float] = []
    episode_losses:  list[float] = []
    best_reward = float("-inf")

    try:
        for ep in range(episodes):
            state        = normalize(env.reset())
            total_reward = 0.0
            total_loss   = 0.0
            n_updates    = 0

            for _ in range(steps_per_episode):
                action                        = agent.choose_action(state)
                next_state, reward, done      = env.step(action)
                next_state                    = normalize(next_state)

                agent.store_transition(state, action, reward, next_state, done)

                if len(agent.replay_buffer) >= min_buffer_size:
                    loss       = agent.train_step_batch(batch_size)
                    total_loss += loss
                    n_updates  += 1

                state        = next_state
                total_reward += reward
                if done:
                    break

            agent.decay_epsilon()

            avg_loss = total_loss / n_updates if n_updates > 0 else 0.0
            episode_rewards.append(total_reward)
            episode_losses.append(avg_loss)

            if total_reward > best_reward:
                best_reward = total_reward
                agent.save(f"checkpoints/dqn_{sim}_best.pt")

            print(
                f"[DQN/{sim}] Ep {ep + 1:>3}/{episodes} | "
                f"Reward {total_reward:>8.2f} | "
                f"Loss {avg_loss:>8.4f} | "
                f"ε {agent.epsilon:.3f}"
            )

    except KeyboardInterrupt:
        print(f"\n\nInterrupted at episode {len(episode_rewards)}. Saving results so far...")

    finally:
        if hasattr(env, "close"):
            env.close()

    prefix = f"dqn_{sim}"
    _save_results(prefix, episode_rewards, episode_losses)
    print(f"\nDone. Best reward: {best_reward:.2f}  →  checkpoints/dqn_{sim}_best.pt")


def _save_results(prefix, rewards, losses):
    with open(f"results/{prefix}_rewards.txt", "w") as f:
        f.writelines(f"{r}\n" for r in rewards)
    with open(f"results/{prefix}_losses.txt", "w") as f:
        f.writelines(f"{l}\n" for l in losses)
    print(f"Results saved to results/{prefix}_rewards.txt / _losses.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim",      choices=["cityflow", "sumo"], default="cityflow")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--steps",    type=int, default=200)
    args = parser.parse_args()
    train(args.sim, args.episodes, args.steps)
