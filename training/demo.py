"""
Visual demo — watch a trained agent control the intersection in real-time.

For SUMO:     opens sumo-gui, shows cars moving and lights switching live.
For CityFlow: runs headless and saves a replay file for browser visualization.

Usage:
    python training/demo.py                          # DQN on SUMO (default)
    python training/demo.py --method ppo --sim sumo
    python training/demo.py --method fixed --sim sumo
    python training/demo.py --method dqn --sim cityflow
    python training/demo.py --method dqn --sim sumo --delay 300   # slower
    python training/demo.py --method dqn --sim sumo --gui-settings data/sumo/gui.settings.xml
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

STATE_NORM = np.array([50, 50, 50, 50, 50, 50, 50, 50, 200, 200, 1], dtype=np.float32)


def normalize(state):
    return np.array(state, dtype=np.float32) / (STATE_NORM + 1e-8)


def load_agent(method: str, sim: str):
    if method == "fixed":
        return None

    ckpt = f"checkpoints/{method}_{sim}_best.pt"
    if not os.path.exists(ckpt):
        print(f"Checkpoint not found: {ckpt}")
        print(f"Run training first:  python training/train_{method}.py --sim {sim}")
        sys.exit(1)

    if method == "dqn":
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent(state_dim=11, action_dim=2, hidden_dim=128, dueling=True)
        agent.load(ckpt)
        agent.epsilon = 0.0   # pure exploitation during demo
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["dqn", "ppo", "fixed"], default="dqn")
    parser.add_argument("--sim",    choices=["cityflow", "sumo"],     default="sumo")
    parser.add_argument("--steps",  type=int, default=200)
    parser.add_argument("--delay",  type=int, default=200,
                        help="ms between SUMO-GUI frames (higher = slower/more visible)")
    parser.add_argument("--gui-settings", type=str, default="data/sumo/gui.settings.xml",
                        help="SUMO GUI settings XML to restore fixed camera/zoom")
    args = parser.parse_args()

    print("═" * 54)
    print(f"  DEMO  |  {args.method.upper()}  on  {args.sim}")
    print("═" * 54)

    agent = load_agent(args.method, args.sim)

    # ── Create environment ─────────────────────────────────────────────────
    if args.sim == "sumo":
        from env.sumo_traffic_env import SumoTrafficEnv
        gui_settings = args.gui_settings if os.path.exists(args.gui_settings) else None
        env = SumoTrafficEnv(use_gui=True, delay_ms=args.delay, gui_settings_file=gui_settings)
        print("Opening SUMO-GUI …  watch the intersection live.")
        if gui_settings:
            print(f"Using GUI settings: {gui_settings}")
        else:
            print("No GUI settings file found; using default camera view.")
        print("The agent controls the traffic lights automatically.")
        print("Press Ctrl+C to stop.\n")
    else:
        import json, shutil
        from env.traffic_env import TrafficEnv
        # Write a replay-enabled config for CityFlow
        with open("config.json") as f:
            cfg = json.load(f)
        cfg["saveReplay"]      = True
        cfg["roadnetLogFile"]  = "roadnetLogFile.json"
        cfg["replayLogFile"]   = "replayLogFile.txt"
        with open("config_replay.json", "w") as f:
            json.dump(cfg, f, indent=2)
        env = TrafficEnv(config_file="config_replay.json")
        print("CityFlow running in headless mode.")
        print("Replay saved to: replayLogFile.txt")
        print("Visualise at:   https://cityflow-project.github.io/cityflow/\n")

    # ── Run episode ────────────────────────────────────────────────────────
    try:
        state        = normalize(env.reset())
        total_reward = 0.0

        print(f"{'Step':>4}  {'Phase':<10}  {'Reward':>7}  {'Waiting':>7}")
        print("─" * 36)

        for step in range(args.steps):
            action                   = get_action(agent, args.method, state, step)
            next_state, reward, done = env.step(action)
            state                    = normalize(next_state)
            total_reward            += reward

            phase_label = "NS green" if action == 0 else "EW green"
            waiting     = int(next_state[9])
            print(f"{step + 1:>4}  {phase_label:<10}  {reward:>7.2f}  {waiting:>7} veh")

            if done:
                break

        print("─" * 36)
        print(f"Total reward: {total_reward:.2f}")

    except KeyboardInterrupt:
        print("\nDemo stopped.")
    finally:
        if hasattr(env, "close"):
            env.close()
        if args.sim == "cityflow" and os.path.exists("config_replay.json"):
            os.remove("config_replay.json")


if __name__ == "__main__":
    main()
