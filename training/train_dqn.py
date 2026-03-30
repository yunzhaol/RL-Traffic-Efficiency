from env.traffic_env import TrafficEnv
from agents.dqn_agent import DQNAgent


def main():
    env = TrafficEnv()

    state_dim = 11
    action_dim = 2

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=0.00005,
        epsilon=1.0,
        epsilon_decay=0.998,
        epsilon_min=0.10,
        target_update_freq=50,
    )

    episodes = 100
    steps_per_episode = 80
    batch_size = 64
    min_buffer_size = 200

    episode_rewards = []
    episode_losses = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        total_loss = 0.0
        update_count = 0

        for step in range(steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)

            if len(agent.replay_buffer) >= min_buffer_size:
                loss = agent.train_step_batch(batch_size=batch_size)
                total_loss += loss
                update_count += 1

            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        avg_loss = total_loss / update_count if update_count > 0 else 0.0

        episode_rewards.append(total_reward)
        episode_losses.append(avg_loss)

        print(
            f"Episode {episode + 1}/{episodes} | "
            f"Total Reward: {total_reward:.2f} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

    with open("results/rewards.txt", "w") as f:
        for r in episode_rewards:
            f.write(f"{r}\n")

    with open("results/losses.txt", "w") as f:
        for l in episode_losses:
            f.write(f"{l}\n")

    print("\nTraining finished.")
    print("Saved rewards to results/rewards.txt")
    print("Saved losses to results/losses.txt")


if __name__ == "__main__":
    main()