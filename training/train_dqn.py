from env.traffic_env import TrafficEnv
from agents.dqn_agent import DQNAgent


def main():
    env = TrafficEnv()

    state_dim = 11 # was 2 before but we updated states
    action_dim = 2

    agent = DQNAgent(state_dim, action_dim)

    episodes = 20
    steps_per_episode = 50

    episode_rewards = []
    episode_losses = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        total_loss = 0

        for step in range(steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            loss = agent.train_step(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            total_loss += loss

        agent.decay_epsilon()

        avg_loss = total_loss / steps_per_episode

        episode_rewards.append(total_reward)
        episode_losses.append(avg_loss)

        print(
            f"Episode {episode + 1}/{episodes} | "
            f"Total Reward: {total_reward:.2f} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

    print("\nEpisode rewards:")
    print(episode_rewards)

    print("\nEpisode losses:")
    print(episode_losses)

    with open("results/rewards.txt", "w") as f:
        for r in episode_rewards:
            f.write(f"{r}\n")

    with open("results/losses.txt", "w") as f:
        for l in episode_losses:
            f.write(f"{l}\n")


if __name__ == "__main__":
    main()