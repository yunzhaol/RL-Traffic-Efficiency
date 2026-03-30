from env.traffic_env import TrafficEnv


def main():
    env = TrafficEnv()

    episodes = 20
    steps_per_episode = 80

    rewards = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(steps_per_episode):
            # simple alternating signal
            action = step % 2
            next_state, reward, done = env.step(action)

            total_reward += reward
            state = next_state

        rewards.append(total_reward)
        print(f"Episode {ep+1}: {total_reward}")

    print("\nAverage reward:", sum(rewards)/len(rewards))


if __name__ == "__main__":
    main()