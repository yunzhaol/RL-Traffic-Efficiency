from env.traffic_env import TrafficEnv


def main():
    env = TrafficEnv()
    state = env.reset()

    print("Initial state:", state)

    for step in range(10):
        action = step % 2
        next_state, reward, done = env.step(action)
        print(f"Step {step + 1} | action={action} | reward={reward} | next_state={next_state}")


if __name__ == "__main__":
    main()