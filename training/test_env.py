from env.traffic_env import TrafficEnv

env = TrafficEnv()

state = env.reset()
print("Initial state loaded")

for step in range(10):
    action = step % 2
    next_state, reward, done = env.step(action)
    print(f"Step {step + 1}, action={action}, reward={reward}")