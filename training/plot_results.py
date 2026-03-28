import matplotlib.pyplot as plt

with open("results/rewards.txt", "r") as f:
    rewards = [float(line.strip()) for line in f]

with open("results/losses.txt", "r") as f:
    losses = [float(line.strip()) for line in f]

plt.figure()
plt.plot(rewards)
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.tight_layout() # coz the plot before has y-axis cutting off (y is too stretched out)
plt.savefig("results/reward_plot.png")

plt.figure()
plt.plot(losses)
plt.title("Episode Loss")
plt.xlabel("Episode")
plt.ylabel("Average Loss")
plt.savefig("results/loss_plot.png")

print("Plots saved in results/")

