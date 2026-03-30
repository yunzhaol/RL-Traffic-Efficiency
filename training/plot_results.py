import matplotlib.pyplot as plt


with open("results/rewards.txt", "r") as f:
    rewards = [float(line.strip()) for line in f]

with open("results/losses.txt", "r") as f:
    losses = [float(line.strip()) for line in f]

plt.figure(figsize=(8, 5))
plt.plot(rewards)
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.tight_layout()
plt.savefig("results/reward_plot.png")

plt.figure(figsize=(8, 5))
plt.plot(losses)
plt.title("Episode Loss")
plt.xlabel("Episode")
plt.ylabel("Average Loss")
plt.tight_layout()
plt.savefig("results/loss_plot.png")

print("Plots saved in results/")