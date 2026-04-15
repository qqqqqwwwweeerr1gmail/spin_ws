import json
import matplotlib.pyplot as plt

def plot_losses_and_rewards(log_path="train_log.json"):
    with open(log_path) as f:
        logs = json.load(f)

    epochs = logs.get("epochs", [])
    policy_loss = logs.get("policy_loss", [])
    value_loss = logs.get("value_loss", [])
    total_loss = logs.get("total_loss", [])
    episodes = logs.get("episodes", [])

    # Extract episode rewards and their 'indices'
    episode_rewards = [ep["episode_reward"] for ep in episodes]
    episode_epochs = [ep["epoch"] for ep in episodes]
    episode_numbers = list(range(len(episode_rewards)))

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot losses on left y-axis
    ax1.plot(epochs, policy_loss, label="Policy Loss")
    ax1.plot(epochs, value_loss, label="Value Loss")
    ax1.plot(epochs, total_loss, label="Total Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # Create second y-axis for rewards
    ax2 = ax1.twinx()
    ax2.scatter(episode_numbers, episode_rewards, c='orange', s=10, label="Episode Reward", alpha=0.6)
    ax2.set_ylabel("Episode Reward")
    ax2.legend(loc="upper right")

    plt.title("PPO Training Losses and Episode Rewards")
    plt.show()

if __name__ == "__main__":
    plot_losses_and_rewards()





















