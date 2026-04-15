import json
import matplotlib.pyplot as plt

def plot_losses(log_path="train_log.json"):
    with open(log_path) as f:
        logs = json.load(f)

    epochs = logs["epochs"]
    policy_loss = logs["policy_loss"]
    value_loss = logs["value_loss"]
    total_loss = logs["total_loss"]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, policy_loss, label="Policy Loss")
    plt.plot(epochs, value_loss, label="Value Loss")
    plt.plot(epochs, total_loss, label="Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PPO Training Losses")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_losses()
























