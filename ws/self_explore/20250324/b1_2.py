import gym
import numpy as np
import torch
import torch.nn as nn
import random

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Define the custom activation function as a PyTorch module
class BinaryNotActivation(nn.Module):
    def __init__(self):
        super(BinaryNotActivation, self).__init__()

    def forward(self, x):
        """
        PyTorch implementation of the binary NOT activation function.
        Maps 1 to 0 and 0 to 1. Input must be 0 or 1.

        Args:
            x: PyTorch tensor (expected to contain only 0s and 1s)

        Returns:
            Tensor where 1s are mapped to 0s and 0s are mapped to 1s
        """
        # Check that all values are 0 or 1
        if not torch.all((x == 0) | (x == 1)):
            raise ValueError("All input values must be 0 or 1")

        # Apply the NOT operation: 1 - x
        return 1 - x


# Define a simple neural network model for Acrobot
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = nn.Linear(1, 1, bias=True)  # Equivalent to Dense(1, input_shape=(1,))
        self.binarize = lambda x: torch.where(x > 0, torch.tensor(1.0), torch.tensor(0.0))
        self.binary_not = BinaryNotActivation()

    def forward(self, x):
        x = self.linear(x)  # Linear transformation
        x = self.binarize(x)  # Binarize the output to 0 or 1
        x = self.binary_not(x)  # Apply the custom activation
        return x


# Create the model
model = SimpleNet()

# Initialize weights and biases to specific values for clarity
with torch.no_grad():
    model.linear.weight.fill_(1.0)  # Set weight to 1.0
    model.linear.bias.fill_(0.0)  # Set bias to 0.0

# Prepare the input
observation = np.array([1], dtype=np.float32).reshape(1, 1)  # Shape (1, 1)
observation_tensor = torch.tensor(observation)  # Convert to PyTorch tensor

# Set the model to evaluation mode (not training)
model.eval()

# Predict
with torch.no_grad():  # Disable gradient computation for inference
    action_probs = model(observation_tensor)

print("Output:", action_probs.numpy())  # Convert back to NumPy for printing

# Print the layers (PyTorch doesn't have a direct equivalent to model.layers, so we print the modules)
print("\nModel Modules:")
for name, module in model.named_children():
    print(f"{name}: {module}")

# View all model parameters
print("\nModel Parameters:")
for name, param in model.named_parameters():
    print(f"Parameter: {name}, Value: {param.data}")