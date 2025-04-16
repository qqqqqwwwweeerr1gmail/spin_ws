import os
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


# Define a simple neural network model with two inputs
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Input layer now accepts 2 inputs
        self.linear = nn.Linear(2, 1, bias=True)  # 2 inputs, 1 output

    def forward(self, x):
        # Extract the two inputs
        input1 = x[:, 0]  # First input
        input2 = x[:, 1]  # Second input

        # Apply the logic: if two inputs are the same, return 0; else return 1
        output = torch.where(input1 == input2, torch.tensor(0.0), torch.tensor(1.0))
        return output.unsqueeze(1)  # Reshape to match expected output shape


# Create the model
model = SimpleNet()

# Initialize weights and biases to specific values for clarity
with torch.no_grad():
    model.linear.weight.fill_(1.0)  # Set weight to 1.0
    model.linear.bias.fill_(0.0)  # Set bias to 0.0

# Prepare the input (two inputs at the same time)
observation = np.array([[1, 0]], dtype=np.float32)  # Shape (1, 2)
print("Observation:", observation)
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


# Save and reload the model
model_dir_path = fr'/home/ws/git/spin_ws/ws/self_explore/models'
model_name = 'model.pt'
model_path = os.path.join(model_dir_path, model_name)

# Step 1: Save the entire model (structure + parameters)
torch.save(model, model_path)
print(f"Entire model saved to {model_path}")

# Step 2: Reload the entire model from scratch
loaded_model = torch.load(model_path)
loaded_model.eval()  # Set to evaluation mode
print("Entire model reloaded")

# Perform inference with the loaded model
with torch.no_grad():
    loaded_output = loaded_model(observation_tensor)
print("Loaded Model Output:", loaded_output.numpy())

# Verify the outputs match
assert np.array_equal(action_probs.numpy(), loaded_output.numpy()), "Outputs do not match!"
print("Outputs match between original and loaded model")


# Inference with different inputs
for ob in [[1, 0], [0, 0], [1, 1], [0, 1]]:
    ob = np.array(ob, dtype=np.float32).reshape(1, -1)
    print(f"Observation: {ob}")
    result = model(torch.tensor(ob))
    print(f"Result: {result}")