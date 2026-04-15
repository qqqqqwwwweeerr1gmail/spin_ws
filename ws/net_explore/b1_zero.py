import torch
import torch.nn as nn
import torch.nn.init as init  # For parameter initialization


# Define the neural network class
class ZeroNet(nn.Module):
    def __init__(self, hidden_size=10):  # Hidden size can be adjusted; 10 is arbitrary
        super(ZeroNet, self).__init__()
        # Input layer: 1 input -> hidden_size neurons (linear)
        self.input_layer = nn.Linear(1, hidden_size)
        # Middle (hidden) layer: hidden_size -> hidden_size (linear, one middle layer)
        self.middle_layer = nn.Linear(hidden_size, hidden_size)
        # Output layer: hidden_size -> 1 output (linear)
        self.output_layer = nn.Linear(hidden_size, 1)

        # Initialize all parameters (weights and biases) to 0
        self._init_zeros()

    def _init_zeros(self):
        # Set all weights and biases to 0 for each layer
        init.zeros_(self.input_layer.weight)
        init.zeros_(self.input_layer.bias)
        init.zeros_(self.middle_layer.weight)
        init.zeros_(self.middle_layer.bias)
        init.zeros_(self.output_layer.weight)
        init.zeros_(self.output_layer.bias)

    def forward(self, x):
        # Forward pass: purely linear (no activations)
        x = self.input_layer(x)
        x = self.middle_layer(x)
        x = self.output_layer(x)
        return x


# Test the network
if __name__ == "__main__":
    # Create the network
    net = ZeroNet(hidden_size=1)

    # Test with different inputs (should always output 0)
    inputs = [torch.tensor([[5.0]]),  # Positive input
              torch.tensor([[-3.2]]),  # Negative input
              torch.tensor([[0.0]]),  # Zero input
              torch.tensor([[100.0]])]  # Large input

    print("Testing ZeroNet (output should always be 0):")
    for inp in inputs:
        output = net(inp)
        print(f"Input: {inp.item():.2f} -> Output: {output.item():.2f}")

    for name, param in net.named_parameters():
        print(name, param.shape)
        print(param)  # prints tensor values





















