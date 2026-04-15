# zero_net_buildlist.py
# Build a purely-linear network from a build list, e.g. [1,3,2] => Linear(1->3) + Linear(3->2)
# All weights and biases are initialized to zero, so output is always zero regardless of input.

import torch
import torch.nn as nn

class ZeroNet(nn.Module):
    def __init__(self, build_ls):
        """
        build_ls: list of layer widths. len(build_ls) >= 2
          - Example: [in_dim, hidden1, hidden2, ..., out_dim]
          - [1, 3, 2] means input=1, one hidden layer of size 3, output=2
        """
        super().__init__()
        self._validate(build_ls)
        self.build_ls = list(build_ls)

        # Create a sequence of Linear layers connecting consecutive sizes
        layers = []
        for in_dim, out_dim in zip(self.build_ls[:-1], self.build_ls[1:]):
            layers.append(nn.Linear(in_dim, out_dim, bias=True))
        self.layers = nn.ModuleList(layers)

        # Initialize all parameters to zero
        self._init_zeros()

    def _validate(self, build_ls):
        if not isinstance(build_ls, (list, tuple)) or len(build_ls) < 2:
            raise ValueError("build_ls must be a list/tuple with at least [in_dim, out_dim].")
        if any((not isinstance(n, int)) or n <= 0 for n in build_ls):
            raise ValueError("All dimensions in build_ls must be positive integers.")

    def _init_zeros(self):
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()

    def forward(self, x):
        # Purely linear chain, no activations
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    # Example 1: [1, 3, 2] (input 1 -> hidden 3 -> output 2)
    model = ZeroNet([1, 3, 2])

    # Test: outputs are zero regardless of input values
    xs = torch.tensor([[ -5.0],
                       [  0.0],
                       [  7.2]], dtype=torch.float32)  # shape [batch=3, in_dim=1]
    ys = model(xs)
    print("Outputs:\n", ys)  # should be all zeros with shape [3, 2]

    # Inspect parameters (names, shapes, values)
    for name, p in model.named_parameters():
        print(name, tuple(p.shape))
        print(p)

    # Quick check all params are zero
    all_zero = all((p.detach() == 0).all().item() for p in model.parameters())
    print("All params zero:", all_zero)
