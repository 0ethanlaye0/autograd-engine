# autograd-engine

A minimal scalar-valued autograd engine and neural network library built from scratch in Python, inspired by Andrej Karpathy's micrograd.

## What's implemented

- **`engine.py`** — The `Value` class: a scalar wrapper that builds a computation graph during the forward pass and supports automatic backpropagation via reverse-mode autodiff. Implements `+`, `*`, `**`, `/`, `-`, `exp`, `tanh`, and their exact local gradients. Topological sort ensures correct backward pass ordering.

- **`nn.py`** — Neural network library built on top of `engine.py`: `Neuron` (single neuron with weights, bias, and tanh activation), `Layer` (a collection of neurons), and `MLP` (multi-layer perceptron composed of stacked layers). All modules expose a `parameters()` method for gradient-based optimization.

- **`micrograd.ipynb`** — Step-by-step notebook showing the full build process: from manual backpropagation on a single expression, to automatic backprop with topological sort, to training an MLP on a nonlinear dataset.

## Quick demo

```python
from engine import Value
from nn import MLP

# Create a 2-input MLP with two hidden layers
model = MLP(2, [4, 4, 1])

# Forward pass
x = [Value(1.0), Value(0.0)]
out = model(x)

# Backward pass
out.backward()

# Train on XOR
xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
ys = [0.0, 1.0, 1.0, 0.0]

for step in range(200):
    preds = [model(x) for x in xs]
    loss = sum((y - p)**2 for y, p in zip(ys, preds))
    for p in model.parameters(): p.grad = 0
    loss.backward()
    for p in model.parameters(): p.data -= 0.05 * p.grad

print(f"Final loss: {loss.data:.4f}")  # should be near 0
```

## Key concepts demonstrated

- Computation graph construction during forward pass
- Reverse-mode automatic differentiation (backpropagation)
- Chain rule applied at each operation node for gradient calculation
- Gradient accumulation fix (`+=` instead of `=`)
- Topological sort for correct backward pass ordering
- Non-linearity (tanh activation) which enables learning on problems a linear model cannot solve

## Credit

Inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd) and his lecture "The spelled-out intro to neural networks and backpropagation."
