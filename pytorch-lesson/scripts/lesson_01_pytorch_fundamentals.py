"""
LESSON 1 — PyTorch Fundamentals, taught with tiny runnable examples.

Run this with: python3 scripts/lesson_01_pytorch_fundamentals.py

Goal: understand FOUR ideas. Everything else in PyTorch is built on these.
  1. Tensors        -- PyTorch's array type (like a numpy array, but smarter)
  2. Autograd        -- PyTorch automatically computes derivatives for you
  3. A "layer"        -- a weighted sum + a nonlinearity, the atomic unit of a network
  4. The training loop -- predict, measure error, compute gradient, nudge weights, repeat

We build up from "one weight" to "a real small network" so each idea lands
before the next one is added.
"""

import torch

print("=" * 70)
print("PART 1: Tensors -- just arrays that can track their own gradients")
print("=" * 70)

# A tensor is created almost exactly like a numpy array.
x = torch.tensor([1.0, 2.0, 3.0])
print("x =", x)
print("x + 10 =", x + 10)
print("x.mean() =", x.mean())

# The one new thing: requires_grad=True tells PyTorch "remember every
# operation done to this tensor, so you can later compute how the final
# result changes if THIS number changes."
w = torch.tensor(2.0, requires_grad=True)
print("\nw =", w, " (a single trainable number, our 'weight')")


print("\n" + "=" * 70)
print("PART 2: Autograd -- automatic differentiation, in the simplest case")
print("=" * 70)

# Suppose our "model" is just: prediction = w * x_input
# and we have ONE training example: x_input = 3, true_answer = 10
x_input = torch.tensor(3.0)
true_answer = torch.tensor(10.0)

prediction = w * x_input
error = (prediction - true_answer) ** 2   # squared error -- how wrong are we?

print(f"prediction = w * x_input = {w.item()} * {x_input.item()} = {prediction.item()}")
print(f"error = (prediction - true_answer)^2 = {error.item()}")

# This one line is the magic: it works out d(error)/d(w) -- i.e. "if I
# nudge w up slightly, does the error go up or down, and by how much?"
error.backward()
print(f"\nPyTorch computed the gradient for us: d(error)/d(w) = {w.grad.item()}")
print("This is the SAME derivative you'd get doing calculus by hand.")
print("Negative gradient means: increasing w will DECREASE the error.")


print("\n" + "=" * 70)
print("PART 3: Gradient descent -- using that gradient to improve w")
print("=" * 70)

# Reset and do this properly over several steps, printing progress.
w = torch.tensor(0.0, requires_grad=True)
learning_rate = 0.01

for step in range(10):
    prediction = w * x_input
    error = (prediction - true_answer) ** 2

    error.backward()              # compute the gradient
    with torch.no_grad():         # temporarily stop tracking (we're just updating, not predicting)
        w -= learning_rate * w.grad   # nudge w a little in the direction that reduces error
        w.grad.zero_()             # clear the gradient before the next step (PyTorch accumulates otherwise)

    print(f"step {step}: w = {w.item():.4f}, prediction = {prediction.item():.4f}, error = {error.item():.4f}")

print(f"\nAfter 10 steps, w = {w.item():.4f} (true answer requires w = {true_answer.item()/x_input.item():.4f})")
print("This loop -- predict, measure error, compute gradient, nudge weights -- IS")
print("training a neural network. A real network just has thousands of w's instead of one,")
print("and PyTorch's nn.Module + optimizer classes do this bookkeeping for you.")


print("\n" + "=" * 70)
print("PART 4: A real 'layer' -- weighted sum + nonlinearity")
print("=" * 70)

# nn.Linear(in_features, out_features) is a full "layer": it holds a
# weight for every input, a bias, and computes: output = (inputs @ weights) + bias
# for as many inputs/outputs as you specify.
import torch.nn as nn

layer = nn.Linear(in_features=3, out_features=1)
example_input = torch.tensor([[1.0, 2.0, 3.0]])   # one example, 3 features
output = layer(example_input)
print("A layer with 3 inputs and 1 output, given input [1, 2, 3]:")
print("output =", output)
print("\nThis is exactly the 'w * x' idea from Part 2, just with 3 weights")
print("(one per input feature) and a bias, added up, instead of 1 weight.")

# The nonlinearity: without it, stacking layers would collapse into one
# big linear equation, unable to learn curves. ReLU is the simplest common
# choice: it just zeroes out negative values, everything else passes through.
relu = nn.ReLU()
test_values = torch.tensor([-2.0, -0.5, 0.0, 1.5, 3.0])
print(f"\nReLU({test_values.tolist()}) = {relu(test_values).tolist()}")
print("Negative numbers become 0. Positive numbers are unchanged. That tiny")
print("'kink' is what lets a stack of layers approximate curved relationships.")

print("\n" + "=" * 70)
print("You now know every ingredient used in the real model in lesson_02.")
print("=" * 70)
