#!/usr/bin/env python3

# run from tld; python3 -m sandbox.test_mlp_backpropagation

from src.multilayer_perceptron import MultiLayerPerceptron

epochs = int(input("How many epochs? "))
mlp1 = MultiLayerPerceptron(layers=[7, 7, 1])
mlp2 = MultiLayerPerceptron(layers=[7, 7, 10])
mlp3 = MultiLayerPerceptron(layers=[7, 7, 7])

# Dataset for the 7 to 1 network
print("Training 7 to 1 network...")
for i in range(epochs):
    mse = 0.0
    mse += mlp1.bp([1, 1, 1, 1, 1, 1, 0], [0.05])  # 0 pattern
    mse += mlp1.bp([0, 1, 1, 0, 0, 0, 0], [0.15])  # 1 pattern
    mse += mlp1.bp([1, 1, 0, 1, 1, 0, 1], [0.25])  # 2 pattern
    mse += mlp1.bp([1, 1, 1, 1, 0, 0, 1], [0.35])  # 3 pattern
    mse += mlp1.bp([0, 1, 1, 0, 0, 1, 1], [0.45])  # 4 pattern
    mse += mlp1.bp([1, 0, 1, 1, 0, 1, 1], [0.55])  # 5 pattern
    mse += mlp1.bp([1, 0, 1, 1, 1, 1, 1], [0.65])  # 6 pattern
    mse += mlp1.bp([1, 1, 1, 0, 0, 0, 0], [0.75])  # 7 pattern
    mse += mlp1.bp([1, 1, 1, 1, 1, 1, 1], [0.85])  # 8 pattern
    mse += mlp1.bp([1, 1, 1, 1, 0, 1, 1], [0.95])  # 9 pattern
    mse = mse / 10.0

# Dataset for the 7 to 10 network
print("Training 7 to 10 network...")
for i in range(epochs):
    mse = 0.0
    mse += mlp2.bp([1, 1, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 0 pattern
    mse += mlp2.bp([0, 1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # 1 pattern
    mse += mlp2.bp([1, 1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0])  # 2 pattern
    mse += mlp2.bp([1, 1, 1, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])  # 3 pattern
    mse += mlp2.bp([0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])  # 4 pattern
    mse += mlp2.bp([1, 0, 1, 1, 0, 1, 1], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])  # 5 pattern
    mse += mlp2.bp([1, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0])  # 6 pattern
    mse += mlp2.bp([1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0])  # 7 pattern
    mse += mlp2.bp([1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0])  # 8 pattern
    mse += mlp2.bp([1, 1, 1, 1, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 9 pattern
    mse = mse / 10.0

# Dataset for the 7 to 7 network
print("Training 7 to 7 network...")
for i in range(epochs):
    mse = 0.0
    mse += mlp3.bp([1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 0])  # 0 pattern
    mse += mlp3.bp([0, 1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0])  # 1 pattern
    mse += mlp3.bp([1, 1, 0, 1, 1, 0, 1], [1, 1, 0, 1, 1, 0, 1])  # 2 pattern
    mse += mlp3.bp([1, 1, 1, 1, 0, 0, 1], [1, 1, 1, 1, 0, 0, 1])  # 3 pattern
    mse += mlp3.bp([0, 1, 1, 0, 0, 1, 1], [0, 1, 1, 0, 0, 1, 1])  # 4 pattern
    mse += mlp3.bp([1, 0, 1, 1, 0, 1, 1], [1, 0, 1, 1, 0, 1, 1])  # 5 pattern
    mse += mlp3.bp([1, 0, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1, 1])  # 6 pattern
    mse += mlp3.bp([1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0])  # 7 pattern
    mse += mlp3.bp([1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1])  # 8 pattern
    mse += mlp3.bp([1, 1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 0, 1, 1])  # 9 pattern
    mse = mse / 10.0

print("Done!\n")
pattern = [1.2]
while pattern[0] >= 0.0:
    pattern = list(map(float, input("Input pattern 'a b c d e f g': ").strip().split()))
    if pattern[0] < 0.0:
        break
    print()
    print("The number recognized by the 7 to 1 network is", int(mlp1.run(pattern) * 10))

    print("The number recognized by the 7 to 10 network is", np.argmax(mlp2.run(pattern)))

    print("The pattern recognized by the 7 to 7 network is", [int(x) for x in (mlp3.run(pattern) + 0.5)], "\n")
