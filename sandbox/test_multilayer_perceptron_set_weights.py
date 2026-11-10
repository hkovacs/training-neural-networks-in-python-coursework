#!/usr/bin/env python3

# run from tld; python3 -m sandbox.test_multilayer_perceptron_set_weights

from src.multilayer_perceptron import MultiLayerPerceptron

mlp = MultiLayerPerceptron(layers=[2, 2, 1])  # mlp
mlp.set_weights([[[-10, -10, 15], [15, 15, -10]], [[10, 10, -15]]])
mlp.print_weights()
print("MLP:")
print("0 0 = {0:.10f}".format(mlp.run([0, 0])[0]))
print("0 1 = {0:.10f}".format(mlp.run([0, 1])[0]))
print("1 0 = {0:.10f}".format(mlp.run([1, 0])[0]))
print("1 1 = {0:.10f}".format(mlp.run([1, 1])[0]))
