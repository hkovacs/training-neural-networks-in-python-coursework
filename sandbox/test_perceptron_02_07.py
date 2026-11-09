#!/usr/bin/env python3

# run from tld; python3 -m sandbox.test_perceptron_02_07

from src.multilayer_perceptron import Perceptron

neuron = Perceptron(inputs=2)
neuron.set_weights([10, 10, -15])  # AND

print("Gate:")
print("0 0 = {0:.10f}".format(neuron.run([0, 0])))
print("0 1 = {0:.10f}".format(neuron.run([0, 1])))
print("1 0 = {0:.10f}".format(neuron.run([1, 0])))
print("1 1 = {0:.10f}".format(neuron.run([1, 1])))
