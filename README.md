# Training Neural Networks in Python

A from-scratch implementation of single-layer Perceptrons and Multi-Layer Perceptrons (MLPs) in Python using NumPy. This repository explores artificial neural network fundamentals—including forward propagation, manual weight initialization, gradient-based backpropagation, and interactive Seven-Segment Display Recognition Neural Networks (SDRNNs) built with Tkinter.

---

## Project Structure

```text
training-neural-networks-in-python/
├── src/
│   └── multilayer_perceptron.py                 # Core Perceptron & MultiLayerPerceptron classes
└── sandbox/
    ├── SDRNN_7to1.py                            # Interactive GUI: 7-7-1 continuous output MLP
    ├── SDRNN_7to7.py                            # Interactive GUI: 7-7-7 pattern-reconstruction MLP
    ├── SDRNN_7to10.py                           # Interactive GUI: 7-7-10 one-hot classification MLP
    ├── test_mlp_backpropagation.py              # Trains a 2-2-1 MLP to learn the XOR logic gate
    ├── test_mlp_epochs.py                       # CLI benchmark comparing 7-7-1, 7-7-10, and 7-7-7 MLPs
    ├── test_multilayer_perceptron_set_weights.py # Hardcoded 2-2-1 MLP weights for an XOR gate
    ├── test_perceptron_and.py                   # Single Perceptron configured as an AND gate
    └── test_perceptron_or.py                    # Single Perceptron configured as an OR gate
```

---

## Core Architecture (`src/multilayer_perceptron.py`)

The core library implements two classes without relying on high-level deep learning frameworks:

* **`Perceptron`**: Models a single artificial neuron with a configurable bias term (`default=1.0`), random weight initialization in \([-1, 1]\), and a **sigmoid activation function**:
  \[\sigma(x) = \frac{1}{1 + e^{-x}}\]
* **`MultiLayerPerceptron`**: Constructs a fully connected feedforward network from an arbitrary layer specification (e.g., `layers=[7, 7, 10]`). Supports:
  * **Forward Propagation** (`run(x)`): Propagates input vectors sequentially through hidden and output layers.
  * **Backpropagation** (`backpropagation(x, y)`): Trains the network on single `(x, y)` pairs using **Mean Squared Error (MSE)** loss and stochastic gradient descent with a configurable learning rate (`eta=0.5`).
  * **Manual Weight Inspection & Configuration** (`set_weights(w_init)`, `print_weights()`): Allows analytical weight assignment and layer-by-layer debugging.

---

## Prerequisites & Installation

### Requirements
* **Python 3.7+**
* **NumPy**
* **Tkinter** (included with most standard Python installations; required for the interactive `SDRNN_*` GUIs)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com<your-username>/training-neural-networks-in-python.git
   cd training-neural-networks-in-python
   ```
2. Install NumPy:
   ```bash
   pip install numpy
   ```
   *(On Debian/Ubuntu Linux, if Tkinter is not installed, run `sudo apt-get install python3-tk`.)*

---

## Usage & Experiments

All sandbox scripts should be executed as Python modules from the **top-level directory (TLD)** so imports from `src` resolve cleanly.

### 1. Seven-Segment Display Recognition GUIs (SDRNN)
Each GUI provides seven interactive sliders (`a` through `g`) arranged in a classic seven-segment display layout, real-time network inference, and live epoch-by-epoch training controls.

| Script | Network Topology | Output Encoding | Observed Performance | Run Command |
| :--- | :--- | :--- | :--- | :--- |
| `SDRNN_7to7.py` | `[7, 7, 7]` | Autoencoder-style segment reconstruction (`0` or `1` per segment) | **Best performer** | `python3 -m sandbox.SDRNN_7to7` |
| `SDRNN_7to10.py` | `[7, 7, 10]` | One-hot vector across digits `0–9` (`np.argmax`) | **Moderate performer** | `python3 -m sandbox.SDRNN_7to10` |
| `SDRNN_7to1.py` | `[7, 7, 1]` | Single scalar value in `[0.05, 0.95]` mapped to digits `0–9` | **Worst performer** | `python3 -m sandbox.SDRNN_7to1` |

### 2. Logic Gates & Backpropagation Scripts
* **Train an XOR Gate via Backpropagation** (3,000 epochs on a `[2, 2, 1]` MLP):
  ```bash
  python3 -m sandbox.test_mlp_backpropagation
  ```
* **Analytical XOR Gate via Manual Weights** (combines `NAND`, `OR`, and `AND` neurons):
  ```bash
  python3 -m sandbox.test_multilayer_perceptron_set_weights
  ```
* **Single-Perceptron Logic Gates (`AND` / `OR`)**:
  ```bash
  python3 -m sandbox.test_perceptron_and
  python3 -m sandbox.test_perceptron_or
  ```
* **Interactive CLI Comparison Across SDRNN Topologies**:
  ```bash
  python3 -m sandbox.test_mlp_epochs
  ```

---

## Quick Library Example

```python
from src.multilayer_perceptron import MultiLayerPerceptron

# Initialize a 2-input, 2-hidden, 1-output network
mlp = MultiLayerPerceptron(layers=[2, 2, 1], bias=1.0, eta=0.5)

# Train on the XOR dataset
training_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0]),
]

for epoch in range(3000):
    for inputs, target in training_data:
        mlp.backpropagation(inputs, target)

# Run inference
print("XOR(1, 0):", round(mlp.run([1, 0])[0], 4))
```

---

## License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.