# =====================================================================
# SARDAR PATEL INSTITUTE OF TECHNOLOGY
# Experiment 4: Single Layer Perceptron - AND Gate
# Source: EXP4_AISC_2024300198 (2) (1).pdf
# =====================================================================

"""Implement an AND gate with a single-layer perceptron."""

import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """A binary perceptron trained with the perceptron learning rule."""

    def __init__(self, learning_rate=1.0, epochs=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = [0.0, 0.0]
        self.bias = 0.0
        self.history = []

    @staticmethod
    def activation(net_input):
        """Return 1 for a non-negative net input and 0 otherwise."""
        return int(net_input >= 0)

    def predict_one(self, inputs):
        net_input = sum(weight * value for weight, value in zip(self.weights, inputs))
        return self.activation(net_input + self.bias)

    def fit(self, inputs, targets):
        """Learn weights and bias from binary training examples."""
        self.history = []
        for epoch in range(1, self.epochs + 1):
            errors = 0
            for example, target in zip(inputs, targets):
                weight_1, weight_2 = self.weights
                net_input = (
                    weight_1 * example[0]
                    + weight_2 * example[1]
                    + self.bias
                )
                prediction = self.activation(net_input)
                update = self.learning_rate * (target - prediction)
                delta_weight_1 = update * example[0]
                delta_weight_2 = update * example[1]
                delta_bias = update

                self.weights = [
                    weight + delta
                    for weight, value in zip(self.weights, example)
                    for delta in [update * value]
                ]
                self.bias += delta_bias
                self.history.append({
                    "epoch": epoch,
                    "x1": example[0],
                    "x2": example[1],
                    "target": target,
                    "net_input": net_input,
                    "prediction": prediction,
                    "weight_1": weight_1,
                    "weight_2": weight_2,
                    "bias": self.bias - delta_bias,
                    "delta_weight_1": delta_weight_1,
                    "delta_weight_2": delta_weight_2,
                    "delta_bias": delta_bias,
                    "new_weight_1": self.weights[0],
                    "new_weight_2": self.weights[1],
                    "new_bias": self.bias,
                })
                errors += int(update != 0)

            if errors == 0:
                break

    def predict(self, inputs):
        return [self.predict_one(example) for example in inputs]

def main():
    # AND is 1 only when both inputs are 1.
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    targets = [0, 0, 0, 1]

    perceptron = Perceptron(learning_rate=1.0, epochs=20)
    perceptron.fit(inputs, targets)
    predictions = perceptron.predict(inputs)

    print("\nAND gate using a single-layer perceptron")
    print("Formula: yin = b + (w1 * x1) + (w2 * x2)")
    print("Update: delta = alpha * (target - prediction)")
    print()
    header = (
        f"{'epoch':>5} {'x1':>3} {'x2':>3} {'target':>6} {'yin':>5} "
        f"{'prediction':>10} {'w1':>5} {'w2':>5} {'b':>5} | "
        f"{'dw1':>5} {'dw2':>5} {'db':>4} | {'new_w1':>8} {'new_w2':>8} {'new_b':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in perceptron.history:
        print(
            f"{row['epoch']:>5} {row['x1']:>3} {row['x2']:>3} "
            f"{row['target']:>6} {row['net_input']:>5.0f} "
            f"{row['prediction']:>10} {row['weight_1']:>5.0f} "
            f"{row['weight_2']:>5.0f} {row['bias']:>5.0f} | "
            f"{row['delta_weight_1']:>5.0f} {row['delta_weight_2']:>5.0f} "
            f"{row['delta_bias']:>4.0f} | {row['new_weight_1']:>8.0f} "
            f"{row['new_weight_2']:>8.0f} {row['new_bias']:>6.0f}"
        )

    print()
    print(f"Final weights: {perceptron.weights}")
    print(f"Final bias: {perceptron.bias}")
    print(f"Final predictions: {predictions}\n")

    assert predictions == targets

    # Graph Generation & Printing
    print("Generating Perceptron Decision Boundary Plot for AND Gate...")
    plt.figure(figsize=(7, 6))

    # Grid for decision region
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    w1, w2 = perceptron.weights
    b = perceptron.bias
    Z = (w1 * xx + w2 * yy + b >= 0).astype(int)

    plt.contourf(xx, yy, Z, alpha=0.25, levels=[-0.5, 0.5, 1.5], colors=['#ff9999', '#99ff99'])

    # Decision line: w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b) / w2
    x_line = np.linspace(-0.5, 1.5, 100)
    if abs(w2) > 1e-5:
        y_line = -(w1 * x_line + b) / w2
        plt.plot(x_line, y_line, 'k--', linewidth=2, label=f'Decision Boundary: {w1:.1f}x1 + {w2:.1f}x2 + {b:.1f} = 0')

    # Scatter data points
    for (x1, x2), target in zip(inputs, targets):
        if target == 1:
            plt.scatter(x1, x2, color='green', s=150, edgecolors='black', linewidth=2, label='Class 1 (True)' if (x1, x2) == (1, 1) else "")
            plt.text(x1 + 0.05, x2, f'({x1},{x2}) -> 1', fontsize=11, fontweight='bold', color='darkgreen')
        else:
            plt.scatter(x1, x2, color='red', s=150, edgecolors='black', linewidth=2, label='Class 0 (False)' if (x1, x2) == (0, 0) else "")
            plt.text(x1 + 0.05, x2, f'({x1},{x2}) -> 0', fontsize=11, fontweight='bold', color='darkred')

    plt.title('Perceptron Decision Boundary - AND Gate', fontsize=14, fontweight='bold')
    plt.xlabel('Input $x_1$', fontsize=12)
    plt.ylabel('Input $x_2$', fontsize=12)
    plt.xlim(-0.3, 1.4)
    plt.ylim(-0.3, 1.4)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left')

    plot_file = 'perceptron_and_gate_plot.png'
    try:
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"[Graph Printed] Plot successfully saved to '{plot_file}'")
    except Exception:
        pass

    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
