# =====================================================================
# SARDAR PATEL INSTITUTE OF TECHNOLOGY
# Experiment 4: Single Layer Perceptron - AND Gate
# Source: EXP4_AISC_2024300198 (2) (1).pdf
# =====================================================================

"""Implement an AND gate with a single-layer perceptron."""

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

if __name__ == "__main__":
    main()
