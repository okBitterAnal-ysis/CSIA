# =====================================================================
# BHARATIYA VIDYA BHAVAN'S SARDAR PATEL INSTITUTE OF TECHNOLOGY
# Experiment 6: Learning Vector Quantization (LVQ)
# Source: AISC_exp6_C3_2024300198 (1).pdf
# =====================================================================

import numpy as np
import matplotlib.pyplot as plt

def lvq():
    print("\n\n" + "=" * 65)
    print("LEARNING VECTOR QUANTIZATION (LVQ)")
    print("=" * 65)

    # Five vectors from Table 1
    X = [
        [0.0000, 0.0000, 1.0000, 1.0000],
        [1.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 1.0000],
        [1.0000, 1.0000, 0.0000, 0.0000],
        [0.0000, 1.0000, 1.0000, 0.0000]
    ]

    # Classes from Table 1
    T = [1, 2, 2, 1, 1]

    # First two vectors are initial reference vectors
    W_initial = [X[0].copy(), X[1].copy()]
    W = [
        X[0].copy(),
        X[1].copy()
    ]

    # Classes corresponding to reference vectors
    W_class = [1, 2]

    # Learning rate
    alpha = 0.1000

    print("\nInitial Weight Vectors:")
    for row in W:
        print("[", end="  ")
        for value in row:
            print(f"{value:.4f}", end="  ")
        print("]")

    print("\nReference Classes:")
    print(W_class)

    print(f"\nLearning Rate = {alpha:.4f}")

    # Remaining three vectors are input vectors
    for i in range(2, 5):
        x = X[i]
        target = T[i]

        print("\n" + "-" * 65)
        print(f"Input Vector {i - 1}")
        print("-" * 65)

        print("Input = [", end="  ")
        for value in x:
            print(f"{value:.4f}", end="  ")
        print("]")

        print(f"Target Class = {target}")

        # Calculate squared Euclidean distance
        distances = []
        print("\nSquared Euclidean Distances:")
        for j in range(2):
            distance = 0.0
            for k in range(4):
                distance += (W[j][k] - x[k]) ** 2
            distances.append(distance)
            print(f"D({j + 1}) = {distance:.4f}")

        # Find winning unit
        winner = distances.index(min(distances))
        print(f"\nWinning Unit = Y{winner + 1}")
        print(f"Winner Class = {W_class[winner]}")

        # Update weights
        if W_class[winner] == target:
            print("\nClass matches target.")
            print("Updating weights TOWARDS the input.")
            for k in range(4):
                W[winner][k] = (
                    W[winner][k] +
                    alpha * (x[k] - W[winner][k])
                )
        else:
            print("\nClass does not match target.")
            print("Updating weights AWAY from the input.")
            for k in range(4):
                W[winner][k] = (
                    W[winner][k] -
                    alpha * (x[k] - W[winner][k])
                )

        # Display updated weights
        print("\nUpdated Weight Vectors:")
        for row in W:
            print("[", end="  ")
            for value in row:
                print(f"{value:.4f}", end="  ")
            print("]")

    # Graph Generation & Printing
    print("\nGenerating LVQ Weight Adaptation & Prototype Vectors Plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x_indices = np.arange(4)
    bar_width = 0.35

    # Prototype W1 (Class 1)
    ax1.bar(x_indices - bar_width/2, W_initial[0], bar_width, label='Initial W1', color='#3498db', edgecolor='black')
    ax1.bar(x_indices + bar_width/2, W[0], bar_width, label='Adapted W1', color='#2ecc71', edgecolor='black')
    ax1.set_title('Reference Vector W1 (Class 1) Adaptation', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Feature Dimension', fontsize=11)
    ax1.set_ylabel('Weight Value', fontsize=11)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([f'x{i+1}' for i in range(4)])
    ax1.set_ylim(-0.2, 1.2)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    ax1.legend()

    # Prototype W2 (Class 2)
    ax2.bar(x_indices - bar_width/2, W_initial[1], bar_width, label='Initial W2', color='#e74c3c', edgecolor='black')
    ax2.bar(x_indices + bar_width/2, W[1], bar_width, label='Adapted W2', color='#f39c12', edgecolor='black')
    ax2.set_title('Reference Vector W2 (Class 2) Adaptation', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Feature Dimension', fontsize=11)
    ax2.set_ylabel('Weight Value', fontsize=11)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([f'x{i+1}' for i in range(4)])
    ax2.set_ylim(-0.2, 1.2)
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plot_file = 'lvq_prototype_adaptation.png'
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
    lvq()
