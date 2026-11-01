# =====================================================================
# BHARATIYA VIDYA BHAVAN'S SARDAR PATEL INSTITUTE OF TECHNOLOGY
# Experiment 6: Unsupervised Learning - Kohonen Self-Organizing Feature Map with Graph
# Source: AISC_exp6_C3_2024300198 (1).pdf
# =====================================================================

import matplotlib.pyplot as plt

def kohonen():
    print("=" * 65)
    print("KOHONEN SELF-ORGANIZING FEATURE MAP")
    print("=" * 65)

    W = [
        [0.3000, 0.2000, 0.1000, 0.8000, 0.4000],
        [0.5000, 0.6000, 0.7000, 0.9000, 0.2000]
    ]
    initial_W = [row.copy() for row in W]

    print("\nPART (a)")
    print("-" * 65)

    x1 = 0.2000
    x2 = 0.4000
    alpha = 0.2000

    print(f"Input Vector  = ({x1:.4f}, {x2:.4f})")
    print(f"Learning Rate = {alpha:.4f}")

    print("\nInitial Weight Matrix:")
    for row in W:
        print("[", end="  ")
        for value in row:
            print(f"{value:.4f}", end="  ")
        print("]")

    distances = []
    print("\nSquared Euclidean Distances:")
    for j in range(5):
        distance = (
            (W[0][j] - x1) ** 2 +
            (W[1][j] - x2) ** 2
        )
        distances.append(distance)
        print(f"D({j + 1}) = {distance:.4f}")

    winner = distances.index(min(distances))
    print(f"\nWinning Cluster Unit = Y{winner + 1}")

    # Update winner's weights
    W[0][winner] = (
        W[0][winner] +
        alpha * (x1 - W[0][winner])
    )
    W[1][winner] = (
        W[1][winner] +
        alpha * (x2 - W[1][winner])
    )

    print("\nUpdated Weight Matrix:")
    for row in W:
        print("[", end="  ")
        for value in row:
            print(f"{value:.4f}", end="  ")
        print("]")

    print("\n\nPART (b)")
    print("-" * 65)

    W = [
        [0.3000, 0.2000, 0.1000, 0.8000, 0.4000],
        [0.5000, 0.6000, 0.7000, 0.9000, 0.2000]
    ]

    x1_b = 0.6000
    x2_b = 0.6000
    alpha_b = 0.1000

    print(f"Input Vector  = ({x1_b:.4f}, {x2_b:.4f})")
    print(f"Learning Rate = {alpha_b:.4f}")

    distances_b = []
    for j in range(5):
        distance = (
            (W[0][j] - x1_b) ** 2 +
            (W[1][j] - x2_b) ** 2
        )
        distances_b.append(distance)

    winner_b = distances_b.index(min(distances_b))
    print(f"Winning Cluster Unit = Y{winner_b + 1}")

    W[0][winner_b] += alpha_b * (x1_b - W[0][winner_b])
    W[1][winner_b] += alpha_b * (x2_b - W[1][winner_b])

    # Graph Generation & Printing
    print("\nGenerating and displaying Kohonen SOM 2D Feature Map plot...")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Initial weights
    ax.scatter(initial_W[0], initial_W[1], color='red', s=120, label='Initial Cluster Weights (Y1-Y5)', zorder=3)
    for j in range(5):
        ax.annotate(f'Y{j+1}(init)', (initial_W[0][j], initial_W[1][j]), textcoords="offset points", xytext=(5, 5))

    # Updated weights
    ax.scatter(W[0], W[1], color='green', marker='^', s=120, label='Updated Cluster Weights', zorder=3)
    for j in range(5):
        ax.annotate(f'Y{j+1}(upd)', (W[0][j], W[1][j]), textcoords="offset points", xytext=(5, -15))

    # Input points
    ax.scatter([x1], [x2], color='blue', marker='X', s=150, label='Input Vector Part (a)', zorder=4)
    ax.scatter([x1_b], [x2_b], color='purple', marker='X', s=150, label='Input Vector Part (b)', zorder=4)

    ax.set_title('Kohonen Self-Organizing Feature Map (Weight Adaptation in 2D)')
    ax.set_xlabel('Feature Dimension 1 (w1)')
    ax.set_ylabel('Feature Dimension 2 (w2)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    plot_file = 'kohonen_som_clusters.png'
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
    kohonen()
