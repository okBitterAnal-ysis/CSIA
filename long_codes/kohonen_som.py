# =====================================================================
# BHARATIYA VIDYA BHAVAN'S SARDAR PATEL INSTITUTE OF TECHNOLOGY
# Experiment 6: Unsupervised Learning - Kohonen Self-Organizing Feature Map
# Source: AISC_exp6_C3_2024300198 (1).pdf
# =====================================================================

def kohonen():
    print("=" * 65)
    print("KOHONEN SELF-ORGANIZING FEATURE MAP")
    print("=" * 65)

    W = [
        [0.3000, 0.2000, 0.1000, 0.8000, 0.4000],
        [0.5000, 0.6000, 0.7000, 0.9000, 0.2000]
    ]

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

    x1 = 0.6000
    x2 = 0.6000
    alpha = 0.1000

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

if __name__ == "__main__":
    kohonen()
