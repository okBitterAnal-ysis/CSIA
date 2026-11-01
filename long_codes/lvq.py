# =====================================================================
# BHARATIYA VIDYA BHAVAN'S SARDAR PATEL INSTITUTE OF TECHNOLOGY
# Experiment 6: Learning Vector Quantization (LVQ)
# Source: AISC_exp6_C3_2024300198 (1).pdf
# =====================================================================

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

if __name__ == "__main__":
    lvq()
