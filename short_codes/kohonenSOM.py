#!/usr/bin/env python3
# Experiment 6: Kohonen Self-Organizing Feature Map (KSOFM)

# Initial Weight Matrix: 2 features x 5 cluster units (from lab sheet)
W = [
    [0.3, 0.2, 0.1, 0.8, 0.4],
    [0.5, 0.6, 0.7, 0.9, 0.2]
]

print("--- Initial Weight Matrix (2 x 5) ---")
for row in W:
    print([round(v, 4) for v in row])

# Interactive inputs
x1 = float(input("\nEnter feature x1 (e.g. 0.2): "))
x2 = float(input("Enter feature x2 (e.g. 0.4): "))
alpha = float(input("Enter learning rate alpha (e.g. 0.2): "))

# 1. Calculate squared Euclidean distance to each cluster unit
distances = []
print("\nSquared Euclidean Distances:")
for j in range(5):
    d = (W[0][j] - x1) ** 2 + (W[1][j] - x2) ** 2
    distances.append(d)
    print(f"D({j + 1}) = {d:.4f}")

# 2. Select winning cluster unit (Best Matching Unit)
winner = distances.index(min(distances))
print(f"\nWinning Cluster Unit: Y{winner + 1}")

# 3. Update winning unit's weights
W[0][winner] += alpha * (x1 - W[0][winner])
W[1][winner] += alpha * (x2 - W[1][winner])

# 4. Display Updated Weights
print("\n--- Updated Weight Matrix ---")
for row in W:
    print([round(v, 4) for v in row])
