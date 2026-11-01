# Experiment 6: Learning Vector Quantization (LVQ)

# Initial Reference Vectors and their Classes (from lab sheet)
W = [
    [0.0, 0.0, 1.0, 1.0],  # Reference Vector 1 (Class 1)
    [1.0, 0.0, 0.0, 0.0]   # Reference Vector 2 (Class 2)
]
w_class = [1, 2]

print("--- Initial Reference Vectors ---")
for idx, row in enumerate(W):
    print(f"W{idx + 1} (Class {w_class[idx]}): {row}")

# Interactive inputs
alpha = float(input("\nEnter learning rate alpha (e.g. 0.1): "))
x = list(map(float, input("Enter 4 input features separated by space (e.g. 0 0 0 1): ").split()))
target = int(input("Enter target class (1 or 2, e.g. 2): "))

# 1. Compute squared Euclidean distance to each reference vector
distances = []
print("\nSquared Euclidean Distances:")
for j in range(len(W)):
    d = sum((W[j][k] - x[k]) ** 2 for k in range(4))
    distances.append(d)
    print(f"D(W{j + 1}) = {d:.4f}")

# 2. Determine winning unit
winner = distances.index(min(distances))
print(f"\nWinning Unit : W{winner + 1} (Class {w_class[winner]})")
print(f"Target Class : {target}")

# 3. Update winning unit weights
if w_class[winner] == target:
    print("Class MATCHES target -> Moving weights TOWARDS the input.")
    for k in range(4):
        W[winner][k] += alpha * (x[k] - W[winner][k])
else:
    print("Class DOES NOT match target -> Moving weights AWAY from the input.")
    for k in range(4):
        W[winner][k] -= alpha * (x[k] - W[winner][k])

# 4. Display updated vectors
print("\n--- Updated Reference Vectors ---")
for idx, row in enumerate(W):
    print(f"W{idx + 1}: {[round(v, 4) for v in row]}")
