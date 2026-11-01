# Experiment 7: Autoassociative Memory Network (Hebbian Learning)
import numpy as np

# Training Pattern
x = np.array(list(map(int, input("Enter bipolar training pattern (e.g. -1 1 1 1): ").split())))

# 1. Outer Product Hebbian Learning: W = X^T * X
W = np.outer(x, x)
print("\n--- Weight Matrix (W) ---")
print(W)

# 2. Test with Input Pattern
x_test = np.array(list(map(int, input("\nEnter test pattern with missing/mistake entries (e.g. 0 1 1 1): ").split())))

# 3. Recall Net Input and Bipolar Activation
yin = np.dot(x_test, W)
y = np.where(yin > 0, 1, np.where(yin < 0, -1, 0))

# 4. Results
print("\n--- Recall Results ---")
print("Input Test Pattern :", x_test)
print("Calculated Yin     :", yin)
print("Recalled Pattern   :", y)
print("Original Pattern   :", x)
print("Recall Successful? :", np.array_equal(y, x))
