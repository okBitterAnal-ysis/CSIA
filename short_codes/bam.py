#!/usr/bin/env python3
# Experiment 7: Bidirectional Associative Memory (BAM) Network
import numpy as np

# Two Bipolar Pattern Pairs (E and F from lab sheet)
E = np.array([1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1])
F = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1])
T_E = np.array([-1, 1])
T_F = np.array([1, 1])

# Weight Matrix: W = E^T * T_E + F^T * T_F
W = np.outer(E, T_E) + np.outer(F, T_F)
print("--- BAM Weight Matrix (15 x 2) ---")
print(W)

# Interactive choice
choice = input("\nTest stored pattern E or F? (Enter E or F): ").strip().upper()
x_test = E if choice == "E" else F
expected = T_E if choice == "E" else T_F

# Forward Recall: Yin = X * W, bipolar step activation
yin = np.dot(x_test, W)
y = np.where(yin > 0, 1, -1)

print(f"\n--- Recall for Pattern {choice} ---")
print("Net Input Yin    :", yin)
print("Recalled Output  :", y)
print("Expected Target  :", expected)
print("Correct Recall?  :", np.array_equal(y, expected))
