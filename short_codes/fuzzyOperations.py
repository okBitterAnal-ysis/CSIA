#!/usr/bin/env python3
# Experiment 8: Fuzzy Sets and Relations Operations
import numpy as np

# 1. Fuzzy Sets Inputs
a = np.array(list(map(float, input("Enter Fuzzy Set A (e.g. 0.2 0.3 0.4 0.5): ").split())))
b = np.array(list(map(float, input("Enter Fuzzy Set B (e.g. 0.1 0.2 0.2 1.0): ").split())))

print("\n--- Fuzzy Set Operations ---")
print("Union (A ∪ B)       :", np.maximum(a, b))
print("Intersection (A ∩ B):", np.minimum(a, b))
print("Complement A'       :", np.round(1 - a, 2))
print("Algebraic Sum       :", np.round(a + b - (a * b), 2))
print("Algebraic Product   :", np.round(a * b, 2))
print("Bounded Sum         :", np.minimum(a + b, 1.0))
print("Bounded Difference  :", np.maximum(a - b, 0.0))

# Cartesian Product
print("\nCartesian Product (A × B):")
cartesian = np.minimum(a[:, None], b[None, :])
print(np.round(cartesian, 2))

# 2. Fuzzy Relations Composition
print("\n--- Fuzzy Relations Composition ---")
r_rows = int(input("Enter number of rows in Relation R (e.g. 2): "))
print("Enter rows of R one by one (e.g. 0.6 0.3):")
R = np.array([list(map(float, input().split())) for _ in range(r_rows)])

s_rows = int(input("Enter number of rows in Relation S (e.g. 2): "))
print("Enter rows of S one by one (e.g. 1.0 0.5 0.3):")
S = np.array([list(map(float, input().split())) for _ in range(s_rows)])

# Compositions
max_min = np.array([[np.max(np.minimum(r_row, S[:, col])) for col in range(S.shape[1])] for r_row in R])
max_prod = np.array([[np.max(r_row * S[:, col]) for col in range(S.shape[1])] for r_row in R])

print("\nMax-Min Composition (R ∘ S):")
print(np.round(max_min, 2))

print("\nMax-Product Composition (R · S):")
print(np.round(max_prod, 2))
