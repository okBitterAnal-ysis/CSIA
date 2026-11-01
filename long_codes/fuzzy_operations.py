# =====================================================================
# BHARATIYA VIDYA BHAVAN'S SARDAR PATEL INSTITUTE OF TECHNOLOGY
# Experiment 8: Fuzzy Sets and Fuzzy Relations Operations
# Source: AISC_EXP8_dhirensacher_2024300198.pdf
# =====================================================================

import numpy as np
np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, legacy='1.13')

# ---------------------------------------------------------------------
# Problem 1: Basic Fuzzy Set Operations (Union, Intersection, Complement)
# ---------------------------------------------------------------------
print("=" * 60)
print("Problem 1: Basic Fuzzy Set Operations")
print("=" * 60)

mach = np.array([0.64, 0.645, 0.65, 0.655, 0.66])
A = np.array([0, 0.75, 1, 0.5, 0])
B = np.array([0, 0.25, 0.75, 1, 0.5])

union = np.maximum(A, B)
intersection = np.minimum(A, B)
A_complement = 1 - A
B_complement = 1 - B

print("Mach:", mach)
print("A:", A)
print("B:", B)

print("\na) A ∪ B:", union)
print("\nb) A ∩ B:", intersection)
print("\nc) A':", A_complement)
print("\nd) B':", B_complement)
print("\ne) A' ∩ B':", np.minimum(A_complement, B_complement))
print("\nf) A' ∪ B':", np.maximum(A_complement, B_complement))

# ---------------------------------------------------------------------
# Problem 2: Algebraic and Bounded Fuzzy Operations
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("Problem 2: Algebraic and Bounded Operations")
print("=" * 60)

elements = np.array([1, 2, 3, 4])
A2 = np.array([0.2, 0.3, 0.4, 0.5])
B2 = np.array([0.1, 0.2, 0.2, 1])

sum_val = A2 + B2
sub_val = A2 - B2
mul_val = A2 * B2

algebraic_sum = sum_val - mul_val
algebraic_pdt = mul_val
Bounded_sum = np.minimum(sum_val, 1)
Bounded_diff = np.maximum(sub_val, 0)

print("Algebraic Sum        : ", algebraic_sum)
print("Algebraic Product    : ", algebraic_pdt)
print("Bounded sum          : ", Bounded_sum)
print("Bounded difference   : ", Bounded_diff)

# ---------------------------------------------------------------------
# Problem 3: Cartesian Product of Fuzzy Sets
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("Problem 3: Cartesian Product (A x B)")
print("=" * 60)

A3 = np.array([0.3, 0.7, 1.0])
B3 = np.array([0.4, 0.9])

R = np.zeros((len(A3), len(B3)))

for i in range(len(A3)):
    for j in range(len(B3)):
        R[i][j] = min(A3[i], B3[j])

print("Fuzzy Set A:", A3)
print("Fuzzy Set B:", B3)
print("\nCartesian Product R = A x B:")
print(R)

# ---------------------------------------------------------------------
# Problem 4: Fuzzy Relations Composition (Max-Min and Max-Product)
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("Problem 4: Relations Composition (Max-Min and Max-Product)")
print("=" * 60)

R_rel = np.array([
    [0.6, 0.3],
    [0.2, 0.9]
])

S_rel = np.array([
    [1.0, 0.5, 0.3],
    [0.8, 0.4, 0.7]
])

# Max-Min Composition
T_max_min = np.zeros((R_rel.shape[0], S_rel.shape[1]))
for i in range(R_rel.shape[0]):
    for j in range(S_rel.shape[1]):
        values = []
        for k in range(R_rel.shape[1]):
            values.append(min(R_rel[i][k], S_rel[k][j]))
        T_max_min[i][j] = max(values)

print("Max-Min Composition:")
print(T_max_min)

# Max-Product Composition
T_max_product = np.zeros((R_rel.shape[0], S_rel.shape[1]))
for i in range(R_rel.shape[0]):
    for j in range(S_rel.shape[1]):
        values = []
        for k in range(R_rel.shape[1]):
            values.append(R_rel[i][k] * S_rel[k][j])
        T_max_product[i][j] = max(values)

print("\nMax-Product Composition:")
print(T_max_product)
