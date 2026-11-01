#!/usr/bin/env python3
# Experiment 7: Associative Memory Networks (Autoassociative & BAM)
import numpy as np

print("=" * 55)
print("       ASSOCIATIVE MEMORY NETWORKS")
print("=" * 55)

# PART 1: AUTOASSOCIATIVE MEMORY
print("\n[PART 1: AUTOASSOCIATIVE MEMORY]")
x_auto = np.array([-1, 1, 1, 1])
W_auto = np.outer(x_auto, x_auto)
print("Training Pattern:", x_auto)
print("Weight Matrix:\n", W_auto)

test_input = input("Enter test vector (e.g. 0 1 1 1): ").strip()
x_test = np.array(list(map(int, test_input.split())))

yin_auto = np.dot(x_test, W_auto)
y_auto = np.where(yin_auto > 0, 1, np.where(yin_auto < 0, -1, 0))
print(f"Test: {x_test} -> Yin: {yin_auto} -> Recalled: {y_auto} (Correct? {np.array_equal(y_auto, x_auto)})")

# PART 2: BIDIRECTIONAL ASSOCIATIVE MEMORY (BAM)
print("\n[PART 2: BIDIRECTIONAL ASSOCIATIVE MEMORY (BAM)]")
E = np.array([1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1])
F = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1])
T_E = np.array([-1, 1])
T_F = np.array([1, 1])

W_bam = np.outer(E, T_E) + np.outer(F, T_F)
print("BAM Weight Matrix calculated (15 x 2).")

choice = input("Enter pattern to test in BAM (E or F): ").strip().upper()
test_vec, target = (E, T_E) if choice == "E" else (F, T_F)
yin_bam = np.dot(test_vec, W_bam)
y_bam = np.where(yin_bam > 0, 1, -1)

print(f"Test Pattern {choice} -> Yin: {yin_bam} -> Output: {y_bam} (Target: {target}, Correct? {np.array_equal(y_bam, target)})")
