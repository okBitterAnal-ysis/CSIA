#!/usr/bin/env python3
# =====================================================================
# BHARATIYA VIDYA BHAVAN'S SARDAR PATEL INSTITUTE OF TECHNOLOGY
# Experiment 7: Auto Associative Memory Network
# Source: AISC_exp7_C3_2024300198.pdf
# =====================================================================

import numpy as np
import os
import matplotlib
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Training pattern
X = np.array([-1, 1, 1, 1])

# Create weight matrix using Hebbian learning
W = np.outer(X, X)

print("\nAutoassociative Memory Network")
print("\nTraining Pattern:")
print(X)

print("\nWeight Matrix:")
print(W)

# Test patterns
Test_pat = [
    [-1, 1, 1, 1],      # Original pattern
    [0, 1, 1, 1],       # One missing entry
    [-1, -1, 1, 1],     # One mistake entry
    [0, 0, 1, 1],       # Two missing entries
    [-1, -1, -1, 1]     # Two mistake entries
]

labels = [
    "Original Pattern",
    "One missing entry",
    "One mistake entry",
    "Two missing entries",
    "Two mistake entries"
]

# Required output
required_output = np.array([-1, 1, 1, 1])

# Test each pattern
for label, X_test in zip(labels, Test_pat):
    X_test = np.array(X_test)

    # Calculate net input
    Yin = np.dot(X_test, W)

    # Bipolar activation function
    Y = np.where(
        Yin > 0, 1,
        np.where(Yin < 0, -1, 0)
    )

    print("\n" + label)
    print("Input Pattern     :", X_test)
    print("Net Input         :", Yin)
    print("Recalled Pattern  :", Y)

    # Check whether the required output is obtained
    if np.array_equal(Y, required_output):
        print("Result            : Correct response")
    else:
        print("Result            : Incorrect response")

# Graph Generation & Printing
print("\nGenerating Autoassociative Memory Weight Matrix & Testing Heatmap...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Plot 1: Weight Matrix
cax1 = ax1.matshow(W, cmap='Blues')
fig.colorbar(cax1, ax=ax1)
ax1.set_title('Autoassociative Weight Matrix ($W = X^T X$)', pad=15, fontweight='bold')
for (i, j), val in np.ndenumerate(W):
    ax1.text(j, i, f'{val}', ha='center', va='center', color='white' if val > 0 else 'black', fontweight='bold')

# Plot 2: Test Pattern Recall Summary
test_matrix = np.array(Test_pat)
cax2 = ax2.matshow(test_matrix, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax2, ax=ax2)
ax2.set_title('Test Input Patterns Tested', pad=15, fontweight='bold')
ax2.set_yticks(range(len(labels)))
ax2.set_yticklabels([l.replace(" entry", "").replace(" entries", "") for l in labels], fontsize=9)
ax2.set_xticks(range(4))
ax2.set_xticklabels([f'x{i+1}' for i in range(4)])
for (i, j), val in np.ndenumerate(test_matrix):
    ax2.text(j, i, f'{val}', ha='center', va='center', color='black', fontweight='bold')

plt.tight_layout()
plot_file = 'autoassociative_memory_plot.png'
try:
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"[Graph Printed] Plot successfully saved to '{plot_file}'")
except Exception:
    pass

try:
    plt.show()
except Exception:
    pass
