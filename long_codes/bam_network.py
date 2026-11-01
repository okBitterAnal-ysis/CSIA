#!/usr/bin/env python3
# =====================================================================
# BHARATIYA VIDYA BHAVAN'S SARDAR PATEL INSTITUTE OF TECHNOLOGY
# Experiment 7: Bidirectional Associative Memory (BAM) Network with Graph
# Source: AISC_exp7_C3_2024300198.pdf
# =====================================================================

import numpy as np
import os
import matplotlib
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Input pattern E (5x3 grid representation)
E = np.array([
    1, 1, 1,
    1, -1, -1,
    1, 1, 1,
    1, -1, -1,
    1, 1, 1
])

# Input pattern F (5x3 grid representation)
F = np.array([
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    -1, -1, 1,
    -1, -1, 1
])

# Target output patterns
T_E = np.array([-1, 1])
T_F = np.array([1, 1])

# Create weight matrix using outer product rule
W = np.outer(E, T_E) + np.outer(F, T_F)

print("\nBAM Network")
print("\nWeight Matrix (15x2):")
print(W)

# Test E
Yin_E = np.dot(E, W)
Y_E = np.where(Yin_E > 0, 1, -1)

print("\nTest E")
print("Input Pattern    :", E)
print("Net Input        :", Yin_E)
print("Output           :", Y_E)
print("Expected Output  :", T_E)
print("Result           :", "Correct Response" if np.array_equal(Y_E, T_E) else "Incorrect Response")

# Test F
Yin_F = np.dot(F, W)
Y_F = np.where(Yin_F > 0, 1, -1)

print("\nTest F")
print("Input Pattern    :", F)
print("Net Input        :", Yin_F)
print("Output           :", Y_F)
print("Expected Output  :", T_F)
print("Result           :", "Correct Response" if np.array_equal(Y_F, T_F) else "Incorrect Response")

# Graph Generation & Printing
print("\nGenerating and saving BAM Pattern Visualizations (5x3 Grids)...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

# Plot Pattern E
ax1.imshow(E.reshape(5, 3), cmap='Blues_r', vmin=-1, vmax=1)
ax1.set_title(f"Stored Pattern 'E'\nTarget: {T_E}")
ax1.set_xticks(range(3))
ax1.set_yticks(range(5))
for r in range(5):
    for c in range(3):
        ax1.text(c, r, f"{E.reshape(5, 3)[r, c]}", ha='center', va='center', color='orange', fontweight='bold')

# Plot Pattern F
ax2.imshow(F.reshape(5, 3), cmap='Blues_r', vmin=-1, vmax=1)
ax2.set_title(f"Stored Pattern 'F'\nTarget: {T_F}")
ax2.set_xticks(range(3))
ax2.set_yticks(range(5))
for r in range(5):
    for c in range(3):
        ax2.text(c, r, f"{F.reshape(5, 3)[r, c]}", ha='center', va='center', color='orange', fontweight='bold')

plt.tight_layout()
plot_file = 'bam_pattern_recall.png'
try:
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"[Graph Printed] Plot successfully saved to '{plot_file}'")
except Exception:
    pass

try:
    plt.show()
except Exception:
    pass
