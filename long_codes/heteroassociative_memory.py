# =====================================================================
# BHARATIYA VIDYA BHAVAN'S SARDAR PATEL INSTITUTE OF TECHNOLOGY
# Experiment 7: Heteroassociative Memory Network
# Source: AISC_exp7_C3_2024300198.pdf
# =====================================================================

import numpy as np
import matplotlib.pyplot as plt

# Training input vectors
S = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 1]
])

# Target output vectors
T = np.array([
    [0, 1],
    [0, 1],
    [1, 0],
    [1, 0]
])

# Initialize weight matrix
W = np.zeros((4, 2), dtype=int)

# Outer product learning rule
for i in range(len(S)):
    W = W + np.outer(S[i], T[i])

print("\nHeteroassociative Memory Network")
print("\nTraining Input Vectors:")
print(S)

print("\nTarget Output Vectors:")
print(T)

print("\nWeight Matrix:")
print(W)

# Test the network using all training inputs
for i in range(len(S)):
    X = S[i]

    # Calculate net input
    Yin = np.dot(X, W)

    # Binary activation function
    Y = np.where(Yin > 0, 1, 0)

    print("\nTest", i + 1)
    print("Input Pattern    :", X)
    print("Net Input        :", Yin)
    print("Output           :", Y)
    print("Expected Output  :", T[i])

    # Check whether output matches expected target
    if np.array_equal(Y, T[i]):
        print("Result           : Correct Response")
    else:
        print("Result           : Incorrect Response")

# Graph Generation & Printing
print("\nGenerating Heteroassociative Memory Weight Matrix & Association Map...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Plot 1: Weight Matrix
cax1 = ax1.matshow(W, cmap='Purples')
fig.colorbar(cax1, ax=ax1)
ax1.set_title(r'Heteroassociative Weight Matrix ($W = \sum S_i^T T_i$)', pad=15, fontweight='bold')
ax1.set_xlabel('Output Units ($y_1, y_2$)', fontsize=10)
ax1.set_ylabel('Input Units ($x_1..x_4$)', fontsize=10)
for (i, j), val in np.ndenumerate(W):
    ax1.text(j, i, f'{val}', ha='center', va='center', color='white' if val > 1 else 'black', fontweight='bold')

# Plot 2: Input Patterns & Target Class
cax2 = ax2.matshow(S, cmap='Blues')
fig.colorbar(cax2, ax=ax2)
ax2.set_title('Training Input Vectors ($S$)', pad=15, fontweight='bold')
ax2.set_yticks(range(4))
ax2.set_yticklabels([f'Pattern {i+1} $\\rightarrow$ {T[i]}' for i in range(4)], fontsize=10)
ax2.set_xticks(range(4))
ax2.set_xticklabels([f'x{i+1}' for i in range(4)])
for (i, j), val in np.ndenumerate(S):
    ax2.text(j, i, f'{val}', ha='center', va='center', color='white' if val == 1 else 'black', fontweight='bold')

plt.tight_layout()
plot_file = 'heteroassociative_memory_plot.png'
try:
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"[Graph Printed] Plot successfully saved to '{plot_file}'")
except Exception:
    pass

try:
    plt.show()
except Exception:
    pass
