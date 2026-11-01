#!/usr/bin/env python3
# =====================================================================
# BHARATIYA VIDYA BHAVAN'S SARDAR PATEL INSTITUTE OF TECHNOLOGY
# Experiment 10: Adaptive Neuro-Fuzzy Inference System (ANFIS) with Graph
# 5-Layer First-Order Sugeno Neuro-Fuzzy Architecture
# =====================================================================

import math
import numpy as np
import os
import matplotlib
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

def gaussian_membership(x, c, sigma):
    """Layer 1: Computes Gaussian membership grade for given center and spread."""
    return math.exp(-((x - c) ** 2) / (2 * (sigma ** 2)))

def run_anfis_simulation(x1, x2):
    """
    Simulates the forward pass of a 5-layer ANFIS model:
    Layer 1: Fuzzification using Gaussian membership functions
    Layer 2: Rule Firing Strengths (Product T-norm)
    Layer 3: Normalization of Firing Strengths
    Layer 4: First-order Sugeno linear consequences: f_i = p_i*x1 + q_i*x2 + r_i
    Layer 5: Overall Defuzzified Crisp Output: y = sum(w_bar_i * f_i)
    """
    # Premise Parameters: [center, sigma]
    c_A1, s_A1 = 1.0, 1.0
    c_A2, s_A2 = 3.0, 1.0
    c_B1, s_B1 = 2.0, 1.0
    c_B2, s_B2 = 4.0, 1.0

    # Consequent Parameters: f_i = p_i * x1 + q_i * x2 + r_i
    p1, q1, r1 = 0.5, 0.2, 1.0
    p2, q2, r2 = 0.3, 0.4, 0.5

    # Layer 1: Membership Grades
    mu_A1 = gaussian_membership(x1, c_A1, s_A1)
    mu_A2 = gaussian_membership(x1, c_A2, s_A2)
    mu_B1 = gaussian_membership(x2, c_B1, s_B1)
    mu_B2 = gaussian_membership(x2, c_B2, s_B2)

    # Layer 2: Rule Firing Strengths (T-norm product)
    w1 = mu_A1 * mu_B1
    w2 = mu_A2 * mu_B2

    # Layer 3: Normalized Firing Strengths
    w_sum = w1 + w2
    w1_bar = w1 / w_sum
    w2_bar = w2 / w_sum

    # Layer 4: Consequent Polynomial Evaluation
    f1 = p1 * x1 + q1 * x2 + r1
    f2 = p2 * x1 + q2 * x2 + r2

    # Layer 5: Overall Sum of Outputs
    final_output = (w1_bar * f1) + (w2_bar * f2)

    print("=" * 65)
    print("5-LAYER ADAPTIVE NEURO-FUZZY INFERENCE SYSTEM (ANFIS)")
    print("=" * 65)
    print(f"Inputs: x1 = {x1:.4f}, x2 = {x2:.4f}")
    print("-" * 65)
    print("Layer 1 (Fuzzification - Gaussian Memberships):")
    print(f"  Rule 1 Premise: mu_A1(x1) = {mu_A1:.4f}, mu_B1(x2) = {mu_B1:.4f}")
    print(f"  Rule 2 Premise: mu_A2(x1) = {mu_A2:.4f}, mu_B2(x2) = {mu_B2:.4f}")
    print("\nLayer 2 (Rule Firing Strengths - T-Norm Product):")
    print(f"  w1 = mu_A1 * mu_B1 = {w1:.4f}")
    print(f"  w2 = mu_A2 * mu_B2 = {w2:.4f}")
    print("\nLayer 3 (Normalized Firing Strengths):")
    print(f"  w1_bar = w1 / (w1 + w2) = {w1_bar:.4f}")
    print(f"  w2_bar = w2 / (w1 + w2) = {w2_bar:.4f}")
    print("\nLayer 4 (Consequent Outputs - Sugeno Linear Functions):")
    print(f"  f1 = {p1}*x1 + {q1}*x2 + {r1} = {f1:.4f}")
    print(f"  f2 = {p2}*x1 + {q2}*x2 + {r2} = {f2:.4f}")
    print("\nLayer 5 (Overall Crisp System Output):")
    print(f"  Y = (w1_bar * f1) + (w2_bar * f2) = {final_output:.4f}")
    print("=" * 65)

    # Graph Generation & Printing
    print("\nGenerating and saving ANFIS Gaussian Membership Functions plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x_range = np.linspace(-1, 6, 200)

    # Input 1 Memberships
    gauss_A1 = np.exp(-((x_range - c_A1) ** 2) / (2 * (s_A1 ** 2)))
    gauss_A2 = np.exp(-((x_range - c_A2) ** 2) / (2 * (s_A2 ** 2)))
    ax1.plot(x_range, gauss_A1, 'b-', label='A1 (c=1, s=1)')
    ax1.plot(x_range, gauss_A2, 'r-', label='A2 (c=3, s=1)')
    ax1.axvline(x1, color='k', linestyle='--', label=f'Input x1 = {x1}')
    ax1.set_title('ANFIS Feature x1 Gaussian Memberships')
    ax1.set_xlabel('Feature x1')
    ax1.set_ylabel('Membership Grade')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Input 2 Memberships
    gauss_B1 = np.exp(-((x_range - c_B1) ** 2) / (2 * (s_B1 ** 2)))
    gauss_B2 = np.exp(-((x_range - c_B2) ** 2) / (2 * (s_B2 ** 2)))
    ax2.plot(x_range, gauss_B1, 'g-', label='B1 (c=2, s=1)')
    ax2.plot(x_range, gauss_B2, 'm-', label='B2 (c=4, s=1)')
    ax2.axvline(x2, color='k', linestyle='--', label=f'Input x2 = {x2}')
    ax2.set_title('ANFIS Feature x2 Gaussian Memberships')
    ax2.set_xlabel('Feature x2')
    ax2.set_ylabel('Membership Grade')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plot_file = 'anfis_architecture_plot.png'
    try:
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"[Graph Printed] Plot successfully saved to '{plot_file}'")
    except Exception:
        pass

    try:
        plt.show()
    except Exception:
        pass

    return final_output

if __name__ == "__main__":
    val_x1 = float(input("Enter input feature x1 (e.g., 2.0): "))
    val_x2 = float(input("Enter input feature x2 (e.g., 3.0): "))
    run_anfis_simulation(val_x1, val_x2)
