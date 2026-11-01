#!/usr/bin/env python3
# =====================================================================
# BHARATIYA VIDYA BHAVAN'S SARDAR PATEL INSTITUTE OF TECHNOLOGY
# Experiment 3: Design Half Adder using McCulloch-Pitts Neuron
# Source: AISC_Exp3_2024300198 (3).pdf
# =====================================================================

def mp_neuron(inputs, weights, threshold):
    """
    McCulloch-Pitts Neuron Model:
    Calculates net input y_in = sum(x_i * w_i)
    Applies Binary Step Activation: f(y_in) = 1 if y_in >= threshold else 0
    """
    y_in = sum(x * w for x, w in zip(inputs, weights))
    return 1 if y_in >= threshold else 0

def half_adder(x1, x2):
    # 1. Carry: AND gate (w1=1, w2=1, theta=2)
    carry = mp_neuron(inputs=[x1, x2], weights=[1, 1], threshold=2)

    # 2. Sum: XOR logic decomposed via McCulloch-Pitts
    # Layer 1: Two ANDNOT neurons
    z1 = mp_neuron(inputs=[x1, x2], weights=[1, -1], threshold=1)    # x1 AND NOT x2
    z2 = mp_neuron(inputs=[x1, x2], weights=[-1, 1], threshold=1)    # NOT x1 AND x2

    # Layer 2: OR neuron combining z1 and z2 (w1=1, w2=1, theta=1)
    sum_out = mp_neuron(inputs=[z1, z2], weights=[1, 1], threshold=1)

    return sum_out, carry

if __name__ == "__main__":
    print(f"{'x1':^4} | {'x2':^4} | {'Sum':^5} | {'Carry':^5}")
    print("-" * 26)

    test_cases = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for x1, x2 in test_cases:
        s, c = half_adder(x1, x2)
        print(f"{x1:^4} | {x2:^4} | {s:^5} | {c:^5}")
