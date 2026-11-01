#!/usr/bin/env python3
# =====================================================================
# BHARATIYA VIDYA BHAVAN'S SARDAR PATEL INSTITUTE OF TECHNOLOGY
# Experiment 3: Design Full Adder using McCulloch-Pitts Neuron
# Source: AISC_Exp3_2024300198 (3).pdf
# =====================================================================

def mp_neuron(inputs, weights, threshold):
    """
    McCulloch-Pitts Neuron:
    y_in = sum(x_i * w_i)
    f(y_in) = 1 if y_in >= threshold else 0
    """
    y_in = sum(x * w for x, w in zip(inputs, weights))
    return 1 if y_in >= threshold else 0

def half_adder(x1, x2):
    # Carry: AND gate (theta = 2)
    carry = mp_neuron(inputs=[x1, x2], weights=[1, 1], threshold=2)

    # Sum: XOR gate via two ANDNOT neurons and one OR neuron
    z1 = mp_neuron(inputs=[x1, x2], weights=[1, -1], threshold=1)    # x1 AND NOT x2
    z2 = mp_neuron(inputs=[x1, x2], weights=[-1, 1], threshold=1)    # NOT x1 AND x2
    sum_out = mp_neuron(inputs=[z1, z2], weights=[1, 1], threshold=1)

    return sum_out, carry

def full_adder(a, b, cin):
    # Stage 1: Half Adder on A and B
    s1, c1 = half_adder(a, b)

    # Stage 2: Half Adder on intermediate Sum (s1) and Cin
    final_sum, c2 = half_adder(s1, cin)

    # Stage 3: OR neuron to combine carries (c1 OR c2)
    cout = mp_neuron(inputs=[c1, c2], weights=[1, 1], threshold=1)

    return final_sum, cout

# --- Test & Output Truth Table ---
if __name__ == "__main__":
    print(f"{'A':^4} | {'B':^4} | {'Cin':^5} | {'Sum':^5} | {'Cout':^6}")
    print("-" * 34)

    test_cases = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ]

    for a, b, cin in test_cases:
        s, cout = full_adder(a, b, cin)
        print(f"{a:^4} | {b:^4} | {cin:^5} | {s:^5} | {cout:^6}")
