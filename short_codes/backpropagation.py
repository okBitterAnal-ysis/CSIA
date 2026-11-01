#!/usr/bin/env python3
# Experiment 5: Supervised Learning (Backpropagation Neural Network)
import math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

# 1. Inputs & Hyperparameters
alpha = float(input("Enter learning rate alpha (e.g. 0.25): "))
target = float(input("Enter target output (e.g. 1.0): "))
x1 = float(input("Enter input x1 (e.g. 0.0): "))
x2 = float(input("Enter input x2 (e.g. 1.0): "))

print("\nInput -> Hidden weights:")
v11 = float(input("Enter v11 (e.g. 0.6): "))
v21 = float(input("Enter v21 (e.g. -0.1): "))
v01 = float(input("Enter bias v01 (e.g. 0.3): "))

v12 = float(input("Enter v12 (e.g. -0.3): "))
v22 = float(input("Enter v22 (e.g. 0.4): "))
v02 = float(input("Enter bias v02 (e.g. 0.5): "))

print("\nHidden -> Output weights:")
w1 = float(input("Enter w1 (e.g. 0.4): "))
w2 = float(input("Enter w2 (e.g. 0.1): "))
w0 = float(input("Enter bias w0 (e.g. -0.2): "))

# 2. Forward Pass
z1 = sigmoid(v01 + x1 * v11 + x2 * v21)
z2 = sigmoid(v02 + x1 * v12 + x2 * v22)
y = sigmoid(w0 + z1 * w1 + z2 * w2)

# 3. Output Error & Deltas
delta_y = (target - y) * y * (1 - y)
dw1 = alpha * delta_y * z1
dw2 = alpha * delta_y * z2
dw0 = alpha * delta_y

# 4. Hidden Deltas & Backpropagation
delta_z1 = delta_y * w1 * z1 * (1 - z1)
delta_z2 = delta_y * w2 * z2 * (1 - z2)

dv11 = alpha * delta_z1 * x1
dv21 = alpha * delta_z1 * x2
dv01 = alpha * delta_z1
dv12 = alpha * delta_z2 * x1
dv22 = alpha * delta_z2 * x2
dv02 = alpha * delta_z2

# 5. Output Results
print("\n--- Backpropagation Results ---")
print(f"Hidden Outputs: z1 = {z1:.6f}, z2 = {z2:.6f}")
print(f"Network Output: y = {y:.6f} (Target = {target})")
print(f"Output Delta  : {delta_y:.6f}")

print("\n--- Updated Hidden -> Output Weights ---")
print(f"w1(new) = {w1 + dw1:.6f}")
print(f"w2(new) = {w2 + dw2:.6f}")
print(f"w0(new) = {w0 + dw0:.6f}")

print("\n--- Updated Input -> Hidden Weights ---")
print(f"v11(new) = {v11 + dv11:.6f}, v21(new) = {v21 + dv21:.6f}, v01(new) = {v01 + dv01:.6f}")
print(f"v12(new) = {v12 + dv12:.6f}, v22(new) = {v22 + dv22:.6f}, v02(new) = {v02 + dv02:.6f}")
