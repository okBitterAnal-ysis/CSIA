#!/usr/bin/env python3
# Experiment 4: Single Layer Perceptron (Bipolar Logic Gate Learning)

alpha = float(input("Enter learning rate alpha (e.g. 1.0): "))
theta = float(input("Enter threshold theta (e.g. 0.0): "))
epochs = int(input("Enter max epochs (e.g. 5): "))
n = int(input("Enter number of samples (e.g. 4 for AND gate): "))

print("Enter each sample as 'x1 x2 target' (e.g. for Bipolar AND: 1 1 1 / 1 -1 -1 / -1 1 -1 / -1 -1 -1):")
data = [list(map(float, input().split())) for _ in range(n)]

w1, w2, b = 0.0, 0.0, 0.0

def step(yin):
    return 1 if yin > theta else (-1 if yin < -theta else 0)

print("\nEpoch |  x1    x2    t  |  yin    y  |   w1     w2     b")
print("-" * 55)

for ep in range(1, epochs + 1):
    errors = 0
    for x1, x2, t in data:
        yin = b + w1 * x1 + w2 * x2
        y = step(yin)
        if y != t:
            w1 += alpha * t * x1
            w2 += alpha * t * x2
            b += alpha * t
            errors += 1
        print(f"{ep:^5} | {x1:^5.0f} {x2:^5.0f} {t:^5.0f} | {yin:^5.1f} {y:^5.0f} | {w1:^6.1f} {w2:^6.1f} {b:^6.1f}")
    if errors == 0:
        print(f"\nConverged at Epoch {ep} with zero errors!")
        break

print(f"\nFinal Weights: w1 = {w1:.2f}, w2 = {w2:.2f}, bias = {b:.2f}")
