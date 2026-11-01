# Experiment 10: Adaptive Neuro-Fuzzy Inference System (5-Layer ANFIS)
import math

# Layer 1 Helper: Gaussian Membership Function
def gauss(x, c, s):
    return math.exp(-((x - c) ** 2) / (2 * (s ** 2)))

# User Inputs
x1 = float(input("Enter feature x1 (e.g. 2.0): "))
x2 = float(input("Enter feature x2 (e.g. 3.0): "))

# Premise Parameters: [center, sigma]
c_A1, s_A1 = 1.0, 1.0
c_A2, s_A2 = 3.0, 1.0
c_B1, s_B1 = 2.0, 1.0
c_B2, s_B2 = 4.0, 1.0

# Consequent Parameters: f = p*x1 + q*x2 + r
p1, q1, r1 = 0.5, 0.2, 1.0
p2, q2, r2 = 0.3, 0.4, 0.5

# Layer 1: Fuzzification (Membership values)
mu_A1 = gauss(x1, c_A1, s_A1)
mu_A2 = gauss(x1, c_A2, s_A2)
mu_B1 = gauss(x2, c_B1, s_B1)
mu_B2 = gauss(x2, c_B2, s_B2)

# Layer 2: Rule Firing Strengths (T-norm product)
w1 = mu_A1 * mu_B1
w2 = mu_A2 * mu_B2

# Layer 3: Normalization of Firing Strengths
w_sum = w1 + w2
w1_bar = w1 / w_sum
w2_bar = w2 / w_sum

# Layer 4: Consequent Outputs
f1 = p1 * x1 + q1 * x2 + r1
f2 = p2 * x1 + q2 * x2 + r2

# Layer 5: Overall ANFIS Crisp Output
y = (w1_bar * f1) + (w2_bar * f2)

print("\n--- 5-Layer ANFIS Architecture Output ---")
print(f"Layer 1 (Fuzzification)     : mu_A1 = {mu_A1:.4f}, mu_B1 = {mu_B1:.4f}")
print(f"                             mu_A2 = {mu_A2:.4f}, mu_B2 = {mu_B2:.4f}")
print(f"Layer 2 (Firing Strengths)  : w1 = {w1:.4f}, w2 = {w2:.4f}")
print(f"Layer 3 (Normalized Weights): w1_bar = {w1_bar:.4f}, w2_bar = {w2_bar:.4f}")
print(f"Layer 4 (Consequents)       : f1 = {f1:.4f}, f2 = {f2:.4f}")
print(f"Layer 5 (Final ANFIS Output): Y = {y:.4f}")
