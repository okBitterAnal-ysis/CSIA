#!/usr/bin/env python3
# Experiment 9: Fuzzy Logic Controller (Mamdani Inference)

def trimf(x, a, b, c):
    """Triangular membership function without external dependencies."""
    if x <= a or x >= c:
        return 0.0
    if a < x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    return (c - x) / (c - b) if c != b else 1.0

# 1. Inputs
t = float(input("Enter temperature in °C (0 to 50, e.g. 24): "))
h = float(input("Enter humidity in % (0 to 100, e.g. 60): "))

# 2. Fuzzification
temp_cold = trimf(t, 0, 0, 20)
temp_med = trimf(t, 15, 25, 35)
temp_hot = trimf(t, 30, 50, 50)

hum_low = trimf(h, 0, 0, 50)
hum_high = trimf(h, 50, 100, 100)

# 3. Fuzzy Rules (Mamdani Min-Max Inference)
# Rule 1: IF Temp is Cold THEN Fan is Low
# Rule 2: IF Temp is Medium AND Humidity is Low THEN Fan is Medium
# Rule 3: IF Temp is Medium AND Humidity is High THEN Fan is High
# Rule 4: IF Temp is Hot THEN Fan is High
rule1 = temp_cold
rule2 = min(temp_med, hum_low)
rule3 = min(temp_med, hum_high)
rule4 = temp_hot

fan_low = rule1
fan_med = rule2
fan_high = max(rule3, rule4)

# 4. Defuzzification (Centroid of Aggregated Output)
num, den = 0.0, 0.0
for speed in range(0, 101):
    mu_low = min(fan_low, trimf(speed, 0, 0, 50))
    mu_med = min(fan_med, trimf(speed, 25, 50, 75))
    mu_high = min(fan_high, trimf(speed, 50, 100, 100))
    mu = max(mu_low, mu_med, mu_high)
    num += speed * mu
    den += mu

fan_speed = (num / den) if den != 0 else 50.0

# 5. Output Result
print("\n--- Fuzzy Controller Output ---")
print(f"Temperature: {t} °C")
print(f"Humidity   : {h} %")
print(f"Fan Speed  : {fan_speed:.2f} %")
