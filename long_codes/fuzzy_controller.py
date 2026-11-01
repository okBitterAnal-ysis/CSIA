#!/usr/bin/env python3
# =====================================================================
# BHARATIYA VIDYA BHAVAN'S SARDAR PATEL INSTITUTE OF TECHNOLOGY
# Experiment 9: Fuzzy Logic Controller (Mamdani Inference System) with Graphs
# Source: EXP9_AISC_2024300198.pdf
# =====================================================================

import os
import numpy as np
import matplotlib
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

def trimf(x, abc):
    """Triangular membership function: abc = [a, b, c]."""
    a, b, c = abc
    y = np.zeros_like(x, dtype=float)
    if a != b:
        idx1 = (a < x) & (x <= b)
        y[idx1] = (x[idx1] - a) / (b - a)
    else:
        idx1 = (x == a)
        y[idx1] = 1.0
    if b != c:
        idx2 = (b < x) & (x < c)
        y[idx2] = (c - x[idx2]) / (c - b)
    else:
        idx2 = (x == c)
        y[idx2] = 1.0
    idx_peak = (x == b)
    y[idx_peak] = 1.0
    return y

def trimf_point(x, abc):
    """Scalar triangular membership grade."""
    a, b, c = abc
    if x <= a or x >= c:
        return 0.0
    if a < x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    return (c - x) / (c - b) if c != b else 1.0

def run_fuzzy_air_conditioner(temperature_val, humidity_val):
    """
    Implements a Mamdani Fuzzy Controller for Air Conditioner Fan Speed.
    Inputs:
      - Temperature (0 to 50 deg C): Cold, Medium, Hot
      - Humidity (0 to 100 %): Low, High
    Output:
      - Fan Speed (0 to 100 %): Low, Medium, High
    """
    # 1. Universes of Discourse
    x_temp = np.arange(0, 51, 1)
    x_hum = np.arange(0, 101, 1)
    x_fan = np.arange(0, 101, 1)

    # 2. Membership Function Definitions
    cold_params = [0, 0, 20]
    med_temp_params = [15, 25, 35]
    hot_params = [30, 50, 50]

    low_hum_params = [0, 0, 50]
    high_hum_params = [50, 100, 100]

    low_fan_params = [0, 0, 50]
    med_fan_params = [25, 50, 75]
    high_fan_params = [50, 100, 100]

    # 3. Fuzzification of Crisp Inputs
    mu_cold = trimf_point(temperature_val, cold_params)
    mu_med_temp = trimf_point(temperature_val, med_temp_params)
    mu_hot = trimf_point(temperature_val, hot_params)

    mu_low_hum = trimf_point(humidity_val, low_hum_params)
    mu_high_hum = trimf_point(humidity_val, high_hum_params)

    # 4. Fuzzy Rule Evaluation (Mamdani Min-Max Inference)
    # Rule 1: IF Temperature is Cold THEN Fan Speed is Low
    r1 = mu_cold

    # Rule 2: IF Temperature is Medium AND Humidity is Low THEN Fan Speed is Medium
    r2 = min(mu_med_temp, mu_low_hum)

    # Rule 3: IF Temperature is Medium AND Humidity is High THEN Fan Speed is High
    r3 = min(mu_med_temp, mu_high_hum)

    # Rule 4: IF Temperature is Hot THEN Fan Speed is High
    r4 = mu_hot

    # Aggregated Consequent Firing Strengths
    w_low = r1
    w_med = r2
    w_high = max(r3, r4)

    # 5. Defuzzification (Centroid of Aggregated Fuzzy Output Set)
    y_low = np.minimum(w_low, trimf(x_fan, low_fan_params))
    y_med = np.minimum(w_med, trimf(x_fan, med_fan_params))
    y_high = np.minimum(w_high, trimf(x_fan, high_fan_params))
    aggregated = np.maximum(y_low, np.maximum(y_med, y_high))

    if np.sum(aggregated) == 0:
        output_fan_speed = 50.0
    else:
        output_fan_speed = np.sum(x_fan * aggregated) / np.sum(aggregated)

    print("=" * 60)
    print("MAMDANI FUZZY LOGIC AIR CONDITIONER CONTROLLER")
    print("=" * 60)
    print(f"Input Temperature : {temperature_val:.2f} °C")
    print(f"Input Humidity    : {humidity_val:.2f} %")
    print("-" * 60)
    print(f"Rule Firing Strengths:")
    print(f"  R1 (Cold -> Low)           : {r1:.4f}")
    print(f"  R2 (Med & LowHum -> Med)   : {r2:.4f}")
    print(f"  R3 (Med & HighHum -> High) : {r3:.4f}")
    print(f"  R4 (Hot -> High)           : {r4:.4f}")
    print("-" * 60)
    print(f"Computed Fan Speed: {output_fan_speed:.2f} %")
    print("=" * 60)

    # 6. Graph Generation & Printing
    print("\nGenerating and saving Fuzzy Controller Membership & Output plots...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 10))

    # Temperature Plot
    ax1.plot(x_temp, trimf(x_temp, cold_params), 'b', label='Cold')
    ax1.plot(x_temp, trimf(x_temp, med_temp_params), 'g', label='Medium')
    ax1.plot(x_temp, trimf(x_temp, hot_params), 'r', label='Hot')
    ax1.axvline(temperature_val, color='k', linestyle='--', label=f'Input: {temperature_val}°C')
    ax1.set_title('Temperature Membership Functions')
    ax1.set_ylabel('Membership')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Humidity Plot
    ax2.plot(x_hum, trimf(x_hum, low_hum_params), 'b', label='Low')
    ax2.plot(x_hum, trimf(x_hum, high_hum_params), 'r', label='High')
    ax2.axvline(humidity_val, color='k', linestyle='--', label=f'Input: {humidity_val}%')
    ax2.set_title('Humidity Membership Functions')
    ax2.set_ylabel('Membership')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Fan Speed Output Plot
    ax3.plot(x_fan, trimf(x_fan, low_fan_params), 'b', linestyle=':', label='Low MF')
    ax3.plot(x_fan, trimf(x_fan, med_fan_params), 'g', linestyle=':', label='Medium MF')
    ax3.plot(x_fan, trimf(x_fan, high_fan_params), 'r', linestyle=':', label='High MF')
    ax3.fill_between(x_fan, 0, aggregated, color='orange', alpha=0.4, label='Aggregated Fuzzy Output')
    ax3.axvline(output_fan_speed, color='magenta', linewidth=2, linestyle='-', label=f'Crisp Output: {output_fan_speed:.1f}%')
    ax3.set_title('Fan Speed Consequent & Defuzzified Centroid Output')
    ax3.set_xlabel('Fan Speed (%)')
    ax3.set_ylabel('Membership')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_file = 'fuzzy_controller_plot.png'
    try:
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"[Graph Printed] Plot successfully saved to '{plot_file}'")
    except Exception:
        pass

    try:
        plt.show()
    except Exception:
        pass

    return output_fan_speed

if __name__ == "__main__":
    temp_in = float(input("Enter ambient temperature (0 to 50 °C, e.g., 28.5): "))
    hum_in = float(input("Enter relative humidity (0 to 100 %, e.g., 65.0): "))
    run_fuzzy_air_conditioner(temp_in, hum_in)
