#!/usr/bin/env python3
# =====================================================================
# BHARATIYA VIDYA BHAVAN'S SARDAR PATEL INSTITUTE OF TECHNOLOGY
# Experiment 9: Fuzzy Logic Controller (Mamdani Inference System) with Graphs
# =====================================================================

import numpy as np
import os
import matplotlib
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

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
    temperature = ctrl.Antecedent(np.arange(0, 51, 1), 'temperature')
    humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
    fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')

    # 2. Membership Functions
    temperature['Cold'] = fuzz.trimf(temperature.universe, [0, 0, 20])
    temperature['Medium'] = fuzz.trimf(temperature.universe, [15, 25, 35])
    temperature['Hot'] = fuzz.trimf(temperature.universe, [30, 50, 50])

    humidity['Low'] = fuzz.trimf(humidity.universe, [0, 0, 50])
    humidity['High'] = fuzz.trimf(humidity.universe, [50, 100, 100])

    fan_speed['Low'] = fuzz.trimf(fan_speed.universe, [0, 0, 50])
    fan_speed['Medium'] = fuzz.trimf(fan_speed.universe, [25, 50, 75])
    fan_speed['High'] = fuzz.trimf(fan_speed.universe, [50, 100, 100])

    # 3. Rule Base Formulation
    rule1 = ctrl.Rule(temperature['Cold'], fan_speed['Low'])
    rule2 = ctrl.Rule(temperature['Medium'] & humidity['Low'], fan_speed['Medium'])
    rule3 = ctrl.Rule(temperature['Medium'] & humidity['High'], fan_speed['High'])
    rule4 = ctrl.Rule(temperature['Hot'], fan_speed['High'])

    # 4. Control System Setup & Execution
    control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
    controller_sim = ctrl.ControlSystemSimulation(control_system)

    controller_sim.input['temperature'] = temperature_val
    controller_sim.input['humidity'] = humidity_val
    controller_sim.compute()

    output_fan_speed = controller_sim.output['fan_speed']

    print("=" * 60)
    print("MAMDANI FUZZY LOGIC AIR CONDITIONER CONTROLLER")
    print("=" * 60)
    print(f"Input Temperature : {temperature_val:.2f} °C")
    print(f"Input Humidity    : {humidity_val:.2f} %")
    print("-" * 60)
    print(f"Computed Fan Speed: {output_fan_speed:.2f} %")
    print("=" * 60)

    # 5. Graph Generation & Printing
    print("\nGenerating and displaying Fuzzy Controller Membership & Output plots...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 10))

    # Plot Temperature
    x_t = temperature.universe
    ax1.plot(x_t, fuzz.trimf(x_t, [0, 0, 20]), 'b', label='Cold')
    ax1.plot(x_t, fuzz.trimf(x_t, [15, 25, 35]), 'g', label='Medium')
    ax1.plot(x_t, fuzz.trimf(x_t, [30, 50, 50]), 'r', label='Hot')
    ax1.axvline(temperature_val, color='k', linestyle='--', label=f'Input: {temperature_val}°C')
    ax1.set_title('Temperature Membership Functions')
    ax1.set_ylabel('Membership')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot Humidity
    x_h = humidity.universe
    ax2.plot(x_h, fuzz.trimf(x_h, [0, 0, 50]), 'b', label='Low')
    ax2.plot(x_h, fuzz.trimf(x_h, [50, 100, 100]), 'r', label='High')
    ax2.axvline(humidity_val, color='k', linestyle='--', label=f'Input: {humidity_val}%')
    ax2.set_title('Humidity Membership Functions')
    ax2.set_ylabel('Membership')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Plot Fan Speed Output
    x_f = fan_speed.universe
    ax3.plot(x_f, fuzz.trimf(x_f, [0, 0, 50]), 'b', label='Low')
    ax3.plot(x_f, fuzz.trimf(x_f, [25, 50, 75]), 'g', label='Medium')
    ax3.plot(x_f, fuzz.trimf(x_f, [50, 100, 100]), 'r', label='High')
    ax3.axvline(output_fan_speed, color='magenta', linewidth=2, linestyle='-', label=f'Crisp Output: {output_fan_speed:.1f}%')
    ax3.set_title('Fan Speed Consequent & Defuzzified Output')
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
