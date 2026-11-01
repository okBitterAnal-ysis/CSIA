# =====================================================================
# BHARATIYA VIDYA BHAVAN'S SARDAR PATEL INSTITUTE OF TECHNOLOGY
# Experiment 9: Fuzzy Logic Controller (Mamdani Inference System)
# =====================================================================

import numpy as np
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

    return output_fan_speed

if __name__ == "__main__":
    temp_in = float(input("Enter ambient temperature (0 to 50 °C, e.g., 28.5): "))
    hum_in = float(input("Enter relative humidity (0 to 100 %, e.g., 65.0): "))
    run_fuzzy_air_conditioner(temp_in, hum_in)
