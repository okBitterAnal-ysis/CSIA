#!/usr/bin/env python3
# Experiment 9: Fuzzy Logic Controller (Mamdani Inference)
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# 1. Define Universes of Discourse
temp = ctrl.Antecedent(np.arange(0, 51, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
fan = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')

# 2. Membership Functions
temp['Cold'] = fuzz.trimf(temp.universe, [0, 0, 20])
temp['Medium'] = fuzz.trimf(temp.universe, [15, 25, 35])
temp['Hot'] = fuzz.trimf(temp.universe, [30, 50, 50])

humidity['Low'] = fuzz.trimf(humidity.universe, [0, 0, 50])
humidity['High'] = fuzz.trimf(humidity.universe, [50, 100, 100])

fan['Low'] = fuzz.trimf(fan.universe, [0, 0, 50])
fan['Medium'] = fuzz.trimf(fan.universe, [25, 50, 75])
fan['High'] = fuzz.trimf(fan.universe, [50, 100, 100])

# 3. Fuzzy Rules
r1 = ctrl.Rule(temp['Cold'], fan['Low'])
r2 = ctrl.Rule(temp['Medium'] & humidity['Low'], fan['Medium'])
r3 = ctrl.Rule(temp['Medium'] & humidity['High'], fan['High'])
r4 = ctrl.Rule(temp['Hot'], fan['High'])

# 4. Simulation Engine
system = ctrl.ControlSystem([r1, r2, r3, r4])
sim = ctrl.ControlSystemSimulation(system)

# User Inputs
t = float(input("Enter temperature in °C (0 to 50, e.g. 24): "))
h = float(input("Enter humidity in % (0 to 100, e.g. 60): "))

sim.input['temperature'] = t
sim.input['humidity'] = h
sim.compute()

print(f"\n--- Fuzzy Controller Output ---")
print(f"Temperature: {t} °C")
print(f"Humidity   : {h} %")
print(f"Fan Speed  : {sim.output['fan_speed']:.2f} %")
