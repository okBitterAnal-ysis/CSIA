# =====================================================================
# BHARATIYA VIDYA BHAVAN'S SARDAR PATEL INSTITUTE OF TECHNOLOGY
# Experiment 5: Supervised Learning (Back Propagation Neural Network)
# Program 2: Student Pass/Fail Prediction
# Source: AISC_EXP5_2024300198 (2).pdf
# =====================================================================

import math
import random

# =========================================================
# Binary Classification - Student Pass/Fail Prediction
# Supervised Learning: Backpropagation Neural Network
# =========================================================

random.seed(42)

# -----------------------------
# 1. Create dummy dataset
# -----------------------------
# Features: study_hours (0-10), attendance (0-100%)
# Label: 1 = Pass, 0 = Fail
# Rule used to generate labels (very light noise, so the
# pattern is easy for the network to learn in a few epochs).

def generate_dataset(n=40):
    data = []
    for _ in range(n):
        study_hours = round(random.uniform(0, 10), 1)
        attendance = round(random.uniform(40, 100), 1)

        score = 0.6 * (study_hours / 10) + 0.4 * (attendance / 100)
        score += random.uniform(-0.02, 0.02)  # very light noise

        label = 1 if score >= 0.55 else 0
        data.append((study_hours, attendance, label))
    return data

dataset = generate_dataset(40)

# -----------------------------
# 2. Normalize features (helps sigmoid training a lot)
# -----------------------------
max_hours = 10.0
max_attendance = 100.0

X = [(h / max_hours, a / max_attendance) for h, a, l in dataset]
Y = [l for h, a, l in dataset]

# -----------------------------
# 3. Activation function
# -----------------------------

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

# -----------------------------
# 4. Initialize network
# -----------------------------
# Architecture: 2 inputs -> 4 hidden neurons -> 1 output neuron
random.seed(1)
n_input = 2
n_hidden = 4
n_output = 1

V = [
    [random.uniform(-1.0, 1.0) for _ in range(n_hidden)]
    for _ in range(n_input)
]
b_hidden = [random.uniform(-1.0, 1.0) for _ in range(n_hidden)]

W = [random.uniform(-1.0, 1.0) for _ in range(n_hidden)]
b_output = random.uniform(-1.0, 1.0)

learning_rate = 6.0  # higher LR compensates for very few epochs
epochs = 4

# -----------------------------
# 5. Training loop (Backpropagation)
# -----------------------------
for epoch in range(1, epochs + 1):
    total_error = 0.0

    for (x1, x2), target in zip(X, Y):
        # ---- Forward propagation ----
        z_in = [
            b_hidden[j] + x1 * V[0][j] + x2 * V[1][j]
            for j in range(n_hidden)
        ]
        z = [sigmoid(v) for v in z_in]

        y_in = b_output + sum(z[j] * W[j] for j in range(n_hidden))
        y = sigmoid(y_in)

        # ---- Back propagation ----
        error = target - y
        delta_output = error * sigmoid_derivative(y)

        delta_hidden = [
            delta_output * W[j] * sigmoid_derivative(z[j])
            for j in range(n_hidden)
        ]

        # ---- Weight updates ----
        for j in range(n_hidden):
            W[j] += learning_rate * delta_output * z[j]
        b_output += learning_rate * delta_output

        for j in range(n_hidden):
            V[0][j] += learning_rate * delta_hidden[j] * x1
            V[1][j] += learning_rate * delta_hidden[j] * x2
            b_hidden[j] += learning_rate * delta_hidden[j]

        total_error += error ** 2

    mse = total_error / len(X)
    rms = math.sqrt(mse)
    print(f"Epoch {epoch} | MSE = {mse:.6f} | RMS = {rms:.6f}")

# -----------------------------
# 6. Evaluate accuracy on training data
# -----------------------------

def predict(x1, x2):
    z_in = [
        b_hidden[j] + x1 * V[0][j] + x2 * V[1][j]
        for j in range(n_hidden)
    ]
    z = [sigmoid(v) for v in z_in]
    y_in = b_output + sum(z[j] * W[j] for j in range(n_hidden))
    y = sigmoid(y_in)
    return y

correct = 0
for (x1, x2), target in zip(X, Y):
    y = predict(x1, x2)
    pred_label = 1 if y >= 0.5 else 0
    if pred_label == target:
        correct += 1

accuracy = correct / len(X) * 100
print(f"\nTraining Accuracy: {accuracy:.2f}% ({correct}/{len(X)})")

# -----------------------------
# 7. Test on a few new (unseen) students
# -----------------------------
print("\nSample Predictions on New Students:")
print("StudyHrs | Attendance | PredictedProb | PredictedLabel")
test_students = [(2, 50), (8, 90), (5, 60), (9, 40), (1, 95), (6.5, 75)]
for hrs, att in test_students:
    y = predict(hrs / max_hours, att / max_attendance)
    label = "Pass" if y >= 0.5 else "Fail"
    print(f"{hrs:8.1f} | {att:10.1f} | {y:13.4f} | {label}")

# -----------------------------
# 8. Print full dataset used for training
# -----------------------------
print("\nDataset used for training (study_hours, attendance, label):")
for row in dataset:
    print(row)
