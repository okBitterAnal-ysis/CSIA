# =====================================================================
# BHARATIYA VIDYA BHAVAN'S SARDAR PATEL INSTITUTE OF TECHNOLOGY
# Experiment 5: Supervised Learning (Back Propagation Neural Network)
# Program 2: Student Pass/Fail Prediction with Loss Curves & Graph
# Source: AISC_EXP5_2024300198 (2).pdf
# =====================================================================

import math
import random
import matplotlib.pyplot as plt

random.seed(42)

# -----------------------------
# 1. Create dummy dataset
# -----------------------------
def generate_dataset(n=40):
    data = []
    for _ in range(n):
        study_hours = round(random.uniform(0, 10), 1)
        attendance = round(random.uniform(40, 100), 1)
        score = 0.6 * (study_hours / 10) + 0.4 * (attendance / 100)
        score += random.uniform(-0.02, 0.02)
        label = 1 if score >= 0.55 else 0
        data.append((study_hours, attendance, label))
    return data

dataset = generate_dataset(40)

max_hours = 10.0
max_attendance = 100.0

X = [(h / max_hours, a / max_attendance) for h, a, l in dataset]
Y = [l for h, a, l in dataset]

# -----------------------------
# 2. Activation functions
# -----------------------------
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

# -----------------------------
# 3. Initialize network (2-4-1)
# -----------------------------
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

learning_rate = 6.0
epochs = 4
mse_history = []
rms_history = []

# -----------------------------
# 4. Training loop (Backpropagation)
# -----------------------------
print("=" * 60)
print("TRAINING BACKPROPAGATION NETWORK (STUDENT PASS/FAIL)")
print("=" * 60)

for epoch in range(1, epochs + 1):
    total_error = 0.0

    for (x1, x2), target in zip(X, Y):
        # Forward pass
        z_in = [
            b_hidden[j] + x1 * V[0][j] + x2 * V[1][j]
            for j in range(n_hidden)
        ]
        z = [sigmoid(v) for v in z_in]

        y_in = b_output + sum(z[j] * W[j] for j in range(n_hidden))
        y = sigmoid(y_in)

        # Backward pass
        error = target - y
        delta_output = error * sigmoid_derivative(y)

        delta_hidden = [
            delta_output * W[j] * sigmoid_derivative(z[j])
            for j in range(n_hidden)
        ]

        # Update weights
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
    mse_history.append(mse)
    rms_history.append(rms)
    print(f"Epoch {epoch} | MSE = {mse:.6f} | RMS = {rms:.6f}")

# -----------------------------
# 5. Evaluate accuracy
# -----------------------------
def predict(x1, x2):
    z_in = [
        b_hidden[j] + x1 * V[0][j] + x2 * V[1][j]
        for j in range(n_hidden)
    ]
    z = [sigmoid(v) for v in z_in]
    y_in = b_output + sum(z[j] * W[j] for j in range(n_hidden))
    return sigmoid(y_in)

correct = 0
for (x1, x2), target in zip(X, Y):
    y = predict(x1, x2)
    pred_label = 1 if y >= 0.5 else 0
    if pred_label == target:
        correct += 1

accuracy = correct / len(X) * 100
print(f"\nTraining Accuracy: {accuracy:.2f}% ({correct}/{len(X)})")

# -----------------------------
# 6. Test on new students
# -----------------------------
print("\nSample Predictions on New Students:")
print("StudyHrs | Attendance | PredictedProb | PredictedLabel")
test_students = [(2, 50), (8, 90), (5, 60), (9, 40), (1, 95), (6.5, 75)]
for hrs, att in test_students:
    y = predict(hrs / max_hours, att / max_attendance)
    label = "Pass" if y >= 0.5 else "Fail"
    print(f"{hrs:8.1f} | {att:10.1f} | {y:13.4f} | {label}")

# -----------------------------
# 7. Graph Generation & Printing
# -----------------------------
print("\nGenerating and displaying Training Loss curves...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Error Curves
epoch_axis = list(range(1, epochs + 1))
ax1.plot(epoch_axis, mse_history, marker='o', color='red', label='MSE (Mean Squared Error)')
ax1.plot(epoch_axis, rms_history, marker='s', color='blue', label='RMS (Root Mean Square)')
ax1.set_title('Training Error vs Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss / Error')
ax1.set_xticks(epoch_axis)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Plot 2: Dataset Scatter Plot (Study Hours vs Attendance)
pass_hours = [h for h, a, l in dataset if l == 1]
pass_att = [a for h, a, l in dataset if l == 1]
fail_hours = [h for h, a, l in dataset if l == 0]
fail_att = [a for h, a, l in dataset if l == 0]

ax2.scatter(pass_hours, pass_att, color='green', label='Pass (1)', s=60, edgecolors='k')
ax2.scatter(fail_hours, fail_att, color='red', marker='x', label='Fail (0)', s=60)
ax2.set_title('Student Classification Distribution')
ax2.set_xlabel('Study Hours (0 - 10)')
ax2.set_ylabel('Attendance % (40 - 100)')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plot_file = 'backpropagation_student_loss_plot.png'
try:
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"[Graph Printed] Plot successfully saved to '{plot_file}'")
except Exception:
    pass

try:
    plt.show()
except Exception:
    pass
