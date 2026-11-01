#!/usr/bin/env python3
# =====================================================================
# SARDAR PATEL INSTITUTE OF TECHNOLOGY
# Experiment 4: Multiclass Perceptron for Handwritten Digit Classification
# Source: EXP4_AISC_2024300198 (2) (1).pdf
# =====================================================================

import os
import matplotlib
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def mnist_perceptron_sklearn():
    print("=" * 65)
    print("MULTICLASS PERCEPTRON DIGIT CLASSIFIER")
    print("=" * 65)

    # 1. Load Dataset (MNIST from OpenML with offline fallback)
    try:
        from sklearn.datasets import fetch_openml
        print("Attempting to fetch MNIST dataset from OpenML...")
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
        X = X / 255.0
        dataset_name = "MNIST (784-Features)"
    except Exception as e:
        print(f"Notice: OpenML download unavailable ({e}).")
        print("Using built-in scikit-learn Handwritten Digits dataset (8x8)...")
        from sklearn.datasets import load_digits
        digits = load_digits()
        X, y = digits.data / 16.0, digits.target.astype(str)
        dataset_name = "Handwritten Digits (8x8)"

    # 2. Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    print(f"Dataset      : {dataset_name}")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples : {len(X_test)}")

    # 3. Model Training
    print("\nTraining Multiclass Perceptron model...")
    clf = Perceptron(max_iter=50, tol=1e-3, random_state=42)
    clf.fit(X_train, y_train)

    # 4. Evaluation
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test) * 100
    print(f"Accuracy     : {accuracy:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    print("\n--- Confusion Matrix ---")
    print(cm)

    # 5. Graph Generation & Printing
    print("\nGenerating and displaying Confusion Matrix plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix for {dataset_name} Perceptron (Accuracy: {accuracy:.1f}%)')

    plot_file = 'perceptron_mnist_confusion_matrix.png'
    try:
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"[Graph Printed] Plot successfully saved to '{plot_file}'")
    except Exception:
        pass

    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    mnist_perceptron_sklearn()
