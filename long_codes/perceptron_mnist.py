# =====================================================================
# SARDAR PATEL INSTITUTE OF TECHNOLOGY
# Experiment 4: Multiclass Perceptron for MNIST Classification
# Source: EXP4_AISC_2024300198 (2) (1).pdf
# =====================================================================

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def mnist_perceptron_sklearn():
    print("Fetching MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=60000, shuffle=False)

    print("Training Perceptron model...")
    clf = Perceptron(max_iter=20, tol=1e-3, random_state=42)
    clf.fit(X_train, y_train)

    print("Evaluating model and generating Confusion Matrix...")
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Confusion Matrix for MNIST Perceptron')
    plt.show()

if __name__ == "__main__":
    mnist_perceptron_sklearn()
