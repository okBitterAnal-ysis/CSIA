#!/usr/bin/env python3
import os
import matplotlib
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def mnist_perceptron_sklearn():
    print("Loading MNIST / Digits dataset...")
    try:
        X, y = fetch_openml(
            'mnist_784', version=1,
            return_X_y=True, as_frame=False, parser='liac-arff'
        )
        X = X / 255.0
        train_sz = 60000
    except Exception:
        print("Using local digits dataset fallback...")
        from sklearn.datasets import load_digits
        digits = load_digits()
        X, y = digits.data / 16.0, digits.target.astype(str)
        train_sz = int(len(X) * 0.8)

    print("Dataset loaded!")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_sz, shuffle=False
    )

    print("Training Perceptron model...")
    clf = Perceptron(max_iter=20, tol=1e-3, random_state=42)
    clf.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=clf.classes_
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Confusion Matrix for MNIST Perceptron')
    plot_file = 'mnist_confusion_matrix.png'
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
