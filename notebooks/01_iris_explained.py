"""
Marimo notebook - Iris classifier walkthrough.

Run with:
    uv run marimo edit 01_iris_explained.py
    # or
    pipx run marimo edit 01_iris_explained.py

This notebook narrates ``src/iris_classifier.py``: it explains the math, shows
the data, and trains the same 2-layer NumPy network so you can poke at every
step interactively.

Marimo is **not** in ``requirements.txt`` — install it with ``uv`` or ``pipx``
when you want to use these notebooks. The ``src/*.py`` scripts are the
canonical runnable interface; this notebook is the narrated walkthrough.
"""

import marimo

__generated_with__ = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Phase 1 — Iris Classifier (NumPy from scratch)

        We build a **2-layer neural network** using only NumPy and train it
        to classify the three species of *Iris* flowers from four physical
        measurements (sepal length, sepal width, petal length, petal width).

        ## Goal

        Reach at least **96% test accuracy** on a held-out 20% of the data.

        ## What you'll see in this notebook

        1. The math of forward propagation, softmax, and cross-entropy.
        2. The Iris dataset loaded and preprocessed.
        3. A NumPy-only training loop with the same hyperparameters as
           ``src/iris_classifier.py``.
        4. Final test accuracy, per-class breakdown, and a training-curve plot.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. Mathematical background

        A 2-layer network computes:

        $$z^{(1)} = W^{(1)} x + b^{(1)}, \quad a^{(1)} = \mathrm{ReLU}(z^{(1)})$$
        $$z^{(2)} = W^{(2)} a^{(1)} + b^{(2)}, \quad \hat{y} = \mathrm{softmax}(z^{(2)})$$

        with **He initialization** for $W^{(1)}$:

        $$W^{(1)}_{ij} \sim \mathcal{N}\!\Bigl(0, \;\sqrt{2 / n_\text{in}}\Bigr)$$

        The cross-entropy loss for one sample is

        $$\mathcal{L} = -\sum_{k=1}^{K} y_k \log \hat{y}_k$$

        averaged over $N$ samples. The gradient with respect to the output
        pre-activations simplifies beautifully:

        $$\frac{\partial \mathcal{L}}{\partial z^{(2)}} = \hat{y} - y$$

        which makes the chain rule for backprop very compact.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"## 2. Load and prepare the data")
    return


@app.cell
def _():
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    iris = load_iris()
    feature_names = iris.feature_names
    class_names = list(iris.target_names)

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # (features, samples) convention used by src/iris_classifier.py
    X_train_t = X_train.T
    X_test_t = X_test.T
    Y_train = np.eye(3)[y_train].T
    Y_test = np.eye(3)[y_test].T

    print(f"Training samples: {X_train_t.shape[1]}")
    print(f"Test samples:     {X_test_t.shape[1]}")
    print(f"Features:         {X_train_t.shape[0]} ({', '.join(feature_names)})")
    print(f"Classes:          {class_names}")

    return X_test_t, Y_train, class_names, np, y_test


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Quick look

        After ``StandardScaler`` each feature has zero mean and unit variance
        on the training set. The same ``scaler.transform`` is applied to the
        test set using the *training* statistics — never refit on test.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"## 3. Define the NumPy network")
    return


@app.cell
def _(np):
    class NeuralNetwork:
        """A 2-layer NN: 4 -> 10 (ReLU) -> 3 (Softmax)."""

        def __init__(self, input_size=4, hidden_size=10, output_size=3, learning_rate=0.1):
            self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
            self.b1 = np.zeros((hidden_size, 1))
            self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
            self.b2 = np.zeros((output_size, 1))
            self.learning_rate = learning_rate
            self.loss_history = []
            self.accuracy_history = []

        @staticmethod
        def relu(z):
            return np.maximum(0, z)

        @staticmethod
        def relu_derivative(z):
            return (z > 0).astype(float)

        @staticmethod
        def softmax(z):
            z_shifted = z - np.max(z, axis=0, keepdims=True)
            exp_z = np.exp(z_shifted)
            return exp_z / np.sum(exp_z, axis=0, keepdims=True)

        @staticmethod
        def cross_entropy(y_pred, y_true):
            n = y_true.shape[1]
            y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return -np.sum(y_true * np.log(y_pred_clipped)) / n

        def forward(self, X):
            z1 = self.W1 @ X + self.b1
            a1 = self.relu(z1)
            z2 = self.W2 @ a1 + self.b2
            a2 = self.softmax(z2)
            return z1, a1, z2, a2

        def backward(self, X, Y, z1, a1, z2, a2):
            n = X.shape[1]
            dz2 = a2 - Y
            dW2 = dz2 @ a1.T / n
            db2 = np.sum(dz2, axis=1, keepdims=True) / n
            da1 = self.W2.T @ dz2
            dz1 = da1 * self.relu_derivative(z1)
            dW1 = dz1 @ X.T / n
            db1 = np.sum(dz1, axis=1, keepdims=True) / n
            return dW1, db1, dW2, db2

        def update(self, grads):
            dW1, db1, dW2, db2 = grads
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

        def train(self, X_train, Y_train, epochs=1500):
            for _ in range(epochs):
                z1, a1, z2, a2 = self.forward(X_train)
                loss = self.cross_entropy(a2, Y_train)
                preds = np.argmax(a2, axis=0)
                labels = np.argmax(Y_train, axis=0)
                acc = np.mean(preds == labels)
                self.loss_history.append(loss)
                self.accuracy_history.append(acc)
                grads = self.backward(X_train, Y_train, z1, a1, z2, a2)
                self.update(grads)
            return self

        def predict(self, X):
            _, _, _, a2 = self.forward(X)
            return np.argmax(a2, axis=0)

    return (NeuralNetwork,)


@app.cell
def _(mo):
    mo.md(r"## 4. Train (1500 epochs, batch GD, lr=0.1)")
    return


@app.cell
def _(NeuralNetwork, X_train_t, Y_train, np):
    np.random.seed(42)
    iris_nn = NeuralNetwork(input_size=4, hidden_size=10, output_size=3, learning_rate=0.1)
    iris_nn.train(X_train_t, Y_train, epochs=1500)
    print(f"Final loss:     {iris_nn.loss_history[-1]:.4f}")
    print(f"Final accuracy: {iris_nn.accuracy_history[-1]:.4f}")
    return (iris_nn,)


@app.cell
def _(mo):
    mo.md(r"## 5. Evaluate on the test set")
    return


@app.cell
def _(X_test_t, class_names, iris_nn, np, y_test):
    preds = iris_nn.predict(X_test_t)
    test_acc = np.mean(preds == y_test)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    print("\nPer-class performance:")
    for i, name in enumerate(class_names):
        mask = y_test == i
        if mask.any():
            cls_acc = np.mean(preds[mask] == y_test[mask])
            print(f"  {name:12s}: {cls_acc * 100:.2f}%")
    return test_acc,


@app.cell
def _(iris_nn, mo):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(iris_nn.loss_history, color="#e74c3c", linewidth=2)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy")
    ax1.grid(True, alpha=0.3)

    ax2.plot(iris_nn.accuracy_history, color="#27ae60", linewidth=2)
    ax2.set_title("Training Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig

    mo.md(r"### Loss and accuracy over training")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### End of notebook

        **What you just did**

        - Built a 2-layer neural network using *only* NumPy.
        - Verified He initialization, ReLU, softmax, and cross-entropy math.
        - Reached $\geq 96\%$ test accuracy on Iris in 1500 epochs of batch
          gradient descent.

        **Where to go next**

        - Phase 2: ``notebooks/02_wine_quality_explained.py`` — same idea,
          PyTorch, larger tabular dataset.
        - Phase 3: ``notebooks/03_mnist_explained.py`` — swap MLP for a CNN.
        """
    )
    return


if __name__ == "__main__":
    app.run()
