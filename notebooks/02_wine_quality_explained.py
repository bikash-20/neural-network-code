"""
Marimo notebook - Wine Quality (PyTorch MLP) walkthrough.

Run with:
    uv run marimo edit 02_wine_quality_explained.py
    # or
    pipx run marimo edit 02_wine_quality_explained.py

This notebook narrates ``src/wine_quality.py``: it explains how we reframe a
regression target (quality score 0-10) as a binary classification task and
how a small MLP plus dropout handles the tabular features.

Marimo is **not** in ``requirements.txt`` — install it with ``uv`` or ``pipx``
when you want to use these notebooks.
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
        # Phase 2 — Wine Quality (PyTorch MLP)

        We move from NumPy to **PyTorch** and tackle a regression problem by
        reframing it as a binary classification task: a wine is "good" if its
        quality score is $\geq 7$ and "not good" otherwise.

        ## Goal

        Reach at least **80% test accuracy** on the held-out 20% of the
        UCI white-wine dataset (4,898 wines, 11 chemical features).

        ## What you'll see in this notebook

        1. The dataset, the binarisation, and the class imbalance.
        2. The 64 -> 32 MLP architecture with Dropout.
        3. A PyTorch training loop with ``CrossEntropyLoss`` and Adam.
        4. Final test accuracy and a training-curve plot.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. Mathematical background

        A 3-layer MLP with dropout:

        $$h^{(1)} = \mathrm{Dropout}\bigl(\mathrm{ReLU}(W^{(1)} x + b^{(1)})\bigr)$$
        $$h^{(2)} = \mathrm{Dropout}\bigl(\mathrm{ReLU}(W^{(2)} h^{(1)} + b^{(2)})\bigr)$$
        $$\hat{y} = \mathrm{softmax}(W^{(3)} h^{(2)} + b^{(3)})$$

        with $W^{(1)} \in \mathbb{R}^{64 \times 11}$,
        $W^{(2)} \in \mathbb{R}^{32 \times 64}$,
        $W^{(3)} \in \mathbb{R}^{2 \times 32}$. Dropout zeros each activation
        with probability $0.2$ during training, which we disable at test time.

        Loss: categorical cross-entropy.
        Optimizer: Adam, learning rate $10^{-3}$.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"## 2. Load and binarise the data")
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import urllib.request
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    csv_path = "/tmp/winequality-white.csv"
    try:
        df = pd.read_csv(csv_path, sep=";")
    except FileNotFoundError:
        urllib.request.urlretrieve(url, csv_path)
        df = pd.read_csv(csv_path, sep=";")

    X = df.drop("quality", axis=1).values
    y = df["quality"].values
    y_binary = (y >= 7).astype(int)

    print(f"Shape: {df.shape}")
    print(f"Class distribution: {np.bincount(y_binary)}  (0=not good, 1=good)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")
    print(f"Feature dimension: {X_train.shape[1]}")
    return X_test, X_train, np, y_test, y_train


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Why binarise?

        The raw quality score is a discrete integer in $[0, 10]$, skewed toward
        the middle. Treating it as regression is possible but binarisation
        gives a clean $K = 2$ classification problem that demonstrates a
        realistic real-world pipeline.

        The class distribution is roughly 70 / 30, which is mild imbalance —
        accuracy is still a reasonable headline metric but precision / recall
        on the minority class would tell a richer story.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"## 3. Define the MLP")
    return


@app.cell
def _():
    import torch
    import torch.nn as nn

    class WineNet(nn.Module):
        """11 -> 64 (ReLU + Dropout) -> 32 (ReLU + Dropout) -> 2."""

        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(11, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 2),
            )

        def forward(self, x):
            return self.network(x)

    return WineNet, torch


@app.cell
def _(mo):
    mo.md(r"## 4. Train (100 epochs, batch size 32, Adam lr=1e-3)")
    return


@app.cell
def _(WineNet, X_test, X_train, torch, y_test, y_train):
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False
    )

    model = WineNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    EPOCHS = 100
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += inputs.size(0)
        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                test_correct += (outputs.argmax(1) == labels).sum().item()
                test_total += inputs.size(0)
        test_loss /= test_total
        test_acc = test_correct / test_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"epoch {epoch:3d} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
            )

    print(f"\nFinal Test Accuracy: {history['test_acc'][-1] * 100:.2f}%")
    return history, model


@app.cell
def _(history, mo):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["test_loss"], label="test")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_acc"], label="train")
    ax2.plot(history["test_acc"], label="test")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
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

        - Loaded the UCI white-wine dataset and binarised the target.
        - Built a 64 -> 32 MLP with two Dropout layers in PyTorch.
        - Trained for 100 epochs with Adam and ``CrossEntropyLoss``,
          reaching the mid-80s% test accuracy range.

        **Where to go next**

        - Phase 3: ``notebooks/03_mnist_explained.py`` — swap tabular features
          for 28x28 images and an MLP for a small CNN.
        """
    )
    return


if __name__ == "__main__":
    app.run()
