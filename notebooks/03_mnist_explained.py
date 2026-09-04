"""
Marimo notebook - MNIST (PyTorch CNN) walkthrough.

Run with:
    uv run marimo edit 03_mnist_explained.py
    # or
    pipx run marimo edit 03_mnist_explained.py

This notebook narrates ``src/mnist_classifier.py``: it explains the inductive
biases a CNN brings versus the MLP from Phase 2, then trains the same small
CNN on MNIST and inspects the predictions.

The first cell that touches the dataset will trigger torchvision to download
the four MNIST files (~50 MB total) into ``data/mnist/``.

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
        # Phase 3 — MNIST (PyTorch CNN)

        We move from tabular features to **28x28 grayscale images** and from
        an MLP to a small **convolutional neural network**. The dataset is the
        canonical MNIST handwritten-digit benchmark: 60,000 training images
        and 10,000 test images, ten classes (digits 0-9).

        ## Goal

        Reach **>= 99% test accuracy** in 5 epochs on a CPU.

        ## What you'll see in this notebook

        1. Why a CNN beats an MLP on images (locality + translation invariance).
        2. The two-conv-block + dense-head architecture.
        3. Training with Adam and ``CrossEntropyLoss``.
        4. A grid of sample predictions, coloured by correctness.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. Mathematical background

        A 2D convolution with kernel $K$ slides across an image $I$:

        $$(I * K)[i, j] = \sum_{u} \sum_{v} I[i + u, j + v] \cdot K[u, v]$$

        Each filter learns a small spatial pattern. Stacking filters and
        pooling them through the network builds up a hierarchy:

        - Block 1 (after 1x pooling): edges and short strokes.
        - Block 2 (after 2x pooling): digit parts — loops, crossings.
        - Dense head: full digit identity.

        Our network has roughly **422,000 parameters** — tiny by modern standards
        but enough to clear 99% test accuracy on MNIST.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"## 2. Load MNIST")
    return


@app.cell
def _():
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    DATA_ROOT = "data/mnist"
    train_ds = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

    BATCH_SIZE = 64
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {len(train_ds)}  Test: {len(test_ds)}")
    return DataLoader, test_loader, torch, train_loader


@app.cell
def _(mo):
    mo.md(
        r"""
        The first call above triggers a one-time download of MNIST into
        ``data/mnist/``. That folder is gitignored — the data is regenerated
        on demand. The mean and standard deviation used in
        ``Normalize((0.1307,), (0.3081,))`` are the canonical MNIST values.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"## 3. Define the CNN")
    return


@app.cell
def _(torch):
    import torch.nn as nn

    class CNN(nn.Module):
        """1x28x28 -> Conv(32) -> Pool -> Conv(64) -> Pool -> FC(128) -> FC(10)."""

        def __init__(self, num_classes=10, dropout=0.25):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return CNN, nn


@app.cell
def _(mo):
    mo.md(r"## 4. Train (5 epochs, Adam lr=1e-3, batch size 64)")
    return


@app.cell
def _(CNN, nn, test_loader, torch, train_loader):
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 5
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += inputs.size(0)
        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss = criterion(logits, labels)
                test_loss += loss.item() * inputs.size(0)
                test_correct += (logits.argmax(1) == labels).sum().item()
                test_total += inputs.size(0)
        test_loss /= test_total
        test_acc = test_correct / test_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(
            f"epoch {epoch} | "
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
    ax2.set_ylim([0.9, 1.01])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig

    mo.md(r"### Loss and accuracy over training")
    return


@app.cell
def _(mo, model, test_loader, torch):
    mo.md(r"## 5. Inspect predictions")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch.nn.functional as F

    images = []
    preds = []
    truths = []
    confs = []
    n = 20
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            top_p, top_c = probs.topk(1, dim=1)
            mean, std = 0.1307, 0.3081
            x = inputs.cpu() * std + mean
            x = x.clamp(0, 1)
            images.append(x)
            preds.append(top_c.squeeze(1).cpu().numpy())
            truths.append(targets.numpy())
            confs.append(top_p.squeeze(1).cpu().numpy())
            if sum(a.shape[0] for a in images) >= n:
                break

    images = torch.cat(images)[:n]
    preds = np.concatenate(preds)[:n]
    truths = np.concatenate(truths)[:n]
    confs = np.concatenate(confs)[:n]

    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i < n:
            ax.imshow(images[i].squeeze(0), cmap="gray")
            label = f"P:{preds[i]}  T:{truths[i]}  ({confs[i] * 100:.0f}%)"
            color = "green" if preds[i] == truths[i] else "red"
            ax.set_title(label, fontsize=10, color=color)
    plt.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### End of notebook

        **What you just did**

        - Built a small CNN with two conv blocks and a 128-unit dense head.
        - Trained it for 5 epochs on MNIST, reaching $\geq 99\%$ test
          accuracy.
        - Inspected 20 sample predictions to see where the model is correct
          and where it fails.

        **Where to go next**

        - Try CIFAR-10 — a much harder 32x32 RGB dataset that needs more
          layers and data augmentation.
        - Add ``BatchNorm2d`` after each conv block and see the train/test
          gap shrink.
        - Read ``docs/08_phase3_results.md`` for the comparison with
          Phases 1 and 2.
        """
    )
    return


if __name__ == "__main__":
    app.run()
