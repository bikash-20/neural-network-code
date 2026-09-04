"""
MNIST Classifier: PyTorch Convolutional Neural Network

This is our third project. We move from fully-connected networks to a
small Convolutional Neural Network and apply it to MNIST handwritten
digit classification.

Dataset: MNIST
- 70,000 grayscale digit images (60k train / 10k test)
- 28x28 pixels, single channel
- 10 classes (digits 0-9)

Architecture:
    Input (1, 28, 28) -> Conv(32) -> ReLU -> MaxPool
                     -> Conv(64) -> ReLU -> MaxPool
                     -> FC(128)   -> ReLU -> Dropout
                     -> FC(10)
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np

from utils import banner, data_path, models_path, plots_path, set_seed


set_seed(42)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def get_mnist_loaders(batch_size: int = 64):
    """
    Download (if needed) MNIST into ``data/mnist`` and return train/test loaders.

    The dataset is gitignored - it is regenerated on first run.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                                  # (H, W) uint8 -> (1, H, W) float32 in [0, 1]
        transforms.Normalize((0.1307,), (0.3081,)),              # MNIST mean/std
    ])

    root = data_path("mnist")
    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Training set: {len(train_ds)} samples")
    print(f"Test set:     {len(test_ds)} samples")
    print(f"Feature shape: 1 x 28 x 28 (single-channel 28x28 image)")
    print(f"Classes:       10 (digits 0-9)")

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CNN(nn.Module):
    """
    A small, well-behaved CNN for MNIST.

    Block 1: Conv(1->32, 3x3, padding=1) + ReLU + MaxPool(2x2)  -> (32, 14, 14)
    Block 2: Conv(32->64, 3x3, padding=1) + ReLU + MaxPool(2x2) -> (64,  7,  7)
    Head:    Flatten + Linear(64*7*7 -> 128) + ReLU + Dropout(0.25) + Linear(128 -> 10)

    Total parameters: ~422,000.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # (32, 28, 28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                          # (32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (64, 14, 14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                          # (64,  7,  7)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, criterion, optimizer, device, log_every: int = 100):
    """Run one training epoch and return (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        seen += inputs.size(0)

        if (batch_idx + 1) % log_every == 0:
            print(
                f"    batch {batch_idx + 1:4d}/{len(loader)} "
                f"| loss={total_loss / seen:.4f} | acc={correct / seen:.4f}"
            )

    return total_loss / seen, correct / seen


def evaluate(model, loader, criterion, device):
    """Evaluate on a loader; return (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            seen += inputs.size(0)
    return total_loss / seen, correct / seen


def train_model(
    model,
    train_loader,
    test_loader,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cpu",
):
    """Full training loop with history + periodic print, mirrors wine_quality.py."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [],  "test_acc": [],
    }

    print("\nTraining MNIST CNN...\n")
    print(f"{'Epoch':<6}{'Train Loss':<14}{'Train Acc':<12}{'Test Loss':<12}{'Test Acc':<12}")
    print("-" * 54)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"{epoch:<6}{train_loss:<14.4f}{train_acc:<12.4f}"
            f"{test_loss:<12.4f}{test_acc:<12.4f}"
        )

    return history


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_training_history(history, save_path: str) -> str:
    """Loss and accuracy curves, matching wine_quality.py plot style."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(history["test_loss"], label="Test Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("MNIST: Training & Test Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_acc"], label="Train Accuracy", linewidth=2)
    ax2.plot(history["test_acc"], label="Test Accuracy", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("MNIST: Training & Test Accuracy")
    ax2.set_ylim([0.9, 1.01])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    print(f"\nTraining history plot saved to: {save_path}")
    plt.close(fig)
    return save_path


def plot_sample_predictions(model, test_loader, device, save_path: str, n: int = 20) -> str:
    """A grid of n test images with predicted vs. true labels."""
    model.eval()
    images, preds, truths, confs = [], [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            top_p, top_c = probs.topk(1, dim=1)

            # Reverse the normalization for visualization
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
    fig.suptitle("Sample Predictions  (green=correct, red=wrong)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    print(f"Sample predictions plot saved to: {save_path}")
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    banner("MNIST CLASSIFIER: PYTORCH CNN")
    print("\nTask:           10-class digit classification")
    print("Framework:      PyTorch")
    print("Architecture:   Conv(32) -> Pool -> Conv(64) -> Pool -> FC(128) -> FC(10)\n")

    # Force CPU-friendly defaults unless CUDA is present
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Data
    print("Loading MNIST (downloads on first run)...\n")
    batch_size = 64
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

    # Model
    print("\nCreating neural network...")
    model = CNN(num_classes=10, dropout=0.25)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Train
    history = train_model(
        model,
        train_loader,
        test_loader,
        epochs=5,
        lr=1e-3,
        device=device,
    )

    # Final report
    banner("FINAL RESULTS")
    final_test_acc = history["test_acc"][-1]
    final_test_loss = history["test_loss"][-1]
    print(f"Final Test Accuracy: {final_test_acc * 100:.2f}%")
    print(f"Final Test Loss:     {final_test_loss:.4f}")

    # Plots
    plot_training_history(history, plots_path("mnist_training_history.png"))
    plot_sample_predictions(model, test_loader, device, plots_path("mnist_predictions.png"))

    # Save model
    model_path = models_path("mnist_cnn_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")

    banner("MNIST CLASSIFICATION COMPLETE")


if __name__ == "__main__":
    main()
