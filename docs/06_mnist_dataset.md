# The MNIST Dataset

> **Phase 3 — Dataset overview.** Companion to `src/mnist_classifier.py`.

## Why MNIST?

MNIST is the "hello world" of image classification. It is small, clean, and old enough that it has become a benchmark rather than a challenge — but it is the *right* place to introduce convolutions, because:

1. The data fits in memory (~50 MB).
2. Training finishes in minutes on a CPU.
3. Even tiny models exceed 99% test accuracy, so the *bottleneck* is the architecture, not the data.

## At a glance

| Property        | Value                                  |
|-----------------|----------------------------------------|
| Source          | Yann LeCun et al., NYU (1998)          |
| Total samples   | 70,000                                 |
| Train / Test    | 60,000 / 10,000 (fixed split)          |
| Image size      | 28 x 28 pixels                         |
| Channels        | 1 (grayscale)                          |
| Classes         | 10 (digits 0-9)                        |
| Pixel values    | uint8 in [0, 255]                      |
| Class balance   | Roughly uniform (slight skew toward 1) |

## Visualising the data

A single sample is a `(1, 28, 28)` tensor after `transforms.ToTensor()` and `(0.1307, 0.3081)` normalization. The mean and standard deviation come from the original MNIST training set and are the conventional values to subtract.

```python
mean, std = 0.1307, 0.3081
x = tensor * std + mean  # invert normalization for display
```

## What a CNN sees that an MLP doesn't

Each 28 x 28 image has **784** pixels. A vanilla MLP (Phase 2 style) would flatten them and learn a fully-connected weight matrix of shape `(784, 10)` for a single output layer — already 7,840 weights, before any hidden units. Worse:

- An MLP **does not know** that neighbouring pixels are related. It has to *learn* adjacency by assigning correlated weights to indices that happen to be neighbours. That is wasteful.
- An MLP **does not know** that patterns are translation-invariant. The same "loop of a 9" appearing 5 pixels to the right looks like a *different* input.

A convolutional layer fixes both: it ties nearby weights together (local connectivity) and uses the *same* filter across the whole image (weight sharing). See `docs/07_cnn_theory.md` for the formal treatment.

## Loading

We use torchvision — no manual download script needed:

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train = datasets.MNIST(root="data/mnist", train=True, download=True, transform=transform)
test  = datasets.MNIST(root="data/mnist", train=False, download=True, transform=transform)
```

`download=True` causes torchvision to fetch the four compressed files from `yann.lecun.com` on first run. The resulting directory `data/mnist/MNIST/raw/` is gitignored — it is regenerated on demand.

## Splits

We use the canonical split baked into MNIST:

- **60,000** training images (we keep all of them)
- **10,000** test images (the official held-out set)

We do **not** carve out a validation set in this script. The model is small enough that 5 epochs is a stable regime, and we treat the test set as our final-evaluation set as is standard for Phase 3 tutorials.

## Class label meanings

There are no "real-world" labels to interpret — the class index *is* the digit, `0..9`. Confusion pairs tend to be:

| True | Often confused with |
|------|---------------------|
| 7    | 9                   |
| 4    | 9                   |
| 5    | 3, 6, 8             |
| 8    | 0, 3, 5, 6           |

A small fraction of test errors will always be visually ambiguous even to humans.
