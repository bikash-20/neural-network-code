# Convolutional Neural Networks: Theory

> **Phase 3 — Theory.** Companion to `src/mnist_classifier.py`.

## Why convolutions?

A fully-connected layer treats the input as a flat vector. For an image this throws away two pieces of structure:

1. **Locality** — nearby pixels are correlated (an edge, a stroke, a corner).
2. **Translation invariance** — the same pattern appearing in different places is the same pattern.

A convolutional layer encodes *both* assumptions directly into the architecture.

## The 2D convolution (informal)

A **filter** (or **kernel**) is a small matrix, e.g. a 3 x 3 patch of learned weights. We slide it across the input image, at every position computing the dot product between the filter and the overlapping patch:

$$(I * K)[i, j] = \sum_{u=-1}^{1}\sum_{v=-1}^{1} I[i+u, j+v] \cdot K[u, v]$$

The output is a **feature map**: one channel per filter, same spatial size if we use padding.

## Padding and stride

- **Padding = 1** (with a 3 x 3 kernel) preserves spatial size: $(H, W) \to (H, W)$.
- **Stride = 2** halves the spatial size: downsamples by a factor of 2.
- **Max pooling** with a 2 x 2 window does the same job in a parameter-free way and is what we use here.

## Receptive fields

A unit in the second conv block "sees" a 5 x 5 patch of the original image (because each of its 3 x 3 inputs already saw a 3 x 3 patch, and those overlap). After two pooling stages, deeper layers have a *larger* receptive field — that is how CNNs build up hierarchical features:

- Block 1: edges and strokes
- Block 2: digit parts (loops, crossings)
- Classifier: full digit identity

## Number of parameters

For our `Conv(1 -> 32, 3 x 3)` layer:

$$W \in \mathbb{R}^{32 \times 1 \times 3 \times 3},\quad |W| = 288$$

For an equivalent dense layer mapping a flattened `(1, 28, 28)` image to a hidden width of 32:

$$W \in \mathbb{R}^{32 \times 784},\quad |W| = 25{,}088$$

Convolutions are **~87 x more parameter-efficient** for this local feature, even before pooling.

## The architecture in `src/mnist_classifier.py`

```
Input       : (N, 1, 28, 28)
Conv 3x3, 32: (N, 32, 28, 28)   # padding=1
ReLU
MaxPool 2x2 : (N, 32, 14, 14)
Conv 3x3, 64: (N, 64, 14, 14)   # padding=1
ReLU
MaxPool 2x2 : (N, 64,  7,  7)
Flatten     : (N, 64*7*7 = 3136)
Linear      : (N, 128)
ReLU
Dropout 0.25: (N, 128)
Linear      : (N, 10)
```

Total parameters ~ **422,000** — still tiny compared with any modern vision model, yet enough to clear 99% test accuracy on MNIST in a handful of epochs.

## Why this works so well on MNIST

1. Digits have **strong local structure** — strokes, loops, intersections.
2. They are **largely translation-centred** — the MNIST images are pre-cropped and roughly centred.
3. They are **low resolution** — 28 x 28 is small enough that two pooling stages are sufficient.

For natural images (CIFAR, ImageNet) we would need many more layers, batch normalization, and data augmentation. None of that is needed here.

## When to reach for a CNN

Reach for a CNN when the input is an **image, an audio spectrogram, or any signal with locality and translation structure**. For tabular data (Iris, Wine Quality) an MLP is usually the right starting point — there is no spatial prior to encode.
