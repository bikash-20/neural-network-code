# Documentation Index

This repo is structured as a **researcher's notebook**: theory first, code second, results third. Each phase has a runnable script in `src/`, a Marimo walkthrough in `notebooks/`, and one or more theory notes here in `docs/`.

## Recommended reading paths

### Path A — Theory first (recommended for first-time readers)

Read in order. Each note is short and links forward to the code.

1. `docs/01_iris_dataset.md` — the Iris dataset and what makes it tractable.
2. `docs/02_neural_network_theory.md` — forward propagation, softmax, cross-entropy.
3. `docs/03_training_process.md` — gradient descent, backprop, evaluation.
4. `docs/04_results_analysis.md` — interpreting Iris results and what they tell us.
5. `docs/05_wine_quality.md` — moving from NumPy to PyTorch on tabular data.
6. `docs/06_mnist_dataset.md` — the MNIST dataset and why it asks for a CNN.
7. `docs/07_cnn_theory.md` — convolutions, pooling, receptive fields.
8. `docs/08_phase3_results.md` — comparing all three phases.

### Path B — Code first

If you prefer to read scripts first, the canonical entry points are:

- `src/iris_classifier.py` — pure NumPy, 2-layer NN.
- `src/wine_quality.py` — PyTorch MLP on tabular data.
- `src/mnist_classifier.py` — PyTorch CNN on images.

Run them in that order. Each script is self-contained and prints its own progress.

### Path C — Interactive notebooks (Marimo)

For a narrated walkthrough with inline math and inline plots:

- `notebooks/01_iris_explained.py`
- `notebooks/02_wine_quality_explained.py`
- `notebooks/03_mnist_explained.py`

Each notebook imports or reuses the corresponding `src/*.py` implementation and is meant to be opened with Marimo:

```bash
uv run marimo edit notebooks/01_iris_explained.py
# or
pipx run marimo edit notebooks/01_iris_explained.py
```

## File-to-script map

| Topic                          | Doc                              | Script                              | Notebook                              |
|--------------------------------|----------------------------------|-------------------------------------|---------------------------------------|
| Iris dataset                   | `01_iris_dataset.md`             | `src/iris_classifier.py`            | `notebooks/01_iris_explained.py`      |
| NN math (forward, backprop)    | `02_neural_network_theory.md`    | (covered by iris script)            | `notebooks/01_iris_explained.py`      |
| Training process               | `03_training_process.md`         | (covered by iris script)            | `notebooks/01_iris_explained.py`      |
| Iris results                   | `04_results_analysis.md`         | `src/iris_classifier.py`            | `notebooks/01_iris_explained.py`      |
| Wine Quality (PyTorch intro)   | `05_wine_quality.md`             | `src/wine_quality.py`               | `notebooks/02_wine_quality_explained.py` |
| MNIST dataset                  | `06_mnist_dataset.md`            | `src/mnist_classifier.py`           | `notebooks/03_mnist_explained.py`     |
| CNN theory                     | `07_cnn_theory.md`               | `src/mnist_classifier.py`           | `notebooks/03_mnist_explained.py`     |
| Phase 3 results & comparison   | `08_phase3_results.md`           | `src/mnist_classifier.py`           | `notebooks/03_mnist_explained.py`     |

## Project at a glance

- **Phase 1 — Iris.** 4 -> 10 -> 3 NN, NumPy from scratch, 96.67% test accuracy.
- **Phase 2 — Wine Quality.** 11 -> 64 -> 32 -> 2 MLP + Dropout, PyTorch, 83.67% test accuracy.
- **Phase 3 — MNIST.** Conv(32) -> Conv(64) -> FC(128) -> 10, PyTorch, ~99% test accuracy.
