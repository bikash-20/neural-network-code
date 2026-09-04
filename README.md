# Neural Network Research Lab

> **A first-principles approach to deep learning:** Documenting the transition from pure linear algebra and vector calculus to production-grade PyTorch architectures.

---

## Project Overview

This repository documents an empirical, bottom-up exploration of neural network architectures. Following a strict **observe $\rightarrow$ hypothesize $\rightarrow$ implement $\rightarrow$ evaluate** workflow, this project bridges the gap between high-level abstractions (`torch.nn`) and the underlying multivariate calculus, matrix algebra, and algorithmic design that govern deep learning systems.

---

## Datasets

### 1. Iris Dataset (Phase 1)
- **Samples**: 150 flowers
- **Features**: 4 (sepal length, sepal width, petal length, petal width)
- **Classes**: 3 (Setosa, Versicolor, Virginica)
- **Task**: Multi-class classification
- **Implementation**: Pure NumPy from scratch (zero deep learning frameworks)

### 2. Wine Quality Dataset (Phase 2)
- **Samples**: 4,898 white wines (UCI Repository)
- **Features**: 11 chemical properties
- **Target**: Quality score (0–10), binarized to "good" ($\ge 7$) vs. "not good"
- **Task**: Binary classification
- **Implementation**: PyTorch Multilayer Perceptron (MLP)

### 3. MNIST Dataset (Phase 3)
- **Samples**: 70,000 digit images (60,000 train / 10,000 test)
- **Features**: $28 \times 28$ grayscale pixels
- **Classes**: 10 (digits 0–9)
- **Task**: Multi-class image classification
- **Implementation**: PyTorch Convolutional Neural Network (CNN)

---

## Architecture & Mathematical Foundations

### Phase 1: Pure NumPy Implementation (First Principles)

#### 1. Forward Propagation Mechanics
For an $L$-layer network, the pre-activation state $z^{(l)}$ and activation output $a^{(l)}$ at layer $l \in \{1, \dots, L\}$ are computed as:

$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$

$$a^{(l)} = f^{(l)}\left(z^{(l)}\right)$$

Where $W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$ represents the weight matrix, $b^{(l)} \in \mathbb{R}^{n_l}$ is the bias vector, and $a^{(0)} = x$ is the input sample vector.

#### 2. Multi-Class Probability Calibration (Softmax)
To transform unbounded logit vectors $z^{(L)} \in \mathbb{R}^K$ into a valid probability distribution $\hat{y} \in \Delta^{K-1}$ over $K$ target classes:

$$\hat{y}_i = \sigma(z^{(L)})_i = \frac{e^{z_i^{(L)}}}{\sum_{j=1}^{K} e^{z_j^{(L)}}}, \quad \text{for } i = 1, \dots, K$$

#### 3. Objective Function (Categorical Cross-Entropy)
For a batch of $N$ samples with one-hot encoded ground truth vectors $y_i \in \{0, 1\}^K$:

$$\mathcal{L}(W, b) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})$$

#### 4. The Backpropagation Algorithm (Chain Rule Derivation)
To minimize $\mathcal{L}$, we compute gradients using backward propagation of error vectors $\delta^{(l)} = \frac{\partial \mathcal{L}}{\partial z^{(l)}}$.

**A. Output Layer Gradient ($\delta^{(L)}$):**
Exploiting the Jacobian interaction between Categorical Cross-Entropy and Softmax yields the simplification:

$$\delta^{(L)} = \frac{\partial \mathcal{L}}{\partial z^{(L)}} = \hat{y} - y$$

**B. Hidden Layer Error Propagation ($\delta^{(l)}$):**
By applying the multivariate chain rule across layer transitions:

$$\delta^{(l)} = \left( (W^{(l+1)})^T \delta^{(l+1)} \right) \odot f'^{(l)}\left(z^{(l)}\right)$$

Where $\odot$ represents the Hadamard (element-wise) product and $f'^{(l)}$ is the derivative of the activation function.

**C. Parameter Gradients:**

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T, \qquad \frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}$$

**D. Gradient Updates:**

$$W^{(l)} := W^{(l)} - \alpha \frac{\partial \mathcal{L}}{\partial W^{(l)}}$$

$$b^{(l)} := b^{(l)} - \alpha \frac{\partial \mathcal{L}}{\partial b^{(l)}}$$

---

### Phase 2: Structural Regularization & Optimization

#### 1. Inverted Dropout Regularization
To mitigate co-adaptation of features during training, units are zeroed with probability $p$. During training, activation outputs are scaled by $\frac{1}{1-p}$ to maintain constant expectation $\mathbb{E}[a^{(l)}] = a^{(l)}$ at test time:

$$r_j^{(l)} \sim \text{Bernoulli}(1-p)$$

$$\hat{a}^{(l)} = \frac{1}{1-p} \cdot \left( a^{(l)} \odot r^{(l)} \right)$$

#### 2. Optimization Strategy (Adam Algorithm)
Parameters are updated using adaptive estimates of first ($m_t$) and second ($v_t$) uncentered moments:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

### Phase 3: Spatial Feature Extraction (Convolutional Networks)

#### 1. Discrete 2D Convolution
For an input tensor $X \in \mathbb{R}^{H \times W}$ and a learnable kernel $K \in \mathbb{R}^{k_1 \times k_2}$:

$$S(i, j) = (X * K)(i, j) = \sum_{m} \sum_{n} X(i-m, j-n) K(m, n)$$

Spatial reduction in pooling layers follows:

$$H_{out} = \left\lfloor \frac{H_{in} - K_h + 2P}{S} \right\rfloor + 1$$

---

## Experimental Setup & Performance

| Phase | Dataset | Model Architecture | Key Techniques | Optimization | Test Accuracy | Epochs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | **Iris** *(150 samples)* | 2-Layer Dense Network (4 $\rightarrow$ 8 $\rightarrow$ 3) | Pure NumPy, Analytical Backprop | Full-batch GD ($\alpha=0.01$) | **96.67%** | 1500 |
| **Phase 2** | **Wine Quality** *(4,898 white wines)* | Multi-Layer Perceptron (11 $\rightarrow$ 64 $\rightarrow$ 32 $\rightarrow$ 1) | Inverted Dropout ($p=0.2$), BCE | Adam ($lr=1e-3$) | **83.67%** | 100 |
| **Phase 3** | **MNIST** *(70,000 images)* | CNN (Conv3x3 $\rightarrow$ MaxPool $\rightarrow$ Conv3x3 $\rightarrow$ Linear) | Spatial Feature Maps, Dropout | Adam ($lr=1e-3$) | **99.09%** | 5 |

---

## Research Methodology

1. **Document Everything**: Record hyperparameter choices, convergence logs, and mathematical steps.
2. **Understand First**: Prove equations and backpropagation paths before writing framework abstractions.
3. **Experiment Systematically**: Modify one variable at a time (e.g., learning rate, dropout rate, batch size).
4. **Visualize Progress**: Track loss/accuracy curves, decision boundaries, and prediction errors.
5. **Question Assumptions**: Interrogate why failure modes occur (overfitting, vanishing gradients, capacity limits).

---

## Repository Structure

```text
neural-network/
├── README.md                  # Comprehensive mathematical & structural overview
├── RUN_INSTRUCTIONS.md        # Reproduction & environment setups
├── requirements.txt           # Environment dependencies
├── iris_run.log               # Captured Iris training logs
├── wine_run.log               # Captured Wine training logs
├── docs/                      # Theoretical documentation
│   ├── INDEX.md               # Guided reading path
│   ├── 01_iris_dataset.md     # Phase 1 setup & data analysis
│   ├── 02_neural_net_math.md  # Deep dive into calculus & matrix algebra
│   ├── 03_training_process.md # Optimization & loss behavior
│   ├── 04_results_analysis.md # Phase 1 performance breakdowns
│   ├── 05_wine_quality.md     # Phase 2 experimental setup
│   ├── 06_mnist_dataset.md    # Phase 3 dataset preprocessing
│   ├── 07_cnn_theory.md       # Convolutional mechanics & receptive fields
│   └── 08_phase3_results.md   # Phase 3 evaluation logs
├── notebooks/                 # Interactive Marimo walkthroughs
│   ├── 01_iris_explained.py
│   ├── 02_wine_quality_explained.py
│   └── 03_mnist_explained.py
├── models/                    # Saved model artifacts
│   ├── iris_model.npz
│   ├── wine_quality_model.pth
│   └── mnist_cnn_model.pth
├── plots/                     # Visualizations & loss curves
│   ├── training_history.png
│   ├── wine_training_history.png
│   ├── mnist_training_history.png
│   └── mnist_predictions.png
├── data/                      # Local datasets (gitignored)
│   ├── winequality-white.csv
│   └── mnist/
└── src/                       # Production source implementations
    ├── iris_classifier.py     # Custom NumPy engine
    ├── wine_quality.py        # PyTorch MLP Pipeline
    ├── mnist_classifier.py    # PyTorch Convolutional Engine
    └── utils.py               # Shared path & seeding helpers

Execution Guide
Environment Setup
Bash
# Clone repository
git clone [https://github.com/bikash-20/neural-network-code.git](https://github.com/bikash-20/neural-network-code.git)
cd neural-network-code

# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Reproducing Results
Bash
# Phase 1: Pure NumPy Iris Classifier
python3 src/iris_classifier.py

# Phase 2: PyTorch Wine Quality MLP
python3 src/wine_quality.py

# Phase 3: PyTorch MNIST CNN
python3 src/mnist_classifier.py
Interactive Notebook Exploration
To run the reactive Marimo notebooks for inline step-by-step mathematical walkthroughs:
Bash
marimo edit notebooks/01_iris_explained.py
Primary References & Resources
Nielsen, Michael A. "Neural Networks and Deep Learning", Determination Press, 2015. Online Book
3Blue1Brown. "But what is a neural network?", Visual Mathematics Series. YouTube
MIT 6.S191. "Introduction to Deep Learning". YouTube
LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). "Gradient-based learning applied to document recognition." Proceedings of the IEEE.
Goodfellow, I., Bengio, Y., & Courville, A. "Deep Learning", MIT Press, 2016.
Researcher: Bikash Talukder
Repository: github.com/bikash-20/neural-network-code
License: MIT
http://googleusercontent.com/youtube_content/1    
