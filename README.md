# Shared Computation AI Lab 🧠⚡

An interactive, educational Generative AI dashboard and laboratory built with **Python**, **PyTorch**, and **Streamlit**. This project demonstrates the core machine learning concept of **Shared Computation** and **Parameter Sharing** for next-word prediction using one canonical training sentence:

> **"The student studies AI and builds projects."**

---

## 🎯 Conceptual Overview

In classical sequential models or naive computing architectures, predicting the continuation of each word in a sequence might be conceptualized as separate processes or isolated models. 

Modern deep learning architectures (Transformers, RNNs, and Neural Bigram Models) leverage **Shared Computation**:
1. **Shared Parameters Across Sequence Positions:** A single embedding matrix $E$ and a single projection layer $(W, \mathbf{b})$ process every word token in the vocabulary.
2. **Parallel Batch Matrix Multiplications:** Rather than serializing 6 separate passes, all 6 input tokens are packaged into a batch tensor $\mathbf{X} \in \mathbb{R}^6$ and passed simultaneously through the neural network in **one unified forward pass**.
3. **Joint Gradient Descent:** The Cross-Entropy loss aggregates the error across all 6 target tokens simultaneously, and a single backward pass $\nabla_\Theta \mathcal{L}$ computes gradients to update the shared weights using the Adam optimizer.

---

## 🏗️ Architecture & Model Formulation

### 1. Tokenization & Bigram Pairs
- **Vocabulary ($V = 7$):**
  - `#0`: `"The"`
  - `#1`: `"student"`
  - `#2`: `"studies"`
  - `#3`: `"AI"`
  - `#4`: `"and"`
  - `#5`: `"builds"`
  - `#6`: `"projects"`

- **Six Ground-Truth Bigram Transitions:**
  1. `"The"` $\to$ `"student"`
  2. `"student"` $\to$ `"studies"`
  3. `"studies"` $\to$ `"AI"`
  4. `"AI"` $\to$ `"and"`
  5. `"and"` $\to$ `"builds"`
  6. `"builds"` $\to$ `"projects"`

### 2. Neural Network (`SharedNextWordModel` in `model.py`)
- **Shared Embedding Layer:** `nn.Embedding(vocab_size=7, embedding_dim=d)` (default $d=16$)
  - Weights $E \in \mathbb{R}^{7 \times 16}$ (112 parameters)
- **Shared Linear Output Layer:** `nn.Linear(in_features=16, out_features=7)`
  - Weights $W \in \mathbb{R}^{7 \times 16}$ (112 parameters)
  - Biases $\mathbf{b} \in \mathbb{R}^7$ (7 parameters)
- **Total Trainable Shared Parameters:** **231 parameters**

### 3. Batch Forward Pass Pipeline
$$\mathbf{X} = \begin{bmatrix} 0 & 1 & 2 & 3 & 4 & 5 \end{bmatrix}^T \in \mathbb{N}^6$$
$$\mathbf{H} = E[\mathbf{X}] \in \mathbb{R}^{6 \times 16}$$
$$\mathbf{Z} = \mathbf{H} W^T + \mathbf{b} \in \mathbb{R}^{6 \times 7}$$
$$\mathbf{P}_{i, j} = \frac{\exp(\mathbf{Z}_{i, j})}{\sum_{k=0}^6 \exp(\mathbf{Z}_{i, k})} \in \mathbb{R}^{6 \times 7}$$
$$\mathcal{L} = -\frac{1}{6} \sum_{i=1}^6 \log \mathbf{P}_{i, \mathbf{Y}_i}$$

---

## 🖥️ Dashboard Modules

| Module | Features & Capabilities |
| :--- | :--- |
| **1. Executive Overview** | Metrics summary, sentence breakdown, parameter counts, and theory behind shared computation. |
| **2. Tokenization & Bigrams** | Token dictionary, token ID assignments, interactive bigram cards, and batch tensor inspection. |
| **3. Shared Computation Architecture** | Interactive pipeline diagram, tensor dimension tracker, and live inspection of weight tensors. |
| **4. Before Training (Untrained Model)** | Real softmax probabilities from initial state (seed 42), comparison with theoretical random guess loss ($\ln 7 \approx 1.9459$). |
| **5. Training Engine** | Real PyTorch training with adjustable epochs & learning rate, live loss curve, and accuracy progression. |
| **6. After Training (Model Evaluation)** | Evaluates updated model, shows 100% measured classification accuracy, and sharp diagonal transition probabilities. |
| **7. Live Autoregressive Generation** | Step-by-step autoregressive generation from any starting word (e.g. `"The"`), temperature control, and step distribution charts. |
| **8. Comparative Visualizations** | Plotly grouped bar charts and side-by-side heatmaps comparing Pre-training vs Post-training target probabilities. |
| **9. Downloadable Results & Data** | One-click CSV exports for training pairs, probability matrices, evaluation reports, and loss history. |

---

## 🚀 Quickstart Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12 (macOS, Linux, or Windows)

### 1. Installation

Clone or open the project folder:
```bash
git clone https://github.com/<your-username>/Shared_Computation_Lab.git
cd Shared_Computation_Lab
```

Create and activate a virtual environment:
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
To verify the engine, model mathematics, and Streamlit components:
```bash
python test_app.py
```

### 3. Launch the Dashboard
Run the Streamlit application:
```bash
streamlit run app.py
```
Then open your browser to **`http://localhost:8501`**.

---

## 📁 Repository Structure

```text
Shared_Computation_Lab/
├── .streamlit/
│   └── config.toml     # Streamlit theme configuration (Dark Navy palette)
├── .gitignore          # Git exclusion rules (ignores .venv, __pycache__, etc.)
├── README.md           # Comprehensive project documentation
├── requirements.txt    # Project dependencies (torch, streamlit, plotly, pandas, numpy)
├── app.py              # Streamlit frontend & UI dashboard (Plotly, CSS dark theme, session state)
├── model.py            # Core PyTorch neural network, tokenizer, training loop, and autoregressive generator
└── test_app.py         # Automated test suite (unit tests & Streamlit AppTest integration)
```

---

## 🔬 Experimental Verification & Results

- **Untrained Model:**
  - Initial Cross-Entropy Loss: $\approx 1.95$ (closely matching theoretical $\ln 7 = 1.9459$)
  - Initial Prediction Accuracy: $\approx 16.7\%$ (1/6 correct by chance)
- **Trained Model (250 Epochs, $\text{lr}=0.05$):**
  - Final Loss: $< 0.001$
  - Final Prediction Accuracy: **100% (6/6 correct)**
  - Autoregressive Generation from `"The"`: **`"The student studies AI and builds projects."`**
