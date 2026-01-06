# Softmax & Multi-Class Classification (Explained)

> Your note extends logistic regression from **2 classes** to **K classes** (here K=3).

---

## 🎯 The Setup

You have:
- **Inputs**: $x_0 = 1$ (bias trick), $x_1, x_2, \ldots, x_d$ (features)
- **K = 3 classes** to predict (could be cat, dog, bird for example)
- **Goal**: Output 3 probabilities $\hat{y}_1, \hat{y}_2, \hat{y}_3$ that sum to 1

---

## 🧮 Step 1: Compute a Score for Each Class

Each class $k$ gets its own linear combination (called $u_k$):

$$u_1 = \theta_{10} + \theta_{11}x_1 + \theta_{12}x_2 + \cdots + \theta_{1d}x_d$$

$$u_2 = \theta_{20} + \theta_{21}x_1 + \theta_{22}x_2 + \cdots + \theta_{2d}x_d$$

$$u_3 = \theta_{30} + \theta_{31}x_1 + \theta_{32}x_2 + \cdots + \theta_{3d}x_d$$

**Each class has its OWN set of weights!**

| Class | Weights |
|-------|---------|
| Class 1 | $\theta_{10}, \theta_{11}, \theta_{12}, \ldots, \theta_{1d}$ |
| Class 2 | $\theta_{20}, \theta_{21}, \theta_{22}, \ldots, \theta_{2d}$ |
| Class 3 | $\theta_{30}, \theta_{31}, \theta_{32}, \ldots, \theta_{3d}$ |

---

## 🔄 Step 2: Convert Scores to Probabilities (Softmax)

The scores $u_1, u_2, u_3$ can be any real numbers. We need **probabilities** (positive, sum to 1).

**Softmax** does this:

$$\hat{y}_k = \frac{\exp(u_k)}{\sum_{j=1}^{K} \exp(u_j)}$$

For K = 3:

$$\hat{y}_1 = \frac{e^{u_1}}{e^{u_1} + e^{u_2} + e^{u_3}}$$

$$\hat{y}_2 = \frac{e^{u_2}}{e^{u_1} + e^{u_2} + e^{u_3}}$$

$$\hat{y}_3 = \frac{e^{u_3}}{e^{u_1} + e^{u_2} + e^{u_3}}$$

**Why $\exp$?** Because $e^x > 0$ always, ensuring positive probabilities.

**Why divide by sum?** To normalize so they add up to 1.

---

## 🏷️ Step 3: One-Hot Encoded Labels

Your note shows this table (right side):

| Sample | $x_1, x_2, \ldots, x_d$ | $y_1$ | $y_2$ | $y_3$ |
|--------|------------------------|-------|-------|-------|
| 1 | ... | 1 | 0 | 0 |
| 2 | ... | 1 | 0 | 0 |
| 3 | ... | 0 | 1 | 0 |
| 4 | ... | 0 | 1 | 0 |
| 5 | ... | 0 | 0 | 1 |
| 6 | ... | 0 | 0 | 1 |

**One-hot encoding**: Only ONE of $y_1, y_2, y_3$ is 1, the rest are 0.

- Sample belongs to Class 1 → $[1, 0, 0]$
- Sample belongs to Class 2 → $[0, 1, 0]$
- Sample belongs to Class 3 → $[0, 0, 1]$

---

## 🖼️ The Network Diagram

Your sketch shows:

```
         ┌──→ Σ ──→ u₁ ──→ softmax ──→ ŷ₁
x₀=1 ────┼──→ Σ ──→ u₂ ──→ softmax ──→ ŷ₂
x₁ ──────┼──→ Σ ──→ u₃ ──→ softmax ──→ ŷ₃
x₂ ──────┘
  ⋮
```

- Each input connects to ALL output nodes (fully connected)
- Each Σ computes a weighted sum for one class
- Softmax converts all scores to probabilities together

---

## 🔗 Connection to Binary Logistic Regression

| | Binary (K=2) | Multi-class (K classes) |
|---|--------------|------------------------|
| Activation | Sigmoid | Softmax |
| Output | 1 probability $\hat{y}$ | K probabilities $\hat{y}_1, \ldots, \hat{y}_K$ |
| Labels | 0 or 1 | One-hot vector |
| Parameters | 1 weight vector | K weight vectors |

**Fun fact**: Softmax with K=2 is mathematically equivalent to sigmoid!

---

## 📐 In Matrix Form

If you stack everything:

$$\mathbf{U} = \mathbf{\Theta} \cdot \mathbf{x}$$

Where:
- $\mathbf{x}$ is $(d+1) \times 1$ input vector (including $x_0 = 1$)
- $\mathbf{\Theta}$ is $K \times (d+1)$ weight matrix
- $\mathbf{U}$ is $K \times 1$ score vector

Then apply softmax to get $\hat{\mathbf{y}}$.

---

## ✅ Summary

1. **Each class gets its own weights** → computes a score $u_k$
2. **Softmax** converts scores to probabilities
3. **Labels are one-hot** → only one class is "on"
4. **Training**: Find weights that maximize probability of correct class
