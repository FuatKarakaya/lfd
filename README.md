# Why Convexity of Cross-Entropy Matters (with Proof)

> **Key Takeaway**: The cross-entropy loss for logistic regression is **convex**, which guarantees gradient descent finds the global optimum.

---

## 🎯 Why Should You Care?

Consider two scenarios:

| Loss Function Shape | What Happens with Gradient Descent |
|---------------------|-----------------------------------|
| **Convex** (bowl-shaped) | Always finds the global minimum ✓ |
| **Non-convex** (wavy) | Might get stuck in local minima ✗ |

Logistic regression has a **convex loss** → no matter where you start, gradient descent will find the best possible weights!

---

## 📐 Background: What Is Convexity?

A function $f(x)$ is **convex** if its second derivative is non-negative:

$$f''(x) \geq 0 \quad \text{for all } x$$

For multi-variable functions, the **Hessian matrix** (matrix of second derivatives) must be **positive semi-definite**.

Geometrically: the function curves upward like a bowl 🥣

---

## 🧮 The Proof: Cross-Entropy Is Convex

### Setup

For logistic regression with one weight $w$:

$$\hat{y} = \sigma(wx) = \frac{1}{1 + e^{-wx}}$$

Cross-entropy loss for one sample:

$$\mathcal{L} = -\left[ y \log(\hat{y}) + (1-y) \log(1-\hat{y}) \right]$$

### Step 1: First Derivative

From the gradient descent derivation, we know:

$$\frac{\partial \mathcal{L}}{\partial w} = (\hat{y} - y) \cdot x$$

### Step 2: Second Derivative

$$\frac{\partial^2 \mathcal{L}}{\partial w^2} = \frac{\partial}{\partial w}\left[(\hat{y} - y) \cdot x\right] = x \cdot \frac{\partial \hat{y}}{\partial w}$$

We know from the sigmoid derivative that:

$$\frac{\partial \hat{y}}{\partial w} = \frac{\partial \hat{y}}{\partial s} \cdot \frac{\partial s}{\partial w} = \hat{y}(1-\hat{y}) \cdot x$$

Therefore:

$$\frac{\partial^2 \mathcal{L}}{\partial w^2} = x \cdot \hat{y}(1-\hat{y}) \cdot x = \hat{y}(1-\hat{y}) \cdot x^2$$

### Step 3: Prove It's Non-Negative

For the sum over all data points:

$$\frac{\partial^2 \mathcal{L}}{\partial w^2} = \sum_i \hat{y}^{(i)}(1-\hat{y}^{(i)}) \cdot (x^{(i)})^2$$

**Why is this always $\geq 0$?**

| Term | Range | Sign |
|------|-------|------|
| $\hat{y}^{(i)}$ | $(0, 1)$ | Positive ✓ |
| $(1-\hat{y}^{(i)})$ | $(0, 1)$ | Positive ✓ |
| $(x^{(i)})^2$ | $[0, \infty)$ | Non-negative ✓ |

Product of positives = **positive** ✓

$$\boxed{\frac{\partial^2 \mathcal{L}}{\partial w^2} \geq 0 \implies \text{Cross-Entropy is Convex}}$$

---

## 📊 Multi-Dimensional Case (Hessian)

For weight vector $\mathbf{w}$, the Hessian matrix is:

$$\mathbf{H} = \sum_i \hat{y}^{(i)}(1-\hat{y}^{(i)}) \cdot \mathbf{x}^{(i)} (\mathbf{x}^{(i)})^\top$$

This is a sum of **outer products** scaled by positive values.

For any vector $\mathbf{v}$:

$$\mathbf{v}^\top \mathbf{H} \mathbf{v} = \sum_i \hat{y}^{(i)}(1-\hat{y}^{(i)}) \cdot (\mathbf{v}^\top \mathbf{x}^{(i)})^2 \geq 0$$

Since this is always non-negative, $\mathbf{H}$ is **positive semi-definite** → **convex**! ✓

---

## 💡 Practical Implications

### 1. Guaranteed Convergence
Gradient descent will always converge to the global minimum (with appropriate learning rate).

### 2. No Hyperparameter Sensitivity
Unlike neural networks, you don't need tricks like:
- Multiple random initializations
- Learning rate schedules
- Momentum or Adam optimizer

Simple gradient descent works!

### 3. Unique Solution
There's only one optimal set of weights (up to regularization).

---

## 🎓 Related Exam Questions

> **Q: Is the log-likelihood for logistic regression concave or convex?**

**A**: The **log-likelihood is concave** (we want to maximize it).
The **cross-entropy loss is convex** (we want to minimize it).
They're negatives of each other.

> **Q: Why do we prefer convex loss functions?**

**A**: Because gradient-based optimization is guaranteed to find the global optimum, not just a local one.

> **Q: What's the second derivative of the logistic regression loss?**

**A**: $\sum_i \hat{y}^{(i)}(1-\hat{y}^{(i)}) \cdot (x^{(i)})^2$ — always non-negative!

---

## 🔗 Summary

| Property | Cross-Entropy Loss | Log-Likelihood |
|----------|-------------------|----------------|
| Shape | Convex (bowl) | Concave (dome) |
| Goal | Minimize | Maximize |
| Second derivative | $\geq 0$ | $\leq 0$ |
| Optimization | Guaranteed global min | Guaranteed global max |
