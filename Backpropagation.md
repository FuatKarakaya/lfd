# Backpropagation Explained

> A complete walkthrough of your lecture notes on training neural networks.

---

## 1. The Neural Network Structure

Your notes show a simple feedforward network with three layers of neurons:

```
Layer ℓ  →  Layer i  →  Layer j
 (oₗ)        (oᵢ)        (oⱼ)
```

Each neuron has two key values:

| Symbol | Name | Description |
|--------|------|-------------|
| $u$ | **Pre-activation** | The weighted sum of inputs |
| $o$ | **Output** | Result after applying activation function |

---

## 2. Forward Pass

The forward pass computes outputs from inputs, layer by layer.

### Step 1: Compute the weighted sum

$$
u_i = \sum_{\ell} \theta_{i\ell} \cdot o_\ell
$$

**Translation:** "The input to neuron $i$ is the sum of all previous outputs $o_\ell$, each multiplied by its corresponding weight $\theta_{i\ell}$."

### Step 2: Apply the activation function

$$
o_i = \sigma(u_i)
$$

**Translation:** "The output of neuron $i$ is the activation function $\sigma$ applied to the weighted sum."

> [!NOTE]
> Common activation functions include:
> - **Sigmoid:** $\sigma(x) = \frac{1}{1 + e^{-x}}$
> - **ReLU:** $\sigma(x) = \max(0, x)$
> - **Tanh:** $\sigma(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

---

## 3. The Goal: Minimize Error

We have a **loss function** (error function) that measures how wrong our predictions are:

$$
Err = \sum_i (y^{(i)} - \hat{y}^{(i)})^2
$$

This is **Mean Squared Error (MSE)** — the sum of squared differences between true values $y$ and predictions $\hat{y}$.

**Our goal:** Adjust the weights $\theta$ to minimize this error.

---

## 4. The Chain Rule (Heart of Backpropagation)

To update weights, we need: *"How much does the error change if I change this weight?"*

$$
\frac{\partial Err}{\partial \theta_{i\ell}} = \frac{\partial Err}{\partial u_i} \cdot \frac{\partial u_i}{\partial \theta_{i\ell}}
$$

This breaks into two parts:

| Term | Meaning |
|------|---------|
| $\frac{\partial Err}{\partial u_i}$ | How the error changes with neuron $i$'s input (we call this **$\delta_i$**) |
| $\frac{\partial u_i}{\partial \theta_{i\ell}}$ | How the input changes with the weight (this is just $o_\ell$!) |

So:

$$
\boxed{\frac{\partial Err}{\partial \theta_{i\ell}} = \delta_i \cdot o_\ell}
$$

---

## 5. Computing Delta ($\delta$)

The **delta** for each neuron is the error signal that propagates backward.

### For the Output Layer (Last Layer)

At the output layer $k$, the prediction *is* the output: $\hat{y} = o_k$

$$
\delta_k = \frac{\partial Err}{\partial u_k} = \frac{\partial Err}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial u_k}
$$

For MSE loss $Err = (y - \hat{y})^2$:

$$
\boxed{\delta_k = -2(y - \hat{y}) \cdot \sigma'(u_k)}
$$

### For Hidden Layers (The Recursive Formula)

This is the magic — we compute $\delta_i$ from the deltas of the *next* layer:

$$
\boxed{\delta_i = \sigma'(u_i) \cdot \sum_j \delta_j \cdot \theta_{ji}}
$$

**In words:** 
1. Take each neuron $j$ in the next layer
2. Multiply its delta $\delta_j$ by the weight connecting $i$ to $j$
3. Sum all these contributions
4. Multiply by the derivative of the activation at neuron $i$

> [!IMPORTANT]
> This formula is why it's called **back**propagation — we propagate errors *backward* from output to input!

---

## 6. The Complete Algorithm

```
┌─────────────────────────────────────────┐
│           FORWARD PASS                  │
│  Input → Compute all u and o → Output   │
│           ─────────────────►            │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│          BACKWARD PASS                  │
│  Compute δₖ at output layer             │
│  For each layer (right to left):        │
│      δᵢ = σ'(uᵢ) · Σⱼ δⱼ · θⱼᵢ          │
│           ◄─────────────────            │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│          WEIGHT UPDATE                  │
│  θᵢₗ ← θᵢₗ - η · δᵢ · oₗ                │
│  (η is the learning rate)               │
└─────────────────────────────────────────┘
```

---

## 7. Summary of Key Equations

| Step | Equation | Purpose |
|------|----------|---------|
| Forward (weighted sum) | $u_i = \sum_\ell \theta_{i\ell} \cdot o_\ell$ | Combine inputs |
| Forward (activation) | $o_i = \sigma(u_i)$ | Apply nonlinearity |
| Delta (output layer) | $\delta_k = \frac{\partial Err}{\partial \hat{y}} \cdot \sigma'(u_k)$ | Error at output |
| Delta (hidden layers) | $\delta_i = \sigma'(u_i) \cdot \sum_j \delta_j \cdot \theta_{ji}$ | Propagate error backward |
| Weight gradient | $\frac{\partial Err}{\partial \theta_{i\ell}} = \delta_i \cdot o_\ell$ | How to update weights |

---

## 8. Why Does This Work?

The chain rule lets us decompose a complicated derivative into simpler pieces:

$$
\frac{\partial Err}{\partial \theta} = \underbrace{\frac{\partial Err}{\partial u}}_{\delta} \cdot \underbrace{\frac{\partial u}{\partial \theta}}_{o}
$$

By computing deltas layer-by-layer from output to input, we efficiently reuse calculations — each $\delta_i$ only needs to be computed once!

> [!TIP]
> **Intuition:** Each $\delta$ answers the question: "If I wiggle this neuron's input, how much does the final error wiggle?"

---

## 9. Common Pitfalls

> [!WARNING]
> **Vanishing Gradients:** If $\sigma'(u_i)$ is very small (like for sigmoid when $|u|$ is large), the deltas shrink as they propagate backward. This is why ReLU became popular!

> [!WARNING]  
> **Exploding Gradients:** If weights are too large, deltas can grow exponentially. Solutions include gradient clipping and careful initialization.

---

*Now you understand the math behind every neural network! 🎉*
