# Support Vector Machines: A Complete Guide

## 1. SVM: Linearly Separable and Non-Separable Cases

### The Core Idea

SVM finds the **best separating hyperplane**—the one with maximum margin between classes.

---

### 1.1 The Linearly Separable Case

#### The Decision Boundary

The hyperplane is defined by:

$$f(x) = w^T x + w_0 = 0$$

- $w$: Weight vector perpendicular to the hyperplane
- $w_0$: Bias term (shifts hyperplane from origin)

#### The Constraint: Correct Classification

For every training point $x^t$ with label $r^t \in \{-1, +1\}$:

$$r^t (w^T x^t + w_0) \geq +1$$

| If $r^t = +1$ | If $r^t = -1$ |
|---------------|---------------|
| Requires $w^T x^t + w_0 \geq +1$ | Requires $w^T x^t + w_0 \leq -1$ |

---

### 1.2 Margin Maximization

The distance from the hyperplane to the nearest point is:

$$\text{margin} = \frac{1}{\|w\|}$$

**To maximize margin, we minimize $\|w\|$.** For convenience:

$$\min_{w, w_0} \frac{1}{2}\|w\|^2$$

Subject to: $r^t(w^T x^t + w_0) \geq 1$ for all $t$

---

### 1.3 Soft Margin SVM (Linearly Inseparable)

Introduce slack variables $\xi^t \geq 0$:

$$r^t(w^T x^t + w_0) \geq 1 - \xi^t$$

**Interpretation of $\xi^t$:**

| Value | Meaning |
|-------|---------|
| $\xi^t = 0$ | Correctly classified, outside margin |
| $0 < \xi^t < 1$ | Correct but inside margin |
| $\xi^t = 1$ | On decision boundary |
| $\xi^t > 1$ | Misclassified |

#### The Optimization Problem

$$\min_{w, w_0, \xi} \frac{1}{2}\|w\|^2 + C \sum_t \xi^t$$

Subject to:
- $r^t(w^T x^t + w_0) \geq 1 - \xi^t$
- $\xi^t \geq 0$

#### The Role of $C$

| Large $C$ | Small $C$ |
|-----------|-----------|
| Penalizes errors heavily | Tolerates more errors |
| Narrow margin, risk of overfitting | Wide margin, better generalization |

---

## 2. The Kernel Trick

### 2.1 The Problem: Data That Can't Be Separated by a Line

Imagine this 1D dataset (drug dosage vs. cured/not cured):

```
Not Cured    Cured      Not Cured
   ●  ●    ○ ○ ○ ○ ○     ●  ●
   |--|------|-----|------|--|----→ Dosage
   1  2     3  4  5  6   7  8
```

**No single vertical line can separate ● from ○!**

The middle patients (dosage 3-6) are cured, but low and high dosages don't work.

---

### 2.2 The Solution: Add a Dimension

What if we create a new feature? Let's add $x^2$ (dosage squared):

| Dosage $x$ | $x^2$ | Class |
|------------|-------|-------|
| 1 | 1 | ● Not Cured |
| 2 | 4 | ● Not Cured |
| 3 | 9 | ○ Cured |
| 4 | 16 | ○ Cured |
| 5 | 25 | ○ Cured |
| 6 | 36 | ○ Cured |
| 7 | 49 | ● Not Cured |
| 8 | 64 | ● Not Cured |

Now plot in 2D space $(x, x^2)$:

```
x²
70│                      ●(8,64)
60│
50│                    ●(7,49)
40│                  ○(6,36)
30│                ○(5,25)
20│             ○(4,16)
10│          ○(3,9)
  │     ●(2,4)
  │●(1,1)
  └────────────────────────→ x
   1  2  3  4  5  6  7  8
```

**Now a straight line CAN separate them!** The cured patients form a cluster that's linearly separable in this 2D space.

---

### 2.3 Feature Mapping $\varphi(x)$

This transformation is called **feature mapping**:

$$\varphi(x) = \begin{bmatrix} x \\ x^2 \end{bmatrix}$$

More generally, we can create polynomial features:

$$\varphi(x) = [1, x, x^2, x^3, \ldots]$$

**Problem:** If you have 100 original features and want degree-5 polynomials, you get **millions** of new features. Computing and storing $\varphi(x)$ becomes impossible!

---

### 2.4 The Kernel Trick: The Key Insight

Here's the magic: **SVM only ever needs dot products between transformed points**, never the transformed points themselves.

When SVM is trained, it solves an optimization problem (called the **dual form**) that outputs:
- **Which training points are support vectors** (points on/near the margin)
- **An importance weight $\alpha_t$** for each support vector

| $\alpha_t$ value | Meaning |
|------------------|---------|
| $\alpha_t = 0$ | Point $t$ is NOT a support vector (doesn't affect boundary) |
| $\alpha_t > 0$ | Point $t$ IS a support vector; larger = more influence |

#### The Decision Function: Classifying a NEW Point

$$f(x) = \sum_t \alpha_t \cdot r^t \cdot K(x^t, x) + w_0$$

**What each variable means:**

| Symbol | What it is | Where it comes from |
|--------|------------|---------------------|
| $x$ | The **NEW point** you want to classify | Given to you (unknown label) |
| $x^t$ | A **training point** (support vector) | From your training data |
| $r^t$ | The **label** of training point $t$ | From your training data (+1 or -1) |
| $\alpha_t$ | **Importance weight** of support vector $t$ | Learned during training |
| $K(x^t, x)$ | **Similarity** between training point and new point | Computed using kernel |
| $w_0$ | **Bias term** | Learned during training |

#### Step-by-Step Example

**Setup:** You trained an SVM and got 3 support vectors:

| Support Vector | Coordinates $x^t$ | Label $r^t$ | Weight $\alpha_t$ |
|----------------|-------------------|-------------|-------------------|
| $t=1$ | $[2, 1]$ | $+1$ (positive) | $0.5$ |
| $t=2$ | $[1, 3]$ | $+1$ (positive) | $0.3$ |
| $t=3$ | $[5, 4]$ | $-1$ (negative) | $0.8$ |

Bias: $w_0 = 0.1$

**Task:** Classify a NEW point $x = [3, 2]$ (we don't know its label!)

**Step 1:** Compute similarity between $x$ and each support vector using RBF kernel:
- $K(x^1, x) = K([2,1], [3,2]) = 0.6$ (fairly similar)
- $K(x^2, x) = K([1,3], [3,2]) = 0.4$ (less similar)
- $K(x^3, x) = K([5,4], [3,2]) = 0.5$ (moderately similar)

**Step 2:** Plug into the decision function:

$$f(x) = \alpha_1 \cdot r^1 \cdot K(x^1, x) + \alpha_2 \cdot r^2 \cdot K(x^2, x) + \alpha_3 \cdot r^3 \cdot K(x^3, x) + w_0$$

$$f(x) = (0.5)(+1)(0.6) + (0.3)(+1)(0.4) + (0.8)(-1)(0.5) + 0.1$$

$$f(x) = 0.3 + 0.12 - 0.4 + 0.1 = 0.12$$

**Step 3:** Interpret the result:
- $f(x) = 0.12 > 0$ → **Predict positive class (+1)**

**In plain English:** The new point is more similar to the positive support vectors than the negative one, so we predict positive.

---

### 2.5 Numerical Example: Why Kernels Save Computation

Let's use the **polynomial kernel** with degree $d=2$ on 2D points:

$$K(a, b) = (a^T b + 1)^d = (a^T b + 1)^2$$

> **Note:** $d$ is the **degree** (sometimes called $q$ in textbooks). It controls how complex the decision boundary can be. Higher $d$ = more curvy boundary.

**Given two points:**
- $a = [2, 3]$
- $b = [4, 1]$

**Method 1: Explicit transformation (the hard way)**

First compute the 6D transformation:
$$\varphi(x) = [1, \sqrt{2}x_1, \sqrt{2}x_2, x_1^2, \sqrt{2}x_1 x_2, x_2^2]$$

For $a = [2, 3]$:
$$\varphi(a) = [1, \sqrt{2}(2), \sqrt{2}(3), 4, \sqrt{2}(6), 9] = [1, 2.83, 4.24, 4, 8.49, 9]$$

For $b = [4, 1]$:
$$\varphi(b) = [1, \sqrt{2}(4), \sqrt{2}(1), 16, \sqrt{2}(4), 1] = [1, 5.66, 1.41, 16, 5.66, 1]$$

Now compute dot product (6 multiplications + 5 additions):
$$\varphi(a)^T \varphi(b) = 1(1) + 2.83(5.66) + 4.24(1.41) + 4(16) + 8.49(5.66) + 9(1) = 144$$

**Method 2: Kernel trick (the easy way)**

$$K(a, b) = (a^T b + 1)^2$$

Step 1: Compute $a^T b = 2(4) + 3(1) = 8 + 3 = 11$

Step 2: Add 1: $11 + 1 = 12$

Step 3: Square it: $12^2 = 144$ ✓

**Same answer! But only 3 operations instead of 11!**

---

### 2.6 Common Kernel Functions

#### Polynomial Kernel

$$K(x, y) = (x^T y + r)^d$$

- $d$: degree (complexity of boundary)
- $r$: constant (usually 1)
- $d=1$: linear (no transformation)
- $d=2$: quadratic features ($x_1^2, x_1 x_2$, etc.)

#### RBF (Gaussian) Kernel

$$K(x, y) = \exp\left[-\frac{\|x - y\|^2}{2s^2}\right]$$

- Measures "how close" two points are
- $\|x - y\|^2$ = squared distance between points
- $s$ controls the "reach" of each point

| Points | Distance | Kernel Value |
|--------|----------|--------------|
| Very close | Small | $\approx 1$ (very similar) |
| Far apart | Large | $\approx 0$ (very different) |

**Mind-blowing fact:** RBF kernel implicitly maps to **infinite-dimensional** space!

---

### 2.7 How Classification Works with Kernels

After training, SVM stores:
- **Support vectors** $x^1, x^2, \ldots, x^k$ (the important boundary points)
- **Weights** $\alpha_1, \alpha_2, \ldots, \alpha_k$
- **Labels** $r^1, r^2, \ldots, r^k$ (where $r \in \{-1, +1\}$)

To classify a new point $x$:

$$f(x) = \sum_{i=1}^{k} \alpha_i \cdot r^i \cdot K(x^i, x) + w_0$$

**In words:** 
1. Compute similarity between new point and each support vector
2. Weight by importance ($\alpha_i$) and class label ($y_i$)
3. Sum up → positive = class +1, negative = class -1

**Example:** If you have 3 support vectors:

$$f(x) = \alpha_1 r^1 K(x^1, x) + \alpha_2 r^2 K(x^2, x) + \alpha_3 r^3 K(x^3, x) + w_0$$

Only 3 kernel computations needed, regardless of original training set size!

---

## 3. Hinge Loss SVM

### 3.1 The Hinge Loss Function

$$L_{\text{hinge}} = \max(0, 1 - r^t \cdot f(x^t))$$

Where $r^t \in \{-1, +1\}$ is the true label and $f(x^t) = w^T x^t + w_0$ is the model output.

---

### 3.2 Understanding Hinge Loss

The product $r^t \cdot f(x^t)$ is the **functional margin**:

| Condition | Meaning | Loss |
|-----------|---------|------|
| $r^t \cdot f(x^t) \geq 1$ | Correct, outside margin | $0$ |
| $0 < r^t \cdot f(x^t) < 1$ | Correct, inside margin | $1 - r^t \cdot f(x^t)$ |
| $r^t \cdot f(x^t) < 0$ | Misclassified | $> 1$ |

---

### 3.3 Unconstrained Optimization Form

$$\min_w \left[ \frac{1}{2}\|w\|^2 + C \sum_{t} \max(0, 1 - r^t \cdot f(x^t)) \right]$$

**Two parts:**
1. $\frac{1}{2}\|w\|^2$ — Regularization (maximize margin)
2. $C \sum \max(0, 1 - r^t \cdot f(x^t))$ — Hinge loss (penalize violations)

**Connection:** The slack variable equals the hinge loss:

$$\xi^t = \max(0, 1 - r^t \cdot f(x^t))$$

---

## Summary

| Concept | Formula |
|---------|---------|
| Hard margin objective | $\min \frac{1}{2}\|w\|^2$ s.t. $r^t(w^T x^t + w_0) \geq 1$ |
| Soft margin objective | $\min \frac{1}{2}\|w\|^2 + C\sum \xi^t$ |
| Polynomial kernel | $K(x, y) = (x^T y + 1)^d$ |
| RBF kernel | $K(x, y) = \exp\left[-\frac{\|x - y\|^2}{2s^2}\right]$ |
| Hinge loss | $\max(0, 1 - r^t \cdot f(x^t))$ |
| Unconstrained SVM | $\min \frac{1}{2}\|w\|^2 + C\sum \max(0, 1 - r^t \cdot f(x^t))$ |

---

## Your Question: "Do We Get All Pairs?"

**No!** Only **support vectors** matter after training.

At prediction for new point $x$:

$$f(x) = \sum_{\text{support vectors } t} \alpha_t \cdot r^t \cdot K(x^t, x) + w_0$$

If you have 10,000 training points but only 50 support vectors, you compute only 50 kernels per prediction.
