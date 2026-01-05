# Decision Tree: Root Node Impurity Calculation

## What is a Decision Tree?

A decision tree is a machine learning algorithm that makes predictions by splitting data based on features. Before we can decide which feature to split on, we need to measure how "mixed" or "impure" our data is.

---

## Our Dataset

We have 10 animals with 4 features and a class label (+ or -):

| Length | Gills | Beak | Teeth | Class |
|--------|-------|------|-------|-------|
| 3 | N | Y | M | + |
| 4 | N | Y | M | + |
| 3 | N | Y | F | + |
| 5 | N | Y | M | + |
| 5 | N | Y | F | + |
| 5 | Y | Y | M | - |
| 4 | Y | Y | M | - |
| 5 | Y | N | M | - |
| 4 | Y | N | M | - |
| 4 | N | Y | F | - |

---

## Step 1: Count the Classes

Before calculating anything, we need to count how many of each class we have:

- **Total samples:** 10
- **Positive class (+):** 5 samples
- **Negative class (-):** 5 samples

---

## Step 2: Calculate Probabilities

The probability of each class is simply the count divided by total:

```
p(+) = 5 / 10 = 0.5  (50%)
p(-) = 5 / 10 = 0.5  (50%)
```

---

## Method 1: Gini Index

### The Formula

```
Gini = 1 - Σ(p_i)²
```

This means: Take 1, then subtract the sum of each probability squared.

### Step-by-Step Calculation

**Step A:** Square each probability
```
p(+)² = 0.5 × 0.5 = 0.25
p(-)² = 0.5 × 0.5 = 0.25
```

**Step B:** Add the squared probabilities
```
Sum = 0.25 + 0.25 = 0.5
```

**Step C:** Subtract from 1
```
Gini = 1 - 0.5 = 0.5
```

### Result
```
Gini Index = 0.5
```

### What Does This Mean?

- Gini = 0 means the node is "pure" (all samples belong to one class)
- Gini = 0.5 (for 2 classes) means maximum impurity (perfectly mixed)

Our Gini of 0.5 means we have the most mixed possible situation - exactly half positive and half negative.

---

## Method 2: Entropy

### The Formula

```
Entropy = Σ p_i × log₂(1/p_i)
```

This is the same as:
```
Entropy = -Σ p_i × log₂(p_i)
```

### Step-by-Step Calculation

**Step A:** Calculate log₂(1/p) for each class

For the positive class:
```
log₂(1/0.5) = log₂(2) = 1
```

For the negative class:
```
log₂(1/0.5) = log₂(2) = 1
```

**Step B:** Multiply each by its probability
```
For (+): 0.5 × 1 = 0.5
For (-): 0.5 × 1 = 0.5
```

**Step C:** Add them together
```
Entropy = 0.5 + 0.5 = 1.0 bit
```

### Result
```
Entropy = 1.0 bit
```

### What Does This Mean?

- Entropy = 0 means the node is "pure" (all samples belong to one class)
- Entropy = 1.0 (for 2 classes) means maximum impurity (perfectly mixed)

Our Entropy of 1.0 bits means maximum uncertainty - we cannot predict the class better than random guessing.

---

## Summary Table

| Measure | Formula | Our Result | Meaning |
|---------|---------|------------|---------|
| **Gini Index** | 1 - Σp_i² | 0.5 | Maximum impurity |
| **Entropy** | Σp_i × log₂(1/p_i) | 1.0 bit | Maximum uncertainty |

---

## Why Does This Matter?

The root node has maximum impurity because our classes are perfectly balanced (5 vs 5). 

**Next step:** We would calculate the impurity AFTER splitting on each feature, then choose the feature that reduces impurity the most! This reduction is called "Information Gain."

---

## Quick Reference: Logarithm Values

If you need to calculate entropy by hand, here are useful log₂ values:

| Value | log₂(Value) |
|-------|-------------|
| 1 | 0 |
| 2 | 1 |
| 4 | 2 |
| 8 | 3 |
| 0.5 | -1 |
| 0.25 | -2 |

Remember: log₂(1/x) = -log₂(x)
