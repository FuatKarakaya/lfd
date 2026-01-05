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

---

# Part 2: Finding the Best Split (First Node) Using Gini

Now we need to find which feature creates the best split. We'll calculate the **Gini Gain** for each feature and pick the one with the highest gain.

## The Goal

We want to split our 10 samples into groups that are MORE pure (less mixed) than the original.

**Original Gini = 0.5** (calculated above)

We need to find which feature REDUCES this impurity the most.

---

## Formula for Weighted Gini After Split

When we split on a feature, we get multiple groups. The weighted Gini is:

```
Weighted Gini = Σ (n_i / n_total) × Gini_i
```

Where:
- n_i = number of samples in group i
- n_total = total samples (10)
- Gini_i = Gini impurity of group i

**Gini Gain = Original Gini - Weighted Gini After Split**

---

## Feature 1: Gills (Y or N)

### Step 1: Split the data by Gills

**Gills = Y (4 samples):**
| Length | Gills | Beak | Teeth | Class |
|--------|-------|------|-------|-------|
| 5 | Y | Y | M | - |
| 4 | Y | Y | M | - |
| 5 | Y | N | M | - |
| 4 | Y | N | M | - |

- Positive (+): 0
- Negative (-): 4

**Gills = N (6 samples):**
| Length | Gills | Beak | Teeth | Class |
|--------|-------|------|-------|-------|
| 3 | N | Y | M | + |
| 4 | N | Y | M | + |
| 3 | N | Y | F | + |
| 5 | N | Y | M | + |
| 5 | N | Y | F | + |
| 4 | N | Y | F | - |

- Positive (+): 5
- Negative (-): 1

### Step 2: Calculate Gini for each group

**Gills = Y:**
```
p(+) = 0/4 = 0
p(-) = 4/4 = 1

Gini = 1 - (0² + 1²) = 1 - 1 = 0   ← PURE! (all negative)
```

**Gills = N:**
```
p(+) = 5/6 = 0.833
p(-) = 1/6 = 0.167

Gini = 1 - (0.833² + 0.167²)
Gini = 1 - (0.694 + 0.028)
Gini = 1 - 0.722
Gini = 0.278
```

### Step 3: Calculate Weighted Gini

```
Weighted Gini = (4/10) × 0 + (6/10) × 0.278
Weighted Gini = 0 + 0.167
Weighted Gini = 0.167
```

### Step 4: Calculate Gini Gain

```
Gini Gain = Original Gini - Weighted Gini
Gini Gain = 0.5 - 0.167
Gini Gain = 0.333
```

---

## Feature 2: Beak (Y or N)

### Step 1: Split the data by Beak

**Beak = Y (7 samples):**
- Positive (+): 5 (rows 1-5)
- Negative (-): 2 (rows 6-7)

**Beak = N (3 samples):**
- Positive (+): 0
- Negative (-): 3 (rows 8-10)

### Step 2: Calculate Gini for each group

**Beak = Y:**
```
p(+) = 5/7 = 0.714
p(-) = 2/7 = 0.286

Gini = 1 - (0.714² + 0.286²)
Gini = 1 - (0.510 + 0.082)
Gini = 1 - 0.592
Gini = 0.408
```

**Beak = N:**
```
p(+) = 0/3 = 0
p(-) = 3/3 = 1

Gini = 1 - (0² + 1²) = 0   ← PURE! (all negative)
```

### Step 3: Calculate Weighted Gini

```
Weighted Gini = (7/10) × 0.408 + (3/10) × 0
Weighted Gini = 0.286 + 0
Weighted Gini = 0.286
```

### Step 4: Calculate Gini Gain

```
Gini Gain = 0.5 - 0.286 = 0.214
```

---

## Feature 3: Teeth (M or F)

### Step 1: Split the data by Teeth

**Teeth = M (7 samples):**
- Positive (+): 3
- Negative (-): 4

**Teeth = F (3 samples):**
- Positive (+): 2
- Negative (-): 1

### Step 2: Calculate Gini for each group

**Teeth = M:**
```
p(+) = 3/7 = 0.429
p(-) = 4/7 = 0.571

Gini = 1 - (0.429² + 0.571²)
Gini = 1 - (0.184 + 0.326)
Gini = 1 - 0.510
Gini = 0.490
```

**Teeth = F:**
```
p(+) = 2/3 = 0.667
p(-) = 1/3 = 0.333

Gini = 1 - (0.667² + 0.333²)
Gini = 1 - (0.444 + 0.111)
Gini = 1 - 0.555
Gini = 0.444
```

### Step 3: Calculate Weighted Gini

```
Weighted Gini = (7/10) × 0.490 + (3/10) × 0.444
Weighted Gini = 0.343 + 0.133
Weighted Gini = 0.476
```

### Step 4: Calculate Gini Gain

```
Gini Gain = 0.5 - 0.476 = 0.024   ← Very small gain!
```

---

## Feature 4: Length (3, 4, or 5)

Length has 3 possible values, so we split into 3 groups.

### Step 1: Split the data by Length

**Length = 3 (2 samples):**
- Positive (+): 2
- Negative (-): 0

**Length = 4 (4 samples):**
- Positive (+): 1
- Negative (-): 3

**Length = 5 (4 samples):**
- Positive (+): 2
- Negative (-): 2

### Step 2: Calculate Gini for each group

**Length = 3:**
```
Gini = 1 - (1² + 0²) = 0   ← PURE! (all positive)
```

**Length = 4:**
```
p(+) = 1/4 = 0.25
p(-) = 3/4 = 0.75

Gini = 1 - (0.25² + 0.75²)
Gini = 1 - (0.0625 + 0.5625)
Gini = 1 - 0.625
Gini = 0.375
```

**Length = 5:**
```
p(+) = 2/4 = 0.5
p(-) = 2/4 = 0.5

Gini = 1 - (0.5² + 0.5²) = 0.5   ← Maximum impurity
```

### Step 3: Calculate Weighted Gini

```
Weighted Gini = (2/10) × 0 + (4/10) × 0.375 + (4/10) × 0.5
Weighted Gini = 0 + 0.15 + 0.2
Weighted Gini = 0.35
```

### Step 4: Calculate Gini Gain

```
Gini Gain = 0.5 - 0.35 = 0.15
```

---

## Final Comparison: Which Feature Wins?

| Feature | Weighted Gini After Split | Gini Gain |
|---------|---------------------------|-----------|
| **Gills** | 0.167 | **0.333** ← WINNER! |
| Beak | 0.286 | 0.214 |
| Length | 0.350 | 0.150 |
| Teeth | 0.476 | 0.024 |

---

## Conclusion: The First Node

**The best feature to split on is: GILLS**

```
                    [Root Node]
                   All 10 samples
                   Gini = 0.5
                        |
                  Split on Gills
                   /          \
                  /            \
         Gills = Y           Gills = N
         4 samples           6 samples
         0+, 4-              5+, 1-
         Gini = 0            Gini = 0.278
         (PURE!)             (needs more splitting)
```

### Why Gills?

1. **Highest Gini Gain (0.333)** - reduces impurity the most
2. **Creates a pure node** - Gills=Y gives us 4 samples that are ALL negative
3. **Simplifies the tree** - one branch is already done (no more splits needed)

The left branch (Gills = Y) is a **leaf node** - we can immediately classify any sample with Gills=Y as negative (-).

The right branch (Gills = N) still has mixed classes (5 positive, 1 negative), so we would continue splitting on that branch using the remaining features.
