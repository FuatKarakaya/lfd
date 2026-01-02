# Factor Analysis (FA)

## Overview

Factor Analysis is a statistical method used to identify **hidden (latent) factors** that explain correlations among observed variables. Unlike PCA, which focuses on total variance, FA specifically models the **shared variance** between variables.

---

## Core Concept

FA assumes that your observed measurements are **caused by** a smaller number of underlying, unobservable factors, plus some unique noise for each variable.

### The Generative Model

$$\mathbf{x} = \mathbf{L}\mathbf{f} + \boldsymbol{\epsilon}$$

Where:
- $\mathbf{x}$ = vector of observed variables (what you measure)
- $\mathbf{f}$ = vector of latent factors (what you're trying to discover)
- $\mathbf{L}$ = factor loading matrix (how each factor influences each variable)
- $\boldsymbol{\epsilon}$ = unique variance / error term (noise specific to each variable)

---

## Key Terms

| Term | Definition |
|------|------------|
| **Latent Factor** | An unobserved variable that influences multiple observed variables |
| **Factor Loading** | The correlation/weight between a factor and an observed variable |
| **Communality** | The proportion of a variable's variance explained by the common factors |
| **Uniqueness** | The proportion of variance specific to that variable alone (not shared) |

---

## Intuitive Example: Intelligence Testing

Imagine you administer 10 different cognitive tests to students:
- Vocabulary
- Reading comprehension
- Verbal analogies
- Arithmetic
- Algebra
- Geometry
- Pattern recognition
- Spatial rotation
- Memory span
- Processing speed

You notice that scores on the first three tests are highly correlated with each other, and scores on tests 4-6 are also correlated with each other.

**FA might reveal:**
- **Factor 1 (Verbal Ability)**: Strongly loads on vocabulary, reading, verbal analogies
- **Factor 2 (Mathematical Ability)**: Strongly loads on arithmetic, algebra, geometry
- **Factor 3 (Fluid Intelligence)**: Loads on pattern recognition, spatial rotation

These factors weren't directly measured—they're *inferred* from the correlation structure.

---

## How Factor Analysis Works

### Step 1: Start with the Correlation Matrix
Examine how all variables correlate with each other.

### Step 2: Extract Factors
Use methods like:
- **Principal Axis Factoring**
- **Maximum Likelihood Estimation**

### Step 3: Determine Number of Factors
Common criteria:
- Eigenvalue > 1 (Kaiser criterion)
- Scree plot inspection
- Parallel analysis
- Theoretical considerations

### Step 4: Rotate Factors (Optional but Common)
Rotation makes factors more interpretable:
- **Varimax** (orthogonal): Keeps factors uncorrelated
- **Promax** (oblique): Allows factors to correlate

### Step 5: Interpret Factor Loadings
High loadings (typically |loading| > 0.4) indicate which variables "belong" to which factor.

---

## FA vs. PCA: Key Differences

| Aspect | PCA | Factor Analysis |
|--------|-----|-----------------|
| **Goal** | Maximize explained variance | Model correlation structure |
| **Assumption** | Components are linear combinations | Observed = factors + unique error |
| **Variance** | Explains total variance | Explains shared variance only |
| **Unique variance** | Not modeled separately | Explicitly modeled as $\epsilon$ |
| **Interpretation** | Mathematical constructs | Often interpreted as real causes |
| **Use case** | Data reduction, preprocessing | Theory building, understanding structure |

### Visual Intuition

```
PCA:          Observed Variables ──────► Components (mathematical summary)

FA:           Latent Factors ──────► Observed Variables + Unique Error
              (underlying causes)     (what we actually measure)
```

---

## When to Use Factor Analysis

✅ **Good Use Cases:**
- Psychology: Identifying personality traits from questionnaire items
- Education: Understanding underlying abilities from test scores
- Marketing: Finding latent consumer preferences from survey responses
- Medicine: Grouping symptoms into underlying syndromes

❌ **Not Ideal When:**
- You just want to reduce dimensions for another algorithm (use PCA)
- You have very few variables
- Variables don't have meaningful correlations
- You need a deterministic transformation

---

## Assumptions of Factor Analysis

1. **Linearity**: Relationships between factors and variables are linear
2. **No multicollinearity**: Variables shouldn't be perfectly correlated
3. **Adequate sample size**: Typically N > 100, or N:p ratio of at least 5:1
4. **Normality**: For maximum likelihood estimation (less critical for principal axis)
5. **Correlation exists**: Variables should be meaningfully correlated

---

## Summary

Factor Analysis is a powerful tool for **discovering latent structure** in data. It goes beyond mere dimensionality reduction by proposing a **causal model**: hidden factors generate the observed correlations. This makes FA particularly valuable in fields like psychology and social sciences, where we often believe that abstract constructs (intelligence, personality, attitudes) underlie measurable behaviors.

---

## Further Reading

- Thurstone, L.L. (1947). *Multiple Factor Analysis*
- Thompson, B. (2004). *Exploratory and Confirmatory Factor Analysis*
- Fabrigar, L.R. & Wegener, D.T. (2012). *Exploratory Factor Analysis*
