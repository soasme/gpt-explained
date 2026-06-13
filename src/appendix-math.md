# Appendix B: Math Primer

This appendix collects every mathematical concept used in the book — one concise explanation each.

---

## Vectors

A **vector** `v ∈ ℝⁿ` is an ordered list of `n` real numbers: `v = [v₁, v₂, …, vₙ]`.

- **Addition:** `u + v = [u₁+v₁, u₂+v₂, …]`
- **Scalar multiplication:** `c·v = [cv₁, cv₂, …]`
- **L2 norm (length):** `‖v‖ = √(v₁² + v₂² + … + vₙ²)`
- **Unit vector:** `v / ‖v‖` — same direction, length 1

---

## Dot Product

`u · v = u₁v₁ + u₂v₂ + … + uₙvₙ`

- A scalar. Measures **alignment**.
- `u · v = ‖u‖ ‖v‖ cos(θ)` where `θ` is the angle between them.
- `u · v > 0`: same general direction
- `u · v = 0`: perpendicular (orthogonal)
- `u · v < 0`: opposite directions

**Used in:** attention scores, logit computation, cosine similarity.

---

## Matrix Multiplication

For `A ∈ ℝ^{m×n}` and `B ∈ ℝ^{n×p}`:

`(AB)[i,j] = Σₖ A[i,k] · B[k,j]`

Result shape: `[m × p]`. Inner dimensions must match.

- Row `i` of `A` dotted with column `j` of `B`.
- Represent linear transformations.

**Used in:** Q, K, V projections; FFN; unembedding.

---

## Softmax

Converts a vector of real numbers to a probability distribution:

`softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)`

- All outputs in `(0, 1)`, sum to 1.
- Exponential amplifies large values.
- **Numerical stability trick:** subtract `max(z)` first: `exp(zᵢ - max(z)) / Σⱼ exp(zⱼ - max(z))`.

**Used in:** attention weights, output probabilities.

---

## Sigmoid

`σ(x) = 1 / (1 + e⁻ˣ)`

- Output in `(0, 1)`.
- Smooth, monotone. Saturates near 0 and 1.
- `σ(0) = 0.5`, `σ(+∞) → 1`, `σ(-∞) → 0`.

**Used in:** gating mechanisms (SwiGLU, LSTM cells).

---

## ReLU and GELU

**ReLU:** `max(0, x)` — linear for positive inputs, zero for negative.

**GELU:** `x · Φ(x)` where `Φ` is the standard normal CDF.

Fast approximation: `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715x³)))`

- Smooth version of ReLU.
- Used in GPT-2, BERT.

---

## Logarithm and Exponential

`log(x)` is the inverse of `exp(x) = eˣ`.

Key identities:
- `log(ab) = log(a) + log(b)`
- `log(aᵇ) = b·log(a)`
- `exp(a + b) = exp(a) · exp(b)`
- `log(exp(x)) = x`

**Used in:** cross-entropy loss, softmax derivations.

---

## Mean and Variance

For `{x₁, …, xₙ}`:

- **Mean:** `μ = (1/n) Σᵢ xᵢ`
- **Variance:** `σ² = (1/n) Σᵢ (xᵢ − μ)²`
- **Std dev:** `σ = √σ²`
- **Standardized:** `x̂ᵢ = (xᵢ − μ) / σ` — mean 0, std 1.

**Used in:** Layer Normalization.

---

## Gradient and Gradient Descent

The **gradient** of a scalar function `L(θ)` with respect to parameters `θ` is a vector of partial derivatives: `∇L = [∂L/∂θ₁, ∂L/∂θ₂, …]`.

It points in the direction of **steepest increase**. To minimize `L`, move in the **opposite** direction:

```
θ ← θ − α ∇L(θ)
```

Where `α` is the **learning rate**.

In neural networks, gradients are computed via **backpropagation** — the chain rule applied recursively through all layers.

---

## Cross-Entropy Loss

For a true label `y` (an integer, the correct next token) and predicted probabilities `p` (a length-`|V|` vector):

```
L = −log(p[y])
```

- If `p[y] = 1.0` (perfect prediction): `L = 0`
- If `p[y] = 0.01` (very wrong): `L ≈ 4.6`

In practice, averaged over all positions in a batch.

---

## Sine and Cosine

For angle `θ` (in radians):
- `sin(θ)` and `cos(θ)` oscillate between -1 and 1.
- Period: `2π ≈ 6.28`.
- `sin(0) = 0`, `cos(0) = 1`.
- Relationship: `sin²(θ) + cos²(θ) = 1`.

At different frequencies `ω`: `sin(ωθ)` oscillates `ω` times faster.

**Used in:** sinusoidal positional encoding.

---

## Taylor Series (Brief)

Many smooth functions can be approximated as polynomials near a point. The exponential function:

```
eˣ = 1 + x + x²/2! + x³/3! + …
```

This is why `e⁰ = 1`, and `eˣ ≈ 1 + x` for small `x`.

**Used in:** deriving softmax, GELU approximation.
