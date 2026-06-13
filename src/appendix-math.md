# Appendix B: Math Primer

This appendix collects every mathematical concept used in the book — one concise explanation each.

---

## Vectors

A **vector** $v \in \mathbb{R}^n$ is an ordered list of `n` real numbers: $v = [v_1, v_2, \ldots, v_n]$.

- **Addition:** $u + v = [u_{1+v1}, u_{2+v2}, \ldots]$
- **Scalar multiplication:** $c\cdot v = [cv_1, cv_2, \ldots]$
- **L2 norm (length):** $\|v\| = \sqrt{(v_1^2 + v_2^2 + \ldots + v_n^2)}$
- **Unit vector:** `v / ‖v‖` — same direction, length 1

---

## Dot Product

$u \cdot v = u_{1v1} + u_{2v2} + \ldots + u_{nvn}$

- A scalar. Measures **alignment**.
- $u \cdot v = \|u\| \|v\| \cos(\theta)$ where `θ` is the angle between them.
- $u \cdot v > 0$: same general direction
- $u \cdot v = 0$: perpendicular (orthogonal)
- $u \cdot v < 0$: opposite directions

**Used in:** attention scores, logit computation, cosine similarity.

---

## Matrix Multiplication

For $A \in \mathbb{R}^{m\times n}$ and $B \in \mathbb{R}^{n\times p}$:

$(AB)[i,j] = Σ_k A[i,k] \cdot B[k,j]$

Result shape: $[m \times p]$. Inner dimensions must match.

- Row `i` of `A` dotted with column `j` of `B`.
- Represent linear transformations.

**Used in:** Q, K, V projections; FFN; unembedding.

---

## Softmax

Converts a vector of real numbers to a probability distribution:

$\operatorname{softmax}(z)_i = \exp(z_i) / Σ_j \exp(z_j)$

- All outputs in `(0, 1)`, sum to 1.
- Exponential amplifies large values.
- **Numerical stability trick:** subtract `max(z)` first: $\exp(z_i - max(z)) / Σ_j \exp(z_j - max(z))$.

**Used in:** attention weights, output probabilities.

---

## Sigmoid

$\sigma(x) = 1 / (1 + e^{-}ˣ)$

- Output in `(0, 1)`.
- Smooth, monotone. Saturates near 0 and 1.
- $\sigma(0) = 0.5$, $\sigma(+\infty) \to 1$, $\sigma(-\infty) \to 0$.

**Used in:** gating mechanisms (SwiGLU, LSTM cells).

---

## ReLU and GELU

**ReLU:** `max(0, x)` — linear for positive inputs, zero for negative.

**GELU:** $x \cdot \Phi(x)$ where $\Phi$ is the standard normal CDF.

Fast approximation: $0.5\cdot x\cdot(1 + \tanh(\sqrt{(2/\pi)}\cdot(x + 0.044715x^3)))$

- Smooth version of ReLU.
- Used in GPT-2, BERT.

---

## Logarithm and Exponential

`log(x)` is the inverse of `exp(x) = eˣ`.

Key identities:
- `log(ab) = log(a) + log(b)`
- $\log(aᵇ) = b\cdot \log(a)$
- $\exp(a + b) = \exp(a) \cdot \exp(b)$
- `log(exp(x)) = x`

**Used in:** cross-entropy loss, softmax derivations.

---

## Mean and Variance

For ${x_1, \ldots, x_n}$:

- **Mean:** $\mu = (1/n) Σ_i x_i$
- **Variance:** $\sigma^2 = (1/n) Σ_i (x_i - \mu)^2$
- **Std dev:** $\sigma = \sqrt{1}\sigma^2$
- **Standardized:** $x̂_i = (x_i - \mu) / \sigma$ — mean 0, std 1.

**Used in:** Layer Normalization.

---

## Gradient and Gradient Descent

The **gradient** of a scalar function `L(θ)` with respect to parameters `θ` is a vector of partial derivatives: $∇L = [\partial L/\partial\theta_1, \partial L/\partial\theta_2, \ldots]$.

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
- If `p[y] = 0.01` (very wrong): $L \approx 4.6$

In practice, averaged over all positions in a batch.

---

## Sine and Cosine

For angle `θ` (in radians):
- `sin(θ)` and `cos(θ)` oscillate between -1 and 1.
- Period: $2\pi \approx 6.28$.
- `sin(0) = 0`, `cos(0) = 1`.
- Relationship: $sin^2(\theta) + cos^2(\theta) = 1$.

At different frequencies $\omega$: $\sin(\omega\theta)$ oscillates $\omega$ times faster.

**Used in:** sinusoidal positional encoding.

---

## Taylor Series (Brief)

Many smooth functions can be approximated as polynomials near a point. The exponential function:

$$
eˣ = 1 + x + x^2/2! + x^3/3! + \ldots
$$

This is why `e⁰ = 1`, and $eˣ \approx 1 + x$ for small `x`.

**Used in:** deriving softmax, GELU approximation.
