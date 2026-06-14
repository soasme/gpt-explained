# Chapter 7: The Transformer Block — Putting It Together

> *"Add the new to the old. Normalize. Repeat. That's the residual stream."*

A transformer block combines multi-head attention and the feed-forward network with two crucial wiring techniques: **residual connections** and **layer normalization**. Without these, deep stacks of transformer blocks would fail to train — gradients would vanish or explode.

A GPT model is simply a stack of `N` identical transformer blocks. GPT-2 small: 12 blocks. GPT-3 175B: 96 blocks.

---

## 7.1 The Idea

A GPT model is a stack of identical blocks — GPT-2 small has 12, GPT-3 has 96. Stacking many layers should make the model more powerful, but in practice it causes two problems: information gets distorted as it passes through many transformations, and very deep networks are notoriously hard to train.

A transformer block solves both problems with two simple ideas.

**Residual connections** (also called skip connections): instead of replacing a vector entirely, each sub-layer only adds its output to the original. The input is preserved and combined with the new information. Think of it as writing notes in the margin rather than rewriting the whole page. Each layer only needs to learn *what to change*, not what the whole answer should be. This keeps information flowing cleanly all the way through, even in very deep stacks.

**Layer normalization**: the numbers inside a vector can grow or shrink unpredictably as the model trains. If they drift too far, the training process becomes unstable. Layer normalization resets them to a consistent scale before each sub-layer processes them. It is a housekeeping step: it does not change which direction the vector points, just how large the numbers are.

Together, these two techniques make it possible to stack dozens of transformer blocks without losing control of training.

---

## 7.2 The Math

A single transformer block (Pre-LN variant, used by most modern GPTs):

```
x₁ = x + MHA(LayerNorm(x))       ← attention sub-layer
x₂ = x₁ + FFN(LayerNorm(x₁))    ← FFN sub-layer
```

This is the **Pre-Layer-Norm** architecture. The original "Attention Is All You Need" paper used Post-LN (`output = LayerNorm(x + sub-layer(x))`), but Pre-LN is more stable to train and is used in GPT-2 onwards.

### Layer Normalization

For a vector $x \in \mathbb{R}^d$:

$$
\operatorname{LayerNorm}(x) = \gamma \odot (x - \mu)/\sigma + \beta

where:
  \mu = (1/d) Σ_i x_i            (mean of the vector)
  \sigma = \sqrt{((1/d)} Σ_i (x_i - \mu)^2)  (standard deviation)
  \gamma, \beta \in \mathbb{R}^d                   (learned scale and shift, per-dimension)
  \odot = element-wise multiplication
$$

`LayerNorm` ensures each vector has **mean 0 and variance 1** before the sub-layers process it. The learned $\gamma$ and $\beta$ allow the network to undo the normalization if needed.

> **Math Minute — Variance and Standard Deviation**
> The variance of a set of numbers ${x_1, \ldots, x_n}$ measures how spread out they are:
> $Var = (1/n) Σ_i (x_i - \mu)^2$ where $\mu$ is the mean.
> The standard deviation $\sigma = \sqrt{Var}$ is in the same units as the original values.
> Dividing by $\sigma$ makes the spread equal to 1 — "standardizing."

---

## 7.3 The Residual Stream

A powerful way to understand the GPT architecture is through the lens of the **residual stream** (Elhage et al., Anthropic 2021):

The input embedding is injected into a "stream" — a vector of dimension `d` per token. Each transformer block **reads from** this stream (via attention and FFN) and **adds back to** it (via residual connections). The stream carries information across all blocks.

$$
stream^0 = token_embeddings + positional_embeddings    [T \times d]
stream^1 = stream^0 + MHA(LN(stream^0))
stream^1 = stream^1 + FFN(LN(stream^1))
stream^2 = stream^1 + MHA(LN(stream^1))
stream^2 = stream^2 + FFN(LN(stream^2))
\ldots
streamᴺ = final output
$$

This view clarifies that attention heads and FFN neurons are not arranged sequentially — they are all writing to the same shared workspace.

---

## 7.4 The Matrix: Worked Example

Let `T = 2`, `d = 4`.

Input embedding (position-encoded):
```
x = stream⁰[0] = [1.0, -0.5, 0.8, -0.2]
```

**LayerNorm step:**
```
μ = (1.0 - 0.5 + 0.8 - 0.2) / 4 = 1.1 / 4 = 0.275
σ² = [(1.0-0.275)² + (-0.5-0.275)² + (0.8-0.275)² + (-0.2-0.275)²] / 4
   = [0.526 + 0.600 + 0.275 + 0.226] / 4
   = 1.627 / 4 = 0.407
σ  = √0.407 ≈ 0.638

x_norm = [(1.0-0.275)/0.638, (-0.5-0.275)/0.638, ...]
       = [1.136, -1.215, 0.823, -0.745]

(With γ=1, β=0 for simplicity: LN(x) = x_norm)
```

**MHA produces** (let's say): `mha_out = [0.3, 0.7, -0.1, 0.5]`

**Residual connection:**
```
x₁ = x + mha_out = [1.0+0.3, -0.5+0.7, 0.8-0.1, -0.2+0.5]
   = [1.3, 0.2, 0.7, 0.3]
```

**Second LayerNorm + FFN + residual** (analogously) → $x_2$

The key insight: $x_1$ contains **both** the original information (from `x`) and the new information (from `mha_out`). Nothing is overwritten — the residual stream accumulates.

![Transformer block: attention + residual + LN + FFN + residual](images/ch07-transformer-block.png)

---

## 7.5 The Code: Transformer Block in Scheme

Layer normalization operates on a single vector: compute its mean $\mu$ and standard deviation $\sigma$, then shift each entry to mean 0, variance 1. Learned parameters $\gamma$ and $\beta$ then rescale and shift the normalized result — initialized to $\gamma=1, \beta=0$ so the transformation starts as an identity. This lets the model undo normalization when a sub-layer needs unnormalized values. The small $\epsilon=10^{-5}$ prevents division by zero when a vector is constant.

```scheme
(define (make-matrix rows cols)
  (list rows cols (make-vector (* rows cols) 0.0)))
(define (matrix-rows M) (car M))
(define (matrix-cols M) (cadr M))
(define (matrix-data M) (caddr M))
(define (matrix-ref  M i j)
  (vector-ref  (matrix-data M) (+ (* i (matrix-cols M)) j)))
(define (matrix-set! M i j v)
  (vector-set! (matrix-data M) (+ (* i (matrix-cols M)) j) v))

(define (layer-norm-vec x gamma beta)
  (let* ((n   (vector-length x))
         (eps 1e-5)
         (mu  (let sum ((i 0) (s 0.0))
                (if (= i n) (/ s n)
                    (sum (+ i 1) (+ s (vector-ref x i))))))
         (var (let sum ((i 0) (s 0.0))
                (if (= i n) (/ s n)
                    (let ((d (- (vector-ref x i) mu)))
                      (sum (+ i 1) (+ s (* d d)))))))
         (std (sqrt (+ var eps)))
         (out (make-vector n)))
    (let loop ((i 0))
      (when (< i n)
        (vector-set! out i
          (+ (* (vector-ref gamma i) (/ (- (vector-ref x i) mu) std))
             (vector-ref beta i)))
        (loop (+ i 1))))
    out))
```

`layer-norm-matrix` applies `layer-norm-vec` independently to each row of a $[T \times d]$ matrix. Every token position is normalized separately — the mean and variance are computed within that token's vector, not across the batch. This is what distinguishes layer normalization from batch normalization: the statistics are per-example and per-position, making behavior identical at training and inference time.

```scheme
(define (layer-norm-matrix M gamma beta)
  (let* ((T (matrix-rows M))
         (d (matrix-cols M))
         (R (make-matrix T d)))
    (let loop ((i 0))
      (when (< i T)
        (let* ((row (make-vector d))
               (_ (let cp ((j 0))
                    (when (< j d)
                      (vector-set! row j (matrix-ref M i j))
                      (cp (+ j 1)))))
               (normed (layer-norm-vec row gamma beta)))
          (let cp ((j 0))
            (when (< j d)
              (matrix-set! R i j (vector-ref normed j))
              (cp (+ j 1)))))
        (loop (+ i 1))))
    R))
```

`make-ln-params` returns the pair $(\gamma, \beta)$ with the identity initialization $\gamma_i = 1$, $\beta_i = 0$ for all $i$. At the start of training this means layer norm is a pure normalization with no learned transformation. As training proceeds, $\gamma$ and $\beta$ are updated by gradient descent to whatever values help prediction.

```scheme
(define (make-ln-params d)
  (list (make-vector d 1.0)
        (make-vector d 0.0)))

(define (ln-gamma p) (car p))
(define (ln-beta  p) (cadr p))
```

`make-block-params` collects all four parameter groups for one transformer block: the MHA weights, the FFN weights, and two layer norms. Having all block parameters in one list makes it easy to pass a block around as a value — `make-transformer-stack` will build a list of such values, one per layer.

```scheme
(define (make-block-params d num-heads)
  (list (make-mha-params d num-heads)
        (make-ffn-params d)
        (make-ln-params d)
        (make-ln-params d)))

(define (block-mha  b) (car b))
(define (block-ffn  b) (cadr b))
(define (block-ln1  b) (caddr b))
(define (block-ln2  b) (cadddr b))
```

`transformer-block-forward` is the Pre-LN residual block. The two-line formula $x_1 = x + \text{MHA}(\text{LN}_1(x))$, $x_2 = x_1 + \text{FFN}(\text{LN}_2(x_1))$ becomes two `let` bindings. The residual additions (`matrix-add`) are the key: even if the sub-layers produce near-zero outputs, the original $x$ passes through unchanged. This is what makes hundred-layer stacks trainable — gradients flow directly along the residual path without passing through any nonlinearity. Pre-LN (normalizing before the sub-layer, not after) is more stable than Post-LN for deep stacks.

```scheme
(define (matrix-add A B)
  (let* ((m (matrix-rows A)) (n (matrix-cols A))
         (C (make-matrix m n)))
    (let loop-i ((i 0))
      (when (< i m)
        (let loop-j ((j 0))
          (when (< j n)
            (matrix-set! C i j (+ (matrix-ref A i j) (matrix-ref B i j)))
            (loop-j (+ j 1))))
        (loop-i (+ i 1))))
    C))

(define (transformer-block-forward X block)
  (let* ((ln1     (block-ln1 block))
         (ln2     (block-ln2 block))
         (X1-norm (layer-norm-matrix X (ln-gamma ln1) (ln-beta ln1)))
         (mha-out (car (multi-head-attention X1-norm (block-mha block))))
         (X1      (matrix-add X mha-out))
         (X2-norm (layer-norm-matrix X1 (ln-gamma ln2) (ln-beta ln2)))
         (ffn-out (ffn-forward X2-norm (block-ffn block)))
         (X2      (matrix-add X1 ffn-out)))
    X2))
```

`forward-stack` threads the input matrix $X$ through a list of blocks using `fold` — each block receives the output of the previous one. This is the complete transformer stack: `N` applications of `transformer-block-forward`, chained. The accumulator pattern matches exactly the residual stream metaphor: each block adds its contribution to a growing, evolving representation.

```scheme
(define (forward-stack X blocks)
  (let loop ((x X) (bs blocks))
    (if (null? bs)
        x
        (loop (transformer-block-forward x (car bs)) (cdr bs)))))
```

**Demo.** We build a 3-block stack with $d=16$, $H=4$ heads, and a random $[4 \times 16]$ input. We print the Frobenius norm after each block to confirm the residual stream grows stably rather than exploding or collapsing.

```scheme
(define (matrix-frob-norm M)
  (let ((sum 0.0))
    (let loop-i ((i 0))
      (when (< i (matrix-rows M))
        (let loop-j ((j 0))
          (when (< j (matrix-cols M))
            (let ((v (matrix-ref M i j)))
              (set! sum (+ sum (* v v))))
            (loop-j (+ j 1))))
        (loop-i (+ i 1))))
    (sqrt sum)))

(define (make-random-matrix rows cols)
  (let ((M     (make-matrix rows cols))
        (scale (sqrt (/ 2.0 (+ rows cols)))))
    (let loop-i ((i 0))
      (when (< i rows)
        (let loop-j ((j 0))
          (when (< j cols)
            (matrix-set! M i j (* scale (- (* 2.0 (random 1.0)) 1.0)))
            (loop-j (+ j 1))))
        (loop-i (+ i 1))))
    M))

(let* ((d          16)
       (num-heads  4)
       (num-layers 3)
       (T          4)
       (blocks     (let build ((n num-layers) (acc '()))
                     (if (= n 0) acc
                         (build (- n 1) (cons (make-block-params d num-heads) acc)))))
       (X          (make-random-matrix T d)))
  (display "Input norm: ") (display (matrix-frob-norm X)) (newline)
  (let loop ((x X) (bs blocks) (layer 1))
    (unless (null? bs)
      (let ((x-out (transformer-block-forward x (car bs))))
        (display "After block ") (display layer) (display ": norm = ")
        (display (matrix-frob-norm x-out)) (newline)
        (loop x-out (cdr bs) (+ layer 1))))))
```

Run with `mit-scheme --quiet --load transformer-block.scm`.

---

## 7.6 Why N Blocks?

A single transformer block has limited capacity. Each block:
- Runs one round of attention (all heads) — **mixes information globally**
- Runs one FFN per position — **processes each token's updated representation**

But complex language tasks require many rounds of computation. By stacking `N` blocks, the model builds increasingly abstract representations:

- Early blocks: local patterns, syntax, common phrases
- Middle blocks: semantic roles, entity tracking
- Late blocks: task-specific reasoning, pragmatics

The residual stream carries information across all blocks. Later blocks can read from and write to everything earlier blocks computed.

---

## 7.7 Key Takeaways

- A transformer block = `MHA → residual → LN → FFN → residual → LN` (Pre-LN ordering).
- **Residual connections** let gradients flow directly and prevent vanishing gradients.
- **Layer normalization** keeps activations at mean 0, std 1 within each sub-layer.
- The **residual stream** is the shared workspace that all blocks read from and write to.
- GPT stacks `N` identical blocks; capacity scales with `N` and `d`.

> **What's next?** After `N` blocks, we have a final matrix $X_final \in \mathbb{R}^{T\times d}$. Each row is a rich representation of the corresponding token in context. The last step: turn the final vector for position `T` into a probability distribution over the vocabulary — that's **vocabulary projection** in Chapter 8.


---

![Full transformer block (Pre-LN) — LayerNorm before each sub-layer, residual connections around both.](images/ch07_block.png)

*Full transformer block (Pre-LN) — LayerNorm before each sub-layer, residual connections around both.*

![The residual stream — information flows unimpeded through addition, gradients flow backward.](images/ch07_residual.png)

*The residual stream — information flows unimpeded through addition, gradients flow backward.*

