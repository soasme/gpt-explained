# Chapter 7: The Transformer Block — Putting It Together

> *"Add the new to the old. Normalize. Repeat. That's the residual stream."*

A transformer block combines multi-head attention and the feed-forward network with two crucial wiring techniques: **residual connections** and **layer normalization**. Without these, deep stacks of transformer blocks would fail to train — gradients would vanish or explode.

A GPT model is simply a stack of `N` identical transformer blocks. GPT-2 small: 12 blocks. GPT-3 175B: 96 blocks.

---

## 7.1 The Idea

The core insight of residual networks (He et al., 2015, ResNets in computer vision): instead of learning `f(x)`, learn the **residual** `f(x) - x`, then add back the input:

```
output = x + f(x)
```

Why does this help? During backpropagation, gradients flow directly through the `+` operation. Even if the sub-network `f` produces near-zero gradients, the gradient "highway" through the skip connection keeps training alive. This makes it possible to train networks hundreds of layers deep.

**Layer normalization** stabilizes the activations within each sub-layer. Without it, the distributions of activations shift as the model trains, making training unstable.

---

## 7.2 The Math

A single transformer block (Pre-LN variant, used by most modern GPTs):

```
x₁ = x + MHA(LayerNorm(x))       ← attention sub-layer
x₂ = x₁ + FFN(LayerNorm(x₁))    ← FFN sub-layer
```

This is the **Pre-Layer-Norm** architecture. The original "Attention Is All You Need" paper used Post-LN (`output = LayerNorm(x + sub-layer(x))`), but Pre-LN is more stable to train and is used in GPT-2 onwards.

### Layer Normalization

For a vector `x ∈ ℝᵈ`:

```
LayerNorm(x) = γ ⊙ (x - μ)/σ + β

where:
  μ = (1/d) Σᵢ xᵢ            (mean of the vector)
  σ = √((1/d) Σᵢ (xᵢ - μ)²)  (standard deviation)
  γ, β ∈ ℝᵈ                   (learned scale and shift, per-dimension)
  ⊙ = element-wise multiplication
```

`LayerNorm` ensures each vector has **mean 0 and variance 1** before the sub-layers process it. The learned `γ` and `β` allow the network to undo the normalization if needed.

> **Math Minute — Variance and Standard Deviation**
> The variance of a set of numbers `{x₁, …, xₙ}` measures how spread out they are:
> `Var = (1/n) Σᵢ (xᵢ − μ)²` where `μ` is the mean.
> The standard deviation `σ = √Var` is in the same units as the original values.
> Dividing by `σ` makes the spread equal to 1 — "standardizing."

---

## 7.3 The Residual Stream

A powerful way to understand the GPT architecture is through the lens of the **residual stream** (Elhage et al., Anthropic 2021):

The input embedding is injected into a "stream" — a vector of dimension `d` per token. Each transformer block **reads from** this stream (via attention and FFN) and **adds back to** it (via residual connections). The stream carries information across all blocks.

```
stream⁰ = token_embeddings + positional_embeddings    [T × d]
stream¹ = stream⁰ + MHA(LN(stream⁰))
stream¹ = stream¹ + FFN(LN(stream¹))
stream² = stream¹ + MHA(LN(stream¹))
stream² = stream² + FFN(LN(stream²))
…
streamᴺ = final output
```

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

**Second LayerNorm + FFN + residual** (analogously) → `x₂`

The key insight: `x₁` contains **both** the original information (from `x`) and the new information (from `mha_out`). Nothing is overwritten — the residual stream accumulates.

![Transformer block: attention + residual + LN + FFN + residual](images/ch07-transformer-block.png)

---

## 7.5 The Code: Transformer Block in Lisp

```lisp
;;;; transformer-block.lisp
;;;; One complete transformer block with Pre-LN architecture

;; (load "attention.lisp")
;; (load "multi-head-attention.lisp")
;; (load "feed-forward-network.lisp")

;;; ─── Layer Normalization ─────────────────────────────────────────

(defun layer-norm-1d (x gamma beta eps)
  "LayerNorm for a single vector X.
   gamma, beta : learned scale and shift (same length as x).
   eps         : small value for numerical stability (e.g., 1e-5)."
  (let* ((n (length x))
         (mu (/ (reduce #'+ x) n))
         (var (/ (reduce #'+ (map 'vector
                                  (lambda (xi) (expt (- xi mu) 2))
                                  x))
                 n))
         (std (sqrt (+ var eps)))
         (out (make-array n :element-type 'single-float)))
    (dotimes (i n out)
      (setf (aref out i)
            (+ (* (aref gamma i)
                  (/ (- (aref x i) mu) std))
               (aref beta i))))))

(defun layer-norm-matrix (M gamma beta &optional (eps 1e-5))
  "Apply LayerNorm independently to each row of M [T × d]."
  (destructuring-bind (T-len d) (array-dimensions M)
    (let ((result (make-array (list T-len d) :element-type 'single-float)))
      (dotimes (i T-len result)
        (let* ((row (make-array d :element-type 'single-float))
               (normed nil))
          ;; Extract row
          (dotimes (j d) (setf (aref row j) (aref M i j)))
          ;; Normalize
          (setf normed (layer-norm-1d row gamma beta eps))
          ;; Write back
          (dotimes (j d) (setf (aref result i j) (aref normed j))))))))

(defun make-ln-params (d)
  "Create LayerNorm parameters: γ=1, β=0 (identity init)."
  (list :gamma (make-array d :element-type 'single-float :initial-element 1.0)
        :beta  (make-array d :element-type 'single-float :initial-element 0.0)))

;;; ─── Matrix addition (for residual connections) ──────────────────

;; mat-add is defined in attention.lisp

;;; ─── Transformer Block Parameters ───────────────────────────────

(defun make-block-params (d num-heads)
  "Create all parameters for one transformer block."
  (list :mha  (make-mha-params d num-heads)
        :ffn  (make-ffn-params d 4)
        :ln1  (make-ln-params d)
        :ln2  (make-ln-params d)))

;;; ─── Transformer Block Forward Pass ─────────────────────────────

(defun transformer-block-forward (X params)
  "One Pre-LN transformer block:
   x₁ = x  + MHA(LayerNorm₁(x))
   x₂ = x₁ + FFN(LayerNorm₂(x₁))
   Returns x₂ [T × d]."
  (let* ((mha-params (getf params :mha))
         (ffn-params (getf params :ffn))
         (ln1        (getf params :ln1))
         (ln2        (getf params :ln2))
         ;; Sub-layer 1: MHA with pre-norm + residual
         (X-normed-1 (layer-norm-matrix X
                                        (getf ln1 :gamma)
                                        (getf ln1 :beta)))
         (mha-out (multi-head-attention X-normed-1 mha-params))
         (X1 (mat-add X mha-out))
         ;; Sub-layer 2: FFN with pre-norm + residual
         (X-normed-2 (layer-norm-matrix X1
                                        (getf ln2 :gamma)
                                        (getf ln2 :beta)))
         (ffn-out (ffn-forward X-normed-2 ffn-params))
         (X2 (mat-add X1 ffn-out)))
    X2))

;;; ─── Stack of Transformer Blocks ─────────────────────────────────

(defun make-transformer-stack (num-layers d num-heads)
  "Create a list of NUM-LAYERS transformer block parameter plists."
  (loop repeat num-layers
        collect (make-block-params d num-heads)))

(defun forward-transformer-stack (X block-param-list)
  "Run X through a stack of transformer blocks in sequence.
   Returns the final [T × d] output."
  (reduce #'transformer-block-forward block-param-list
          :initial-value X))

;;; ─── Verify residual stream integrity ────────────────────────────

(defun matrix-norm (M)
  "Frobenius norm of M: √(Σ m²ᵢⱼ)"
  (sqrt (let ((sum 0.0))
          (destructuring-bind (r c) (array-dimensions M)
            (dotimes (i r sum)
              (dotimes (j c)
                (incf sum (expt (aref M i j) 2))))))))

;;; ─── Demo ─────────────────────────────────────────────────────────

(let* ((d 16)
       (num-heads 4)
       (num-layers 3)
       (T-len 4)
       (blocks (make-transformer-stack num-layers d num-heads))
       ;; Start with random token+position embeddings
       (X (let ((m (make-array (list T-len d) :element-type 'single-float)))
            (dotimes (i T-len m)
              (dotimes (j d)
                (setf (aref m i j) (* 0.02 (- (random 100.0) 50.0))))))))
  (format t "Input norm: ~,4f~%" (matrix-norm X))
  ;; Pass through each block, reporting norm
  (let ((stream X))
    (loop for params in blocks
          for layer from 1 do
      (setf stream (transformer-block-forward stream params))
      (format t "After block ~a: norm = ~,4f~%" layer (matrix-norm stream)))
    (format t "~%Final stream shape: [~a × ~a]~%"
            (array-dimension stream 0) (array-dimension stream 1))))
```

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

> **What's next?** After `N` blocks, we have a final matrix `X_final ∈ ℝ^{T×d}`. Each row is a rich representation of the corresponding token in context. The last step: turn the final vector for position `T` into a probability distribution over the vocabulary — that's **vocabulary projection** in Chapter 8.
