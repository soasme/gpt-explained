# Chapter 6: Feed-Forward Network — The Model's Memory

> *"Attention decides what to look at. The feed-forward network decides what to think about it."*

After multi-head attention has blended information across positions, each token's vector passes through a **feed-forward network (FFN)**. This is a simple two-layer MLP — the same network, applied independently to every position.

While attention does the *routing* (mixing information across tokens), the FFN does the *processing* (transforming each token's representation). Remarkably, experiments by Geva et al. (2021) showed that the FFN layers act as "key-value memories" — the first layer's neurons trigger on specific input patterns, and the second layer retrieves associated information.

---

## 6.1 The Idea

Attention is about *communication* — each token collects information from other tokens. The feed-forward network is about *computation* — each token processes what it just collected, entirely on its own.

After attention, every token's vector has been updated with context from the surrounding words. The FFN now takes each updated vector and runs it through a private transformation: expand it into a much larger space, filter it through a non-linearity that decides which features are "active," then compress it back to the original size. No information crosses between tokens here — every position is handled independently with the same set of weights.

Why expand and then compress? The expansion gives the model room to express a large number of potential features at once. The non-linearity then selects which combinations matter and switches the rest off. The compression packages the result back into the model's standard vector size.

This expand-filter-compress pipeline is where most of the model's factual knowledge lives. Research has shown that individual neurons in the FFN's expanded space can be associated with specific concepts ("Paris," "past tense," "chemical element"). The model does not store facts in attention — it stores them here.

The same FFN weights are applied to every token position in the sequence: position 1 and position 100 go through identical transformations. Only attention sees position.

---

## 6.2 The Math

> **Math Minute — ReLU and GELU**
> A **non-linearity** is a function that makes the network capable of learning complex patterns beyond linear mappings.
>
> `ReLU(x) = max(0, x)` — simple: negative values → 0, positive values unchanged.
>
> $\operatorname{GELU}(x) = x \cdot \Phi(x)$ where $\Phi$ is the standard normal CDF. GELU is smooth and probabilistic: instead of a hard cut at zero, it "softly" gates the input. GPT uses GELU. LLaMA and most modern models use `SwiGLU`, a gated variant.
>
> `Sigmoid(x) = 1/(1+e⁻ˣ)` squashes any real number to (0,1). Used in gates.

For a sequence input $X \in \mathbb{R}^{T\times d}$, the FFN applies identically to each row:

$$
FFN(X) = \operatorname{GELU}(X W_1 + 1\cdot b_1^{\top}) W_2 + 1\cdot b_2^{\top}
$$

Where $1\cdot b_1^{\top}$ broadcasts the bias across all `T` positions.

Breaking it down:
$$
H = X W_1 + 1\cdot b_1^{\top}       \in \mathbb{R}^{T \times d_ff}     (linear expand)
H' = \operatorname{GELU}(H)             \in \mathbb{R}^{T \times d_ff}     (activation)
Y = H' W_2 + 1\cdot b_2^{\top}       \in \mathbb{R}^{T \times d}        (linear contract)
$$

**Parameter count:**
- $W_1$: $d \times d_ff = d \times 4d = 4d^2$
- $W_2$: $d_ff \times d = 4d^2$
- Biases: $d_ff + d \approx 5d$
- Total: $~8d^2$ — this is **twice** the parameter count of all attention weight matrices combined!

For GPT-2 small (`d=768`): FFN has `~4.7M` parameters per layer (vs `~2.4M` for attention). The FFN is not an afterthought — it dominates model capacity.

---

## 6.3 GELU Deep Dive

The exact GELU formula is:

$$
\operatorname{GELU}(x) = x \cdot \frac{1}{2} \cdot \left[1 + \operatorname{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]
$$

where $\operatorname{erf}$ is the Gauss error function:

$$
\operatorname{erf}(z) = \frac{2}{\sqrt{\pi}} \int_0^z e^{-t^2} \, dt
$$

A fast approximation (used in practice):

$$
\operatorname{GELU}(x) \approx 0.5 \cdot x \cdot (1 + \tanh(\sqrt{(2/\pi)} \cdot (x + 0.044715 \cdot x^3)))
$$

Key behaviors:
- `GELU(0) = 0`
- For large positive `x`: $\operatorname{GELU}(x) \approx x$ (passes through)
- For large negative `x`: $\operatorname{GELU}(x) \approx 0$ (suppressed)
- Smooth everywhere — gradient flows through during training

---

## 6.4 The Matrix: Worked Example

Let `T = 2`, `d = 4`, `d_ff = 8` (2× for brevity, usually 4×).

Input (after attention):
```
X = [[ 0.5,  1.2, -0.3,  0.8],
     [-0.1,  0.7,  0.9, -0.4]]    (2×4)
```

Weight matrices (simplified for hand-calculation):
$$
W_1 (4\times 8): columns are random, let's say it expands X to 8 dims
b_1 (8): all zeros for simplicity

W_2 (8\times 4): contracts back
b_2 (4): all zeros
$$

Let's trace just row 0 of X: `x = [0.5, 1.2, -0.3, 0.8]`

**Step 1 — Expand:** $h = x W_1$
```
(with a hypothetical W₁ this produces a 8-element vector)
h = [1.1, -0.4, 0.8, 2.1, -0.3, 0.6, 1.5, -0.8]
```

**Step 2 — GELU:**
```
GELU(1.1)  ≈ 0.95
GELU(-0.4) ≈ -0.12
GELU(0.8)  ≈ 0.64
GELU(2.1)  ≈ 2.06
GELU(-0.3) ≈ -0.11
GELU(0.6)  ≈ 0.44
GELU(1.5)  ≈ 1.40
GELU(-0.8) ≈ -0.17

h' = [0.95, -0.12, 0.64, 2.06, -0.11, 0.44, 1.40, -0.17]
```

**Step 3 — Contract:** $y = h' W_2$
```
(with a hypothetical W₂ this produces a 4-element vector)
y ≈ [0.7, 1.1, -0.2, 0.9]
```

The output `y` is a transformed version of the input vector — same dimensionality, different values. The transformation was learned to improve next-token prediction.

![FFN: expand then contract with GELU non-linearity](images/ch06-ffn.png)

---

## 6.5 The Code: FFN in Scheme

GELU is the activation function used in GPT-2. Its tanh approximation is $\text{GELU}(x) \approx 0.5 \cdot x \cdot (1 + \tanh(\sqrt{2/\pi} \cdot (x + 0.044715 x^3)))$. For large positive $x$, GELU approaches $x$ — the value passes through. For large negative $x$, it approaches 0 — the value is suppressed. Unlike ReLU, the transition at zero is smooth: the function is differentiable everywhere, so gradients flow cleanly through negative values during training.

```scheme
(define (gelu x)
  (let* ((c     (sqrt (/ 2.0 3.141592653589793)))
         (inner (+ x (* 0.044715 x x x))))
    (* x 0.5 (+ 1.0 (tanh (* c inner))))))
```

`gelu-matrix` applies the scalar GELU to every element of a matrix, producing a result of the same shape. This is the standard pattern for applying an element-wise function across a matrix: allocate a fresh result, iterate over all positions, apply, store.

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

(define (gelu-matrix M)
  (let* ((rows (matrix-rows M))
         (cols (matrix-cols M))
         (R    (make-matrix rows cols)))
    (let loop-i ((i 0))
      (when (< i rows)
        (let loop-j ((j 0))
          (when (< j cols)
            (matrix-set! R i j (gelu (matrix-ref M i j)))
            (loop-j (+ j 1))))
        (loop-i (+ i 1))))
    R))
```

`add-bias` broadcasts a bias vector $b$ across all rows of a matrix $M$. In the FFN, $b_1$ has length $d_{ff}$ and must be added to every row of the $[T \times d_{ff}]$ intermediate result. Broadcasting is nothing more than adding $b[j]$ to $M[i,j]$ for every row $i$ — the same column offset applied uniformly. This is the affine part of the linear layer: without bias, the transformation $XW_1$ can only represent functions through the origin.

```scheme
(define (add-bias M b)
  (let* ((rows (matrix-rows M))
         (cols (matrix-cols M))
         (R    (make-matrix rows cols)))
    (let loop-i ((i 0))
      (when (< i rows)
        (let loop-j ((j 0))
          (when (< j cols)
            (matrix-set! R i j (+ (matrix-ref M i j) (vector-ref b j)))
            (loop-j (+ j 1))))
        (loop-i (+ i 1))))
    R))
```

`make-ffn-params` allocates and initializes the four parameters: $W_1$ expands the model dimension $d$ to $4d$; $W_2$ contracts it back. The inner dimension $4d$ is where the FFN's capacity lives — neurons there fire on specific input patterns and write associated information back. Biases start at zero; Xavier scaling sets the weight magnitudes.

```scheme
(define (make-ffn-params d)
  (let* ((d-ff   (* 4 d))
         (scale1 (sqrt (/ 2.0 (+ d d-ff))))
         (scale2 (sqrt (/ 2.0 (+ d-ff d))))
         (W1 (let ((M (make-matrix d d-ff)))
               (let loop-i ((i 0))
                 (when (< i d)
                   (let loop-j ((j 0))
                     (when (< j d-ff)
                       (matrix-set! M i j (* scale1 (- (* 2.0 (random 1.0)) 1.0)))
                       (loop-j (+ j 1))))
                   (loop-i (+ i 1))))
               M))
         (b1 (make-vector d-ff 0.0))
         (W2 (let ((M (make-matrix d-ff d)))
               (let loop-i ((i 0))
                 (when (< i d-ff)
                   (let loop-j ((j 0))
                     (when (< j d)
                       (matrix-set! M i j (* scale2 (- (* 2.0 (random 1.0)) 1.0)))
                       (loop-j (+ j 1))))
                   (loop-i (+ i 1))))
               M))
         (b2 (make-vector d 0.0)))
    (list W1 b1 W2 b2)))

(define (ffn-w1 p) (car p))
(define (ffn-b1 p) (cadr p))
(define (ffn-w2 p) (caddr p))
(define (ffn-b2 p) (cadddr p))
```

`ffn-forward` composes the operations in the order prescribed by the formula $H = \text{GELU}(XW_1 + b_1) \cdot W_2 + b_2$. Each step is one procedure call; the composition of those calls is the FFN. The input and output both have shape $[T \times d]$; the intermediate $H$ has shape $[T \times 4d]$. No information crosses between token positions here — that is attention's job.

```scheme
(define (matrix-multiply A B)
  (let* ((m (matrix-rows A)) (k (matrix-cols A)) (n (matrix-cols B))
         (C (make-matrix m n)))
    (let loop-i ((i 0))
      (when (< i m)
        (let loop-j ((j 0))
          (when (< j n)
            (let loop-k ((p 0) (acc 0.0))
              (if (= p k)
                  (matrix-set! C i j acc)
                  (loop-k (+ p 1)
                          (+ acc (* (matrix-ref A i p)
                                    (matrix-ref B p j))))))
            (loop-j (+ j 1))))
        (loop-i (+ i 1))))
    C))

(define (ffn-forward X params)
  (let* ((H     (add-bias (matrix-multiply X (ffn-w1 params)) (ffn-b1 params)))
         (H-act (gelu-matrix H))
         (Y     (add-bias (matrix-multiply H-act (ffn-w2 params)) (ffn-b2 params))))
    Y))
```

**Demo.** We build parameters for $d=8$ (inner dimension $4d=32$), create a random $[3 \times 8]$ input, run the forward pass, and print the result. The output has the same shape as the input — the FFN transforms but does not resize.

```scheme
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

(let* ((d      8)
       (T      3)
       (params (make-ffn-params d))
       (X      (make-random-matrix T d))
       (Y      (ffn-forward X params)))
  (display "FFN input X [3x8]:") (newline)
  (let row ((i 0))
    (when (< i T)
      (display "  [")
      (let col ((j 0))
        (when (< j d)
          (display (matrix-ref X i j)) (display " ")
          (col (+ j 1))))
      (display "]") (newline)
      (row (+ i 1))))
  (newline)
  (display "FFN output Y [3x8]:") (newline)
  (let row ((i 0))
    (when (< i T)
      (display "  [")
      (let col ((j 0))
        (when (< j d)
          (display (matrix-ref Y i j)) (display " ")
          (col (+ j 1))))
      (display "]") (newline)
      (row (+ i 1)))))
```

Run with `mit-scheme --quiet --load feed-forward.scm`.

---

## 6.6 The FFN as a Key-Value Memory

A striking insight from Geva et al. (2021): the FFN layers act like **associative memories**.

The first layer $W_1$ stores "keys" — each column of $W_1$ is a pattern to match against the input. GELU fires when the input matches. The second layer $W_2$ stores "values" — the corresponding information to retrieve.

Concretely:
$$
h = \operatorname{GELU}(x W_1)
$$

where each component $h_k \approx x \cdot W_1[:,k]$ — a dot-product "match score" for pattern $k$. High $h_k$ means "input matches pattern $k$."

$$
y = h W_2
$$

Pull out value $W_2[k,:]$ weighted by $h_k$.

This means the model can learn: "if the input contains patterns related to France, activate neuron 47, which retrieves 'Paris' associations."

---

## 6.7 Key Takeaways

- The FFN is a two-layer MLP applied **identically to every position**: $FFN(x) = \operatorname{GELU}(xW_1 + b_1) W_2 + b_2$.
- The inner dimension `d_ff = 4d` means the FFN has **more parameters than the attention layers**.
- GELU is a smooth activation function that "softly gates" negative values.
- The FFN acts as an **associative memory**: first layer matches patterns, second layer retrieves associated information.
- No cross-position communication happens here — that's attention's job.

> **What's next?** We now have all the pieces: attention that mixes information, and an FFN that processes each token's vector. How are they wired together? With **residual connections** and **layer normalization** to form the **transformer block** — Chapter 7.


---

![Feed-forward network — expand → GELU → contract, with GELU vs ReLU activation comparison.](images/ch06_ffn.png)

*Feed-forward network — expand → GELU → contract, with GELU vs ReLU activation comparison.*

![The FFN as associative memory — keys and values stored in the weight matrices.](images/ch06_memory.png)

*The FFN as associative memory — keys and values stored in the weight matrices.*

