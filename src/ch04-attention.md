# Chapter 4: Attention — Tokens Talking to Each Other

> *"Every token asks: 'Who here is relevant to me?' Every other token answers. The answer is a weighted blend of their values."*

Attention is the engine of the transformer. It is the mechanism that lets each token look at all (or some) of the other tokens and decide which ones to borrow information from. Without attention, each token would be processed independently, oblivious to context. With attention, the word "bank" can tell whether the surrounding tokens are about rivers or finance.

This chapter builds scaled dot-product attention from first principles.

---

## 4.1 The Idea

Imagine you are at a library. You arrive with a **query**: "I'm looking for books about Renaissance painting." Every book in the library has a **key** on its spine — a short descriptor of its contents. You compare your query against every key, giving each book a **score**. Then you take a weighted blend of the books' **values** (actual content) — spending more time on high-scoring books.

In the attention mechanism:

- **Query (Q):** "What am I looking for?" — produced from the current token.
- **Key (K):** "What do I contain?" — produced from every token in the sequence.
- **Value (V):** "What information can I give?" — produced from every token in the sequence.

The output for each token is a **weighted sum** of all value vectors, where the weights come from comparing that token's query against every token's key.

> **Math Minute — Softmax**
> `softmax(z)` turns a vector of raw scores into a probability distribution.
> For vector $z = [z_1, z_2, \ldots, z_n]$:
> ```
> softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)
> ```
> Output values are all positive and sum to 1. Large scores get amplified; small scores get suppressed. The largest score "wins" most of the weight.

---

## 4.2 The Math

### Step 1: Project to Q, K, V

Given input $X \in \mathbb{R}^{T\times d}$, we learn three weight matrices:

$$
Wq \in \mathbb{R}^{d\times d_k},   Wk \in \mathbb{R}^{d\times d_k},   Wv \in \mathbb{R}^{d\times d_v}
$$

And compute:

$$
Q = X Wq  \in \mathbb{R}^{T\times d_k}   (queries)
K = X Wk  \in \mathbb{R}^{T\times d_k}   (keys)
V = X Wv  \in \mathbb{R}^{T\times d_v}   (values)
$$

In standard single-head attention, $d_k = d_v = d$.

### Step 2: Compute Attention Scores

$$
S = Q K^{\top} / \sqrt{d_k}  \in \mathbb{R}^{T\times T}
$$

`S[i,j]` measures how relevant token `j` is to token `i`. The division by $\sqrt{d_k}$ prevents the dot products from growing too large (which would push softmax into regions with near-zero gradients).

> **Math Minute — Matrix Multiplication**
> For matrices $A \in \mathbb{R}^{m\times n}$ and $B \in \mathbb{R}^{n\times p}$:
> $(AB)[i,j] = Σ_k A[i,k] \cdot B[k,j]$
> The result has shape $[m\times p]$. Each entry is a dot product of a row of `A` and a column of `B`.

### Step 3: Apply Causal Mask (for autoregressive models)

GPT is **autoregressive**: when predicting token `t`, it must not look at tokens `t+1, t+2, …` (they haven't been generated yet). We enforce this by masking the upper triangle:

$$
M[i,j] = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}
$$

$$
S_{\text{masked}} = S + M
$$

Adding $-\infty$ before softmax effectively zeroes out those attention weights.

### Step 4: Softmax → Attention Weights

$$
A = \operatorname{softmax}(S_{\text{masked}}) \in \mathbb{R}^{T\times T}
$$

`A[i,j]` is now a probability: how much token `i` attends to token `j`.

### Step 5: Weighted Sum of Values

$$
Output = A V   \in \mathbb{R}^{T\times d_v}
$$

`Output[i]` is a weighted blend of all value vectors, guided by how much token `i` attends to each position.

**Full formula:**

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}( QK^{\top} / \sqrt{d_k} + M ) \cdot V
$$

---

## 4.3 The Matrix: Worked Example

Let `T = 3` tokens, `d = 4`, $d_k = d_v = 4$. Input (position-encoded):

```
X = [[ 1.0,  0.0,  1.0,  0.0],   ← token 0: "the"
     [ 0.0,  1.0,  0.0,  1.0],   ← token 1: "cat"
     [ 1.0,  1.0,  0.0,  0.0]]   ← token 2: "sat"
```

Weight matrices (simplified, identity-like):

$$
Wq = Wk = Wv = I_4 \quad (4\times 4 \text{ identity, so } Q=K=V=X)
$$

**Step 1 — Q, K, V (same as X here):**

```
Q = K = V = X
```

**Step 2 — Scores $S = QK^{\top} / \sqrt{4}$:**

```
QKᵀ[0,0] = [1,0,1,0]·[1,0,1,0] = 1+0+1+0 = 2
QKᵀ[0,1] = [1,0,1,0]·[0,1,0,1] = 0+0+0+0 = 0
QKᵀ[0,2] = [1,0,1,0]·[1,1,0,0] = 1+0+0+0 = 1
QKᵀ[1,0] = 0,  QKᵀ[1,1] = 2,  QKᵀ[1,2] = 1
QKᵀ[2,0] = 1,  QKᵀ[2,1] = 1,  QKᵀ[2,2] = 2

S = QKᵀ / 2 =
  [[1.0,  0.0,  0.5],
   [0.0,  1.0,  0.5],
   [0.5,  0.5,  1.0]]
```

**Step 3 — Causal mask:**

```
S_masked =
  [[1.0,   -∞,   -∞],    ← token 0 can only attend to itself
   [0.0,  1.0,   -∞],    ← token 1 can attend to 0 and 1
   [0.5,  0.5,  1.0]]    ← token 2 can attend to all
```

**Step 4 — Softmax:**

```
A[0] = softmax([1.0, -∞, -∞]) = [1.000, 0.000, 0.000]
A[1] = softmax([0.0, 1.0, -∞]) = [0.269, 0.731, 0.000]
A[2] = softmax([0.5, 0.5, 1.0]) = [0.211, 0.211, 0.578]
```

**Step 5 — Output = AV:**

```
Output[0] = 1.000·V[0] + 0.000·V[1] + 0.000·V[2]
          = [1.0, 0.0, 1.0, 0.0]

Output[1] = 0.269·[1,0,1,0] + 0.731·[0,1,0,1]
          = [0.269, 0.731, 0.269, 0.731]

Output[2] = 0.211·[1,0,1,0] + 0.211·[0,1,0,1] + 0.578·[1,1,0,0]
          = [0.789, 0.789, 0.211, 0.211]
```

Token 2 ("sat") is blending information from all three tokens, with most weight on itself.



---

## 4.4 The Code: Scaled Dot-Product Attention in Scheme

Attention is a composition of named operations. We define each piece — multiply, transpose, scale, softmax, mask — and then `scaled-dot-product-attention` is just a single `let*` that threads them together. Reading the procedure is reading the formula.

The matrix abstraction from Chapter 2 carries forward. We add four operations that form the vocabulary for everything in this chapter: multiply, transpose, scale, and add. Each takes matrices in and returns a new matrix out — no mutation of arguments.

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
                  (loop-k (+ p 1) (+ acc (* (matrix-ref A i p) (matrix-ref B p j))))))
            (loop-j (+ j 1))))
        (loop-i (+ i 1))))
    C))

(define (matrix-transpose A)
  (let* ((m (matrix-rows A)) (n (matrix-cols A)) (AT (make-matrix n m)))
    (let loop-i ((i 0))
      (when (< i m)
        (let loop-j ((j 0))
          (when (< j n)
            (matrix-set! AT j i (matrix-ref A i j))
            (loop-j (+ j 1))))
        (loop-i (+ i 1))))
    AT))

(define (matrix-scale A s)
  (let* ((m (matrix-rows A)) (n (matrix-cols A)) (B (make-matrix m n)))
    (let loop-i ((i 0))
      (when (< i m)
        (let loop-j ((j 0))
          (when (< j n)
            (matrix-set! B i j (* s (matrix-ref A i j)))
            (loop-j (+ j 1))))
        (loop-i (+ i 1))))
    B))

(define (matrix-add A B)
  (let* ((m (matrix-rows A)) (n (matrix-cols A)) (C (make-matrix m n)))
    (let loop-i ((i 0))
      (when (< i m)
        (let loop-j ((j 0))
          (when (< j n)
            (matrix-set! C i j (+ (matrix-ref A i j) (matrix-ref B i j)))
            (loop-j (+ j 1))))
        (loop-i (+ i 1))))
    C))
```

`softmax` converts a vector of raw scores into a probability distribution. We subtract the maximum value before exponentiating — this does not change the output (softmax is shift-invariant) but prevents floating-point overflow when scores are large. `softmax-each-row` applies it independently to every row of a matrix, giving each query token its own distribution over keys.

```scheme
(define (softmax v)
  (let* ((n     (vector-length v))
         (max-v (let find-max ((i 1) (m (vector-ref v 0)))
                  (if (= i n) m (find-max (+ i 1) (max m (vector-ref v i))))))
         (exps  (make-vector n))
         (total (let sum ((i 0) (s 0.0))
                  (if (= i n) s
                      (let ((e (exp (- (vector-ref v i) max-v))))
                        (vector-set! exps i e)
                        (sum (+ i 1) (+ s e)))))))
    (let norm ((i 0))
      (when (< i n)
        (vector-set! exps i (/ (vector-ref exps i) total))
        (norm (+ i 1))))
    exps))

(define (softmax-each-row M)
  (let* ((T (matrix-rows M)) (n (matrix-cols M)) (A (make-matrix T n)))
    (let loop ((i 0))
      (when (< i T)
        (let ((row (make-vector n)))
          (let copy ((j 0))
            (when (< j n) (vector-set! row j (matrix-ref M i j)) (copy (+ j 1))))
          (let ((soft (softmax row)))
            (let copy-back ((j 0))
              (when (< j n) (matrix-set! A i j (vector-ref soft j)) (copy-back (+ j 1))))))
        (loop (+ i 1))))
    A))
```

The causal mask fills the upper triangle with $-10^9$. When added to the raw score matrix before softmax, those positions become $\approx 0$ in the attention weights. Token $i$ is effectively forbidden from attending to any position $j > i$ — the future does not exist yet during generation.

```scheme
(define (causal-mask T)
  (let ((M (make-matrix T T)))
    (let loop-i ((i 0))
      (when (< i T)
        (let loop-j ((j (+ i 1)))
          (when (< j T) (matrix-set! M i j -1.0e9) (loop-j (+ j 1))))
        (loop-i (+ i 1))))
    M))
```

`scaled-dot-product-attention` is a direct translation of the formula $\text{Attention}(Q,K,V) = \text{softmax}(QK^\top/\sqrt{d_k} + \text{mask}) \cdot V$. Each term in the formula becomes one procedure call. The result is a pair: the output matrix and the attention weights (useful for inspection).

```scheme
(define (scaled-dot-product-attention Q K V)
  (let* ((dk     (exact->inexact (matrix-cols Q)))
         (T      (matrix-rows Q))
         (S      (matrix-scale (matrix-multiply Q (matrix-transpose K)) (/ 1.0 (sqrt dk))))
         (S-mask (matrix-add S (causal-mask T)))
         (A      (softmax-each-row S-mask))
         (output (matrix-multiply A V)))
    (cons output A)))
```

`self-attention` wraps SDPA with the three linear projections. $Q = XW_q$, $K = XW_k$, $V = XW_v$ — three different "views" of the same input sequence. The weight matrices $W_q, W_k, W_v$ are what the model learns.

```scheme
(define (self-attention X Wq Wk Wv)
  (scaled-dot-product-attention
   (matrix-multiply X Wq)
   (matrix-multiply X Wk)
   (matrix-multiply X Wv)))
```

**Demo.** Run with `mit-scheme --quiet --load attention.scm`. We use identity weight matrices so $Q=K=V=X$, matching the worked example in §4.3.

```scheme
(define (fill-matrix! M vals)
  (let loop ((i 0) (j 0) (vs vals))
    (unless (null? vs)
      (matrix-set! M i j (car vs))
      (let ((j-next (+ j 1)))
        (if (= j-next (matrix-cols M))
            (loop (+ i 1) 0 (cdr vs))
            (loop i j-next (cdr vs)))))))

(let ((X  (make-matrix 3 4))
      (I4 (make-matrix 4 4)))
  (fill-matrix! X  '(1.0 0.0 1.0 0.0  0.0 1.0 0.0 1.0  1.0 1.0 0.0 0.0))
  (fill-matrix! I4 '(1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.0 0.0 0.0 1.0))
  (let* ((result  (self-attention X I4 I4 I4))
         (output  (car result))
         (weights (cdr result)))
    (display "Attention weights [3×3]:") (newline)
    (let loop ((i 0))
      (when (< i 3)
        (display "  [")
        (let col ((j 0))
          (when (< j 3) (display (matrix-ref weights i j)) (display " ") (col (+ j 1))))
        (display "]") (newline)
        (loop (+ i 1))))))
```

---

## 4.5 Key Takeaways

- Attention asks: for each token, **which other tokens are most relevant**?
- Q, K, V are **learned linear projections** of the input.
- The attention score $S[i,j] = Q_i \cdot K_j / \sqrt{d_k}$ measures relevance.
- Causal masking prevents tokens from seeing future positions (critical for text generation).
- The output is a **soft, weighted average** of value vectors.
- Full formula: $\operatorname{Attention}(Q,K,V) = \operatorname{softmax}(QK^{\top}/\sqrt{d_k} + mask) V$

> **What's next?** One attention head can only track one type of relationship at a time. What if we want to simultaneously track syntactic dependencies, coreference, and semantic similarity? We run **multiple attention heads in parallel** — Chapter 5.


---

![Q, K, V projection — three separate linear transforms of the same input X.](images/ch04_qkv.png)

*Q, K, V projection — three separate linear transforms of the same input X.*

![Attention weight heatmap (causal) — rows are queries, columns are keys; future is masked.](images/ch04_attn_weights.png)

*Attention weight heatmap (causal) — rows are queries, columns are keys; future is masked.*

