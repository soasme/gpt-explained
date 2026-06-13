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
M[i,j] = 0    if j \leq i    (allowed to attend)
M[i,j] = -\infty  if j > i    (blocked — future tokens)

S_masked = S + M
$$

Adding $-\infty$ before softmax effectively zeroes out those attention weights.

### Step 4: Softmax → Attention Weights

$$
A = \operatorname{softmax}(S_masked)   \in \mathbb{R}^{T\times T}
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
Wq = Wk = Wv = I_4 (4\times 4 identity, so Q=K=V=X for this example)
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

## 4.4 The Code: Scaled Dot-Product Attention in Lisp

```lisp
;;;; attention.lisp
;;;; Scaled dot-product attention (the core of the transformer)

;;; ─── Prerequisites from earlier chapters ────────────────────────
;; (load "embeddings.lisp")      ; provides make-array utilities
;; (load "positional-encoding.lisp")

;;; ─── Matrix Operations ───────────────────────────────────────────

(defun mat-mul (A B)
  "Multiply matrix A [m×k] by matrix B [k×n]. Returns [m×n] matrix."
  (destructuring-bind (m k-a) (array-dimensions A)
    (destructuring-bind (k-b n) (array-dimensions B)
      (assert (= k-a k-b) () "Matrix dimension mismatch: ~a vs ~a" k-a k-b)
      (let ((C (make-array (list m n) :element-type 'single-float
                                      :initial-element 0.0)))
        (dotimes (i m C)
          (dotimes (j n)
            (dotimes (k k-a)
              (incf (aref C i j) (* (aref A i k) (aref B k j))))))))))

(defun mat-transpose (A)
  "Transpose matrix A [m×n] → [n×m]."
  (destructuring-bind (m n) (array-dimensions A)
    (let ((AT (make-array (list n m) :element-type 'single-float)))
      (dotimes (i m AT)
        (dotimes (j n)
          (setf (aref AT j i) (aref A i j)))))))

(defun mat-scale (A scalar)
  "Multiply every element of A by SCALAR."
  (destructuring-bind (m n) (array-dimensions A)
    (let ((B (make-array (list m n) :element-type 'single-float)))
      (dotimes (i m B)
        (dotimes (j n)
          (setf (aref B i j) (* scalar (aref A i j))))))))

;;; ─── Softmax over each row ───────────────────────────────────────

(defun softmax-row (v)
  "Apply softmax to 1D vector V. Returns new vector summing to 1."
  (let* ((n (length v))
         (max-v (reduce #'max v))
         (exps (map '(vector single-float)
                    (lambda (x) (exp (- x max-v))) v))
         (sum (reduce #'+ exps))
         (out (make-array n :element-type 'single-float)))
    (dotimes (i n out)
      (setf (aref out i) (/ (aref exps i) sum)))))

(defun softmax-rows (M)
  "Apply softmax independently to each row of matrix M [T×T]."
  (destructuring-bind (T-len _) (array-dimensions M)
    (declare (ignore _))
    (let ((A (make-array (array-dimensions M) :element-type 'single-float)))
      (dotimes (i T-len A)
        (let* ((row (make-array T-len :element-type 'single-float))
               (soft-row nil))
          (dotimes (j T-len) (setf (aref row j) (aref M i j)))
          (setf soft-row (softmax-row row))
          (dotimes (j T-len)
            (setf (aref A i j) (aref soft-row j))))))))

;;; ─── Causal mask ─────────────────────────────────────────────────

(defun causal-mask (T-len)
  "Return a [T×T] mask: 0 on/below diagonal, -1e9 above.
   -1e9 is used as a proxy for -∞ (zeroed out by softmax)."
  (let ((M (make-array (list T-len T-len) :element-type 'single-float
                                           :initial-element 0.0)))
    (dotimes (i T-len M)
      (loop for j from (1+ i) below T-len do
        (setf (aref M i j) -1.0e9)))))

(defun mat-add (A B)
  "Element-wise add two same-shape matrices."
  (destructuring-bind (m n) (array-dimensions A)
    (let ((C (make-array (list m n) :element-type 'single-float)))
      (dotimes (i m C)
        (dotimes (j n)
          (setf (aref C i j) (+ (aref A i j) (aref B i j))))))))

;;; ─── Scaled Dot-Product Attention ────────────────────────────────

(defun scaled-dot-product-attention (Q K V &key (mask t))
  "Compute Attention(Q,K,V) = softmax(QKᵀ/√dₖ [+ causal_mask]) · V

   Q : [T × dₖ]
   K : [T × dₖ]
   V : [T × dᵥ]

   Returns:
     output : [T × dᵥ]  — context vectors
     weights: [T × T]   — attention probabilities"
  (let* ((d-k (float (array-dimension Q 1)))
         (T-len (array-dimension Q 0))
         ;; S = QKᵀ / √dₖ
         (S (mat-scale (mat-mul Q (mat-transpose K))
                       (/ 1.0 (sqrt d-k))))
         ;; Optional causal mask
         (S-masked (if mask (mat-add S (causal-mask T-len)) S))
         ;; A = softmax row-wise
         (A (softmax-rows S-masked))
         ;; Output = A · V
         (output (mat-mul A V)))
    (values output A)))

;;; ─── Linear projection (for Q, K, V matrices) ───────────────────

(defun linear (X W)
  "Project X [T × d_in] with weight W [d_in × d_out]. Returns [T × d_out]."
  (mat-mul X W))

(defun rand-weight-matrix (d-in d-out)
  "Xavier-initialized weight matrix [d_in × d_out]."
  (let* ((M (make-array (list d-in d-out) :element-type 'single-float))
         (scale (sqrt (/ 2.0 (+ d-in d-out)))))
    (dotimes (i d-in M)
      (dotimes (j d-out)
        (setf (aref M i j) (* scale (- (random 2.0) 1.0)))))))

;;; ─── Single-head self-attention ──────────────────────────────────

(defun self-attention (X Wq Wk Wv)
  "Full single-head self-attention.
   X  : [T × d]
   Wq : [d × dₖ]
   Wk : [d × dₖ]
   Wv : [d × dᵥ]
   Returns (values output attention-weights)."
  (let ((Q (linear X Wq))
        (K (linear X Wk))
        (V (linear X Wv)))
    (scaled-dot-product-attention Q K V)))

;;; ─── Demo ─────────────────────────────────────────────────────────

(let* ((T-len 3)
       (d 4)
       (dk 4)
       (dv 4)
       ;; Simple input: each row is one token's embedding
       (X (make-array '(3 4) :element-type 'single-float
                      :initial-contents '((1.0 0.0 1.0 0.0)
                                          (0.0 1.0 0.0 1.0)
                                          (1.0 1.0 0.0 0.0))))
       ;; Identity weights for clarity
       (I4 (make-array '(4 4) :element-type 'single-float
                       :initial-contents '((1.0 0.0 0.0 0.0)
                                           (0.0 1.0 0.0 0.0)
                                           (0.0 0.0 1.0 0.0)
                                           (0.0 0.0 0.0 1.0)))))
  (multiple-value-bind (output weights)
      (self-attention X I4 I4 I4)
    (format t "Attention weights A [3×3]:~%")
    (dotimes (i 3)
      (format t "  [")
      (dotimes (j 3) (format t "~7,3f" (aref weights i j)))
      (format t " ]~%"))
    (format t "~%Output [3×4]:~%")
    (dotimes (i 3)
      (format t "  [")
      (dotimes (j 4) (format t "~7,3f" (aref output i j)))
      (format t " ]~%"))))
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

