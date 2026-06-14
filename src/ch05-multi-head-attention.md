# Chapter 5: Multi-Head Attention — Many Conversations at Once

> *"One head hears syntax. Another hears coreference. A third hears topic. They all speak simultaneously."*

A single attention head is expressive, but limited. It produces one set of attention weights — one pattern of "who attends to whom." In natural language, multiple distinct relationships coexist in the same sentence. Consider:

```
"The animal didn't cross the street because it was too tired."
```

- One pattern might link "it" → "animal" (coreference)
- Another might link "cross" → "street" (verb-object)
- Another might track the causal structure "because"

Multi-head attention runs `H` independent attention heads in parallel, each free to specialize on a different relationship type. Their outputs are concatenated and projected back to the model dimension.

---

## 5.1 The Idea

A single attention operation produces one pattern of "who attends to whom." But a sentence carries many different kinds of relationships at the same time.

In "The animal didn't cross the street because it was too tired":
- *it* refers back to *animal* — that is a **coreference** relationship.
- *cross* links to *street* — that is a **verb-object** relationship.
- *because* connects a cause to an effect — that is a **logical** relationship.

One attention head can only focus on one of these at a time. Multi-head attention runs several independent attention operations **in parallel** — each one free to specialize on a different pattern. Each head sees the same input but learns to ask a different question of it.

At the end, the results from all heads are stitched back together and projected into a single vector, the same size as before. The model learns entirely from data which head should track grammar, which should track meaning, which should track proximity — no one programs this in explicitly.

The result is a richer representation than any single head could produce: each token's final vector has been informed by multiple different lenses simultaneously.

---

## 5.2 The Math

**Step 1 — Compute each head independently:**

$$
For h = 1, \ldots, H:
  Q^{h} = X Wq^{h}    \in \mathbb{R}^{T\times d_k}
  K^{h} = X Wk^{h}    \in \mathbb{R}^{T\times d_k}
  V^{h} = X Wv^{h}    \in \mathbb{R}^{T\times d_v}
  head^{h} = \operatorname{softmax}( Q^{h}{K^{h}}^{\top} / \sqrt{d_k} + M ) V^{h}
$$

**Step 2 — Concatenate:**

$$
\operatorname{MultiHead} = concat(head^1, head^2, \ldots, head^{H})   \in \mathbb{R}^{T \times (H\cdot d_v)}
$$

Since $d_v = d/H$, the concatenated output has shape $[T \times d]$ — the same as the input.

**Step 3 — Output projection:**

$$
Wo \in \mathbb{R}^{d\times d}   (learned)

Output = \operatorname{MultiHead} \cdot Wo   \in \mathbb{R}^{T\times d}
$$

The final output projection mixes information across heads.

**Full formula:**

$$
MHA(X) = [head^1 \| head^2 \| \ldots \| head^{H}] Wo

where head^{h} = \operatorname{Attention}(X Wq^{h}, X Wk^{h}, X Wv^{h})
$$

---

## 5.3 The Matrix: Worked Example

Let `T = 3`, `d = 4`, `H = 2` heads, so $d_k = d_v = 2$.

Input:
```
X = [[1, 0, 1, 0],
     [0, 1, 0, 1],
     [1, 1, 0, 0]]   (3×4)
```

**Head 1** uses the first 2 dimensions primarily.
Weight matrices (simplified):

```
Wq¹ = Wk¹ = Wv¹ = [[1,0],[0,1],[0,0],[0,0]]   (4×2 — project to first 2 dims)
```

```
Q¹ = X Wq¹ = [[1,0],[0,1],[1,1]]   (3×2)
K¹ = same
V¹ = same
```

Scores: $S^1 = Q^1{K^1}^{\top} / \sqrt{2}$:
```
Q¹K¹ᵀ = [[1,0,1],[0,1,1],[1,1,2]]
S¹    = [[0.71, 0.00, 0.71],
         [0.00, 0.71, 0.71],
         [0.71, 0.71, 1.41]]
```

After causal mask and softmax:
```
A¹ = [[1.000, 0.000, 0.000],
      [0.414, 0.586, 0.000],
      [0.221, 0.221, 0.558]]
```

```
head¹ = A¹ V¹ = [[1.000, 0.000],
                 [0.586, 0.414],
                 [0.779, 0.779]]    (3×2)
```

**Head 2** focuses on last 2 dimensions:

```
Wq² = Wk² = Wv² = [[0,0],[0,0],[1,0],[0,1]]   (4×2)

Q² = [[1,0],[0,1],[0,0]]

(following the same attention steps...)
head² = [[1.000, 0.000],
         [0.500, 0.500],
         [0.421, 0.211]]    (3×2)
```

**Concatenate:**

```
MultiHead = [head¹ ‖ head²] =
  [[1.000, 0.000, 1.000, 0.000],
   [0.586, 0.414, 0.500, 0.500],
   [0.779, 0.779, 0.421, 0.211]]   (3×4)
```

**Output projection** with $Wo \in \mathbb{R}^{4\times 4}$ mixes the heads' outputs.

![Multi-head attention: two heads capturing different relationship types](images/ch05-multi-head-attention.png)

---

## 5.4 The Code: Multi-Head Attention in Scheme

The matrix abstraction and `matrix-multiply` from Chapter 2 and Chapter 4 carry over. We also use `scaled-dot-product-attention` from Chapter 4 — multi-head attention builds directly on top of it.

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
                  (loop-k (+ p 1)
                          (+ acc (* (matrix-ref A i p)
                                    (matrix-ref B p j))))))
            (loop-j (+ j 1))))
        (loop-i (+ i 1))))
    C))
```

Each attention head is a triple of weight matrices $(W_q, W_k, W_v)$, each of shape $[d \times d_k]$ where $d_k = d/H$. We represent a head as a plain list so selectors stay readable. `make-mha-params` builds $H$ such triples plus the shared output projection $W_o$ of shape $[d \times d]$. Xavier initialization keeps the scale sensible regardless of $H$. Setting $d_k = d/H$ means the total number of parameters stays constant as $H$ grows — the heads share the model dimension equally.

```scheme
(define (make-random-matrix rows cols)
  (let ((M     (make-matrix rows cols))
        (scale (sqrt (/ 2.0 (+ rows cols)))))
    (let row-loop ((i 0))
      (when (< i rows)
        (let col-loop ((j 0))
          (when (< j cols)
            (matrix-set! M i j (* scale (- (* 2.0 (random 1.0)) 1.0)))
            (col-loop (+ j 1))))
        (row-loop (+ i 1))))
    M))

(define (make-head d dk)
  (list (make-random-matrix d dk)
        (make-random-matrix d dk)
        (make-random-matrix d dk)))

(define (head-wq h) (car h))
(define (head-wk h) (cadr h))
(define (head-wv h) (caddr h))

(define (make-mha-params d num-heads)
  (let* ((dk    (/ d num-heads))
         (heads (let build ((n num-heads) (acc '()))
                  (if (= n 0) acc
                      (build (- n 1) (cons (make-head d dk) acc)))))
         (wo    (make-random-matrix d d)))
    (list heads wo)))

(define (mha-heads params) (car params))
(define (mha-wo    params) (cadr params))
```

`hstack` concatenates a list of matrices column-wise, placing each successive matrix's columns immediately after the last. The result's column count equals the sum of all input column counts; the row count is unchanged. This is how we assemble the $H$ head outputs — each of shape $[T \times d_k]$ — into a single $[T \times d]$ matrix before the output projection.

```scheme
(define (hstack matrices)
  (let* ((T    (matrix-rows (car matrices)))
         (cols (map matrix-cols matrices))
         (d    (apply + cols))
         (R    (make-matrix T d)))
    (let outer ((mats matrices) (offset 0))
      (unless (null? mats)
        (let* ((M  (car mats))
               (nc (matrix-cols M)))
          (let loop-i ((i 0))
            (when (< i T)
              (let loop-j ((j 0))
                (when (< j nc)
                  (matrix-set! R i (+ offset j) (matrix-ref M i j))
                  (loop-j (+ j 1))))
              (loop-i (+ i 1))))
          (outer (cdr mats) (+ offset nc)))))
    R))
```

`multi-head-attention` maps over the list of heads, computes SDPA for each independently, collects all outputs, then hstacks them and applies the output projection $W_o$. The `map` gives every head its own private $Q, K, V$ subspace — no head can see another's computations until they are merged by $W_o$. The returned pair carries the projected output and the full list of per-head attention weights, one weight matrix per head.

```scheme
(define (scaled-dot-product-attention Q K V)
  (let* ((dk     (exact->inexact (matrix-cols Q)))
         (T      (matrix-rows Q))
         (S      (matrix-scale (matrix-multiply Q (matrix-transpose K))
                               (/ 1.0 (sqrt dk))))
         (S-mask (matrix-add S (causal-mask T)))
         (A      (softmax-each-row S-mask))
         (output (matrix-multiply A V)))
    (cons output A)))

(define (multi-head-attention X params)
  (let* ((heads   (mha-heads params))
         (Wo      (mha-wo    params))
         (results (map (lambda (h)
                         (scaled-dot-product-attention
                          (matrix-multiply X (head-wq h))
                          (matrix-multiply X (head-wk h))
                          (matrix-multiply X (head-wv h))))
                       heads))
         (head-outputs (map car results))
         (head-weights (map cdr results))
         (concat-out   (hstack head-outputs))
         (output       (matrix-multiply concat-out Wo)))
    (cons output head-weights)))
```

**Demo.** We construct a small model with $d=8$ and $H=2$ heads, build a random $[4 \times 8]$ input, and run one forward pass. The output should have the same shape as the input; we print the attention weights for each head and the final projected output.

```scheme
(let* ((d         8)
       (num-heads 2)
       (T         4)
       (params    (make-mha-params d num-heads))
       (X         (make-random-matrix T d))
       (result    (multi-head-attention X params))
       (output    (car result))
       (weights   (cdr result)))
  (display "Head attention weights:") (newline)
  (let loop ((ws weights) (h 0))
    (unless (null? ws)
      (display "  head ") (display h) (display " [")
      (display (matrix-rows (car ws))) (display "x")
      (display (matrix-cols (car ws))) (display "]") (newline)
      (loop (cdr ws) (+ h 1))))
  (newline)
  (display "MHA output [") (display (matrix-rows output))
  (display "x") (display (matrix-cols output)) (display "]:") (newline)
  (let row ((i 0))
    (when (< i (matrix-rows output))
      (display "  [")
      (let col ((j 0))
        (when (< j (matrix-cols output))
          (display (matrix-ref output i j)) (display " ")
          (col (+ j 1))))
      (display "]") (newline)
      (row (+ i 1)))))
```

Run with `mit-scheme --quiet --load multi-head-attention.scm`.

---

## 5.5 Why Multi-Head Attention Works

Each head learns to specialize. Research has identified heads that:

- Track **syntactic structure** (subject-verb agreement)
- Resolve **coreference** ("it" → "the cat")
- Handle **positional offsets** ("look 2 tokens back")
- Track **rare-word semantics**

These specializations emerge from training, not from explicit design. The model learns which heads are useful for which tasks by adjusting the weight matrices through gradient descent.

> **Math Minute — Expressivity**
> H heads of dimension d/H can represent attention patterns that a single head of dimension d cannot easily learn. This is analogous to having H different "lenses" looking at the same sequence — each lens focuses on different features. The output projection `Wo` then combines the views.

---

## 5.6 Key Takeaways

- Multi-head attention runs `H` attention heads **in parallel**, each in a lower-dimensional subspace $d_k = d/H$.
- Each head has independent `Wq, Wk, Wv` matrices — each learns a different "question to ask."
- Outputs are **concatenated** (not averaged), then projected back to `d` with `Wo`.
- The total parameter count is $3Hd\cdot d_k + d^2 = 3d^2 + d^2$ — same as one large head.
- Different heads specialize in different linguistic relationships.

> **What's next?** After attention mixes information across tokens, each token's vector goes through a small **feed-forward network** — a two-layer MLP applied identically to every position. This is where most of the model's stored "knowledge" lives. Chapter 6.


---

![Multi-head attention — H parallel heads each attend independently, then concat + project.](images/ch05_mha.png)

*Multi-head attention — H parallel heads each attend independently, then concat + project.*

