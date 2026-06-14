# Chapter 2: Embeddings — Numbers to Meaning

> *"A number is an address. A vector is a room."*

We have token IDs: `[10, 3, 4, 8]`. They are integers. The model cannot do useful math on raw integers — adding `10 + 3` gives `13`, which means nothing. To do useful math, each integer needs to become a **vector** — a list of numbers that encodes the token's meaning geometrically.

The mechanism is called **the embedding matrix**, and it is the model's first learned representation.

---

## 2.1 The Idea

Think of the embedding matrix as an enormous lookup table. It has one row per vocabulary entry and `d_model` columns, where `d_model` is the model's **hidden dimension** — a hyperparameter that controls how rich each token's representation is. In GPT-2 small, `d_model = 768`. In LLaMA-3 70B, `d_model = 8192`.

$$
Embedding matrix E:  shape [|V| \times d_model]

  token 0:  [  0.12  -0.83   0.41  \ldots  (d_model numbers) ]
  token 1:  [ -0.55   0.19   0.77  \ldots                    ]
  token 2:  [  0.03   0.61  -0.29  \ldots                    ]
  ...
  token |V|: [ \ldots                                         ]
$$

To embed a token with ID `i`, you just take row `i`. That's it. A single matrix row lookup. In NumPy: `E[i]`. In math: $e_i = E_i$.

For a sequence of `T` tokens with IDs $[i_1, i_2, \ldots, i_t]$, the embedding step produces a matrix of shape $[T \times d_model]$ — one row per token.

> **Math Minute — Vectors**
> A vector $v \in \mathbb{R}^n$ is an ordered list of `n` real numbers: $v = [v_1, v_2, \ldots, v_n]$. You can add vectors (`[1,2] + [3,4] = [4,6]`), scale them ($2\cdot[1,2] = [2,4]$), and measure their similarity via the **dot product** (Chapter 4). Geometrically, a vector is a point — or an arrow — in n-dimensional space.

---

## 2.2 Why Learned Vectors?

The values in the embedding matrix are not hand-crafted. They are initialized randomly and then **learned via gradient descent** during training, just like any other model parameter.

The training objective is next-token prediction: given $t_1, t_2, \ldots, t_n$, predict $t_{n+1}$. Because the model must do this well across billions of examples, it is forced to arrange embeddings such that tokens that appear in similar contexts land near each other in the embedding space.

This produces the famous arithmetic property:

$$
embedding("king") - embedding("man") + embedding("woman") \approx embedding("queen")
$$

No one programmed this. It is a geometric consequence of how the vectors were shaped by the training signal.

> **Math Minute — Dot Product**
> The dot product of two vectors $u = [u_1,\ldots,u_n]$ and $v = [v_1,\ldots,v_n]$ is:
> $u \cdot v = u_{1v1} + u_{2v2} + \ldots + u_{nvn}$
> It is a single number. If both vectors point in the same direction, the dot product is large and positive. If they are perpendicular, it is zero. If they point opposite, it is negative. Dot product measures **alignment**.

---

## 2.3 The Math

Given:
- `|V|` = vocabulary size
- `d` = `d_model` (hidden dimension)
- $E \in \mathbb{R}^{|V| \times d}$ = embedding matrix (learned)
- $i \in {0, \ldots, |V|-1}$ = token ID

The embedding of token `i` is:

$$
e_i = E_i,\cdot  \in \mathbb{R}^d
$$

(Row `i` of matrix `E`.)

For a sequence of `T` tokens $[i_1, \ldots, i_t]$, the embedding operation is:

$$
X = E[[i_1,\ldots,i_t], :]  \in \mathbb{R}^{T \times d}
$$

This is just row indexing — no multiplication. In practice it is implemented as a matrix multiplication with a one-hot vector:

$$
e_i = o_i^{\top} \cdot E    where o_i \in {0,1}^|V|, (o_i)_j = 1 if j=i else 0
$$

But implementations use the index directly because it is faster.

---

## 2.4 The Matrix: Worked Example

Let's use tiny numbers: `|V| = 5`, `d = 4`.

$$
Embedding matrix E (5\times 4):
       col0   col1   col2   col3
tok 0: [ 0.10  -0.20   0.30  -0.40 ]
tok 1: [ 0.50   0.60  -0.70   0.80 ]
tok 2: [-0.90   0.10   0.20  -0.30 ]
tok 3: [ 0.40  -0.50   0.60  -0.70 ]
tok 4: [-0.10   0.80  -0.40   0.50 ]
$$

Input token sequence: `"low lower"` → token IDs `[2, 3, 0]` (hypothetical).

Embedding lookup:

```
X = E[[2, 3, 0], :]

X[0] = E[2] = [-0.90,  0.10,  0.20, -0.30]   ← "low"
X[1] = E[3] = [ 0.40, -0.50,  0.60, -0.70]   ← "low" (part of "lower")
X[2] = E[0] = [ 0.10, -0.20,  0.30, -0.40]   ← "er"

Result X: shape [3 × 4]
```

This $[3 \times 4]$ matrix is what flows into the next stage.

![Embedding matrix lookup: token IDs map to row vectors](images/ch02-embedding-lookup.png)

**Semantic similarity in vector space:**

If we compute the dot product of two token embeddings, we get a scalar measuring how aligned they are:

```
dot(E[0], E[2]) = (0.10)(−0.90) + (−0.20)(0.10) + (0.30)(0.20) + (−0.40)(−0.30)
              = −0.09 − 0.02 + 0.06 + 0.12
              = 0.07   (slightly positive → slight similarity)
```

After training, semantically related tokens will have much higher dot products.

---

## 2.5 The Code: Embedding in Scheme

Before writing a single line of embedding logic, we name what a matrix *is*. The key design move is choosing the right abstraction boundary — the wall between representation and use. A matrix is a rows-by-cols grid of numbers. We store it as a flat vector in row-major order, but nothing above this interface will ever need to know that.

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
```

With the abstraction in place, we can build the embedding matrix. The values are not hand-crafted — they start random and are shaped by gradient descent during training. Xavier initialization scales the random values by $\sqrt{2/(fan_{in}+fan_{out})}$, which keeps activations from exploding or vanishing as signals pass through many layers. We draw each entry uniformly from $[-scale, +scale]$ and store it through the abstraction we just defined.

```scheme
(define (make-random-matrix rows cols)
  (let ((M     (make-matrix rows cols))
        (scale (sqrt (/ 2.0 (+ rows cols)))))
    (let row-loop ((i 0))
      (when (< i rows)
        (let col-loop ((j 0))
          (when (< j cols)
            (matrix-set! M i j
              (* scale (- (* 2.0 (random 1.0)) 1.0)))
            (col-loop (+ j 1))))
        (row-loop (+ i 1))))
    M))
```

The embedding of token $i$ is literally row $i$ of the matrix $E$ — a table lookup, nothing more. $e_i = E[i,:]$. We write a procedure that extracts one row as a fresh vector, making the operation explicit and independent of the matrix's internal storage format.

```scheme
(define (embed-token E token-id)
  (let* ((d (matrix-cols E))
         (v (make-vector d)))
    (let loop ((j 0))
      (when (< j d)
        (vector-set! v j (matrix-ref E token-id j))
        (loop (+ j 1))))
    v))
```

A sequence of $T$ token IDs calls for $T$ row lookups stacked into a $[T \times d]$ matrix. The single-token operation composes naturally into a sequence operation because both work through the same abstraction. We iterate the list of IDs, copying each row in turn.

```scheme
(define (embed-sequence E token-ids)
  (let* ((T (length token-ids))
         (d (matrix-cols E))
         (X (make-matrix T d)))
    (let loop ((ids token-ids) (t 0))
      (unless (null? ids)
        (let copy-row ((j 0))
          (when (< j d)
            (matrix-set! X t j (matrix-ref E (car ids) j))
            (copy-row (+ j 1))))
        (loop (cdr ids) (+ t 1))))
    X))
```

The dot product $u \cdot v = \sum_i u_i v_i$ measures geometric alignment between two vectors. After training, tokens that appear in similar contexts will have high dot products. Cosine similarity normalizes by magnitude so that $\cos\theta = 1$ means identical direction regardless of scale — both procedures will reappear throughout the remaining chapters.

```scheme
(define (dot u v)
  (let loop ((i 0) (acc 0.0))
    (if (= i (vector-length u))
        acc
        (loop (+ i 1) (+ acc (* (vector-ref u i) (vector-ref v i)))))))

(define (vector-norm v) (sqrt (dot v v)))

(define (cosine-similarity u v)
  (/ (dot u v) (* (vector-norm u) (vector-norm v))))
```

A display utility lets us inspect any matrix by label. It loops over rows and columns, printing each entry separated by spaces, and will be reused in every chapter that follows.

```scheme
(define (display-matrix M label)
  (display label)
  (display " [") (display (matrix-rows M))
  (display "x") (display (matrix-cols M)) (display "]:") (newline)
  (let row-loop ((i 0))
    (when (< i (matrix-rows M))
      (display "  [")
      (let col-loop ((j 0))
        (when (< j (matrix-cols M))
          (display (matrix-ref M i j)) (display " ")
          (col-loop (+ j 1))))
      (display "]") (newline)
      (row-loop (+ i 1)))))
```

**Demo.** We build a small embedding matrix, embed a three-token sequence, then print the result and two similarity measures. Before training, the dot product and cosine similarity are near zero — the random vectors are roughly orthogonal to each other. After training they would reflect semantic proximity.

```scheme
(let* ((vocab-size 100)
       (d-model    8)
       (E          (make-random-matrix vocab-size d-model))
       (ids        '(1 2 3))
       (X          (embed-sequence E ids)))
  (display-matrix X "Embedding output X [T=3, d=8]")
  (newline)
  (display "Dot product E[1].E[2]: ")
  (display (dot (embed-token E 1) (embed-token E 2))) (newline)
  (display "Cosine similarity E[1].E[2]: ")
  (display (cosine-similarity (embed-token E 1) (embed-token E 2))) (newline))
```

Run with `mit-scheme --quiet --load embeddings.scm`.

---

## 2.6 Key Takeaways

- An embedding matrix $E \in \mathbb{R}^{|V| \times d}$ maps token IDs to dense vectors.
- The embedding of token `i` is row `i` of `E` — a single **lookup**, no computation.
- For a sequence of `T` tokens, we get a $[T \times d]$ matrix `X`.
- The vectors are **learned**: training nudges similar tokens closer together.
- Semantic arithmetic ($king - man + woman \approx queen$) emerges from the training objective.

> **What's next?** We have a matrix `X` of shape $[T \times d]$. Each row knows *what* the token is, but nothing about *where* it sits in the sequence. Swapping two tokens would produce the same rows in a different order, which the model would treat identically. We need to inject **position information** — that's Chapter 3.


---

![Embedding matrix lookup — each token ID indexes into a row of the weight matrix.](images/ch02_embedding_lookup.png)

*Embedding matrix lookup — each token ID indexes into a row of the weight matrix.*

![Semantic arithmetic in embedding space — king − man + woman ≈ queen.](images/ch02_semantic_space.png)

*Semantic arithmetic in embedding space — king − man + woman ≈ queen.*

