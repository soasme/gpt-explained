# Chapter 2: Embeddings — Numbers to Meaning

> *"A number is an address. A vector is a room."*

We have token IDs: `[10, 3, 4, 8]`. They are integers. The model cannot do useful math on raw integers — adding `10 + 3` gives `13`, which means nothing. To do useful math, each integer needs to become a **vector** — a list of numbers that encodes the token's meaning geometrically.

The mechanism is called **the embedding matrix**, and it is the model's first learned representation.

---

## 2.1 The Idea

Think of the embedding matrix as an enormous lookup table. It has one row per vocabulary entry and `d_model` columns, where `d_model` is the model's **hidden dimension** — a hyperparameter that controls how rich each token's representation is. In GPT-2 small, `d_model = 768`. In LLaMA-3 70B, `d_model = 8192`.

```
Embedding matrix E:  shape [|V| × d_model]

  token 0:  [  0.12  -0.83   0.41  …  (d_model numbers) ]
  token 1:  [ -0.55   0.19   0.77  …                    ]
  token 2:  [  0.03   0.61  -0.29  …                    ]
  ...
  token |V|: [ …                                         ]
```

To embed a token with ID `i`, you just take row `i`. That's it. A single matrix row lookup. In NumPy: `E[i]`. In math: `eᵢ = Eᵢ`.

For a sequence of `T` tokens with IDs `[i₁, i₂, …, iₜ]`, the embedding step produces a matrix of shape `[T × d_model]` — one row per token.

> **Math Minute — Vectors**
> A vector `v ∈ ℝⁿ` is an ordered list of `n` real numbers: `v = [v₁, v₂, …, vₙ]`. You can add vectors (`[1,2] + [3,4] = [4,6]`), scale them (`2·[1,2] = [2,4]`), and measure their similarity via the **dot product** (Chapter 4). Geometrically, a vector is a point — or an arrow — in n-dimensional space.

---

## 2.2 Why Learned Vectors?

The values in the embedding matrix are not hand-crafted. They are initialized randomly and then **learned via gradient descent** during training, just like any other model parameter.

The training objective is next-token prediction: given `t₁, t₂, …, tₙ`, predict `tₙ₊₁`. Because the model must do this well across billions of examples, it is forced to arrange embeddings such that tokens that appear in similar contexts land near each other in the embedding space.

This produces the famous arithmetic property:

```
embedding("king") − embedding("man") + embedding("woman") ≈ embedding("queen")
```

No one programmed this. It is a geometric consequence of how the vectors were shaped by the training signal.

> **Math Minute — Dot Product**
> The dot product of two vectors `u = [u₁,…,uₙ]` and `v = [v₁,…,vₙ]` is:
> `u · v = u₁v₁ + u₂v₂ + … + uₙvₙ`
> It is a single number. If both vectors point in the same direction, the dot product is large and positive. If they are perpendicular, it is zero. If they point opposite, it is negative. Dot product measures **alignment**.

---

## 2.3 The Math

Given:
- `|V|` = vocabulary size
- `d` = `d_model` (hidden dimension)
- `E ∈ ℝ^{|V| × d}` = embedding matrix (learned)
- `i ∈ {0, …, |V|−1}` = token ID

The embedding of token `i` is:

```
eᵢ = Eᵢ,·  ∈ ℝᵈ
```

(Row `i` of matrix `E`.)

For a sequence of `T` tokens `[i₁, …, iₜ]`, the embedding operation is:

```
X = E[[i₁,…,iₜ], :]  ∈ ℝ^{T × d}
```

This is just row indexing — no multiplication. In practice it is implemented as a matrix multiplication with a one-hot vector:

```
eᵢ = oᵢᵀ · E    where oᵢ ∈ {0,1}^|V|, (oᵢ)ⱼ = 1 if j=i else 0
```

But implementations use the index directly because it is faster.

---

## 2.4 The Matrix: Worked Example

Let's use tiny numbers: `|V| = 5`, `d = 4`.

```
Embedding matrix E (5×4):
       col0   col1   col2   col3
tok 0: [ 0.10  -0.20   0.30  -0.40 ]
tok 1: [ 0.50   0.60  -0.70   0.80 ]
tok 2: [-0.90   0.10   0.20  -0.30 ]
tok 3: [ 0.40  -0.50   0.60  -0.70 ]
tok 4: [-0.10   0.80  -0.40   0.50 ]
```

Input token sequence: `"low lower"` → token IDs `[2, 3, 0]` (hypothetical).

Embedding lookup:

```
X = E[[2, 3, 0], :]

X[0] = E[2] = [-0.90,  0.10,  0.20, -0.30]   ← "low"
X[1] = E[3] = [ 0.40, -0.50,  0.60, -0.70]   ← "low" (part of "lower")
X[2] = E[0] = [ 0.10, -0.20,  0.30, -0.40]   ← "er"

Result X: shape [3 × 4]
```

This `[3 × 4]` matrix is what flows into the next stage.

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

## 2.5 The Code: Embedding in Lisp

```lisp
;;;; embeddings.lisp
;;;; Embedding matrix: integer IDs → dense vectors

;;; ─── Data Structures ────────────────────────────────────────────
;;
;; A MATRIX is a 2D array: (make-array '(rows cols) :element-type 'single-float)
;; A VECTOR is a 1D array: (make-array cols :element-type 'single-float)

(defun make-embedding-matrix (vocab-size d-model)
  "Create a |V| × d_model embedding matrix, Xavier-initialized."
  (let* ((arr (make-array (list vocab-size d-model)
                          :element-type 'single-float))
         (scale (sqrt (/ 2.0 (+ vocab-size d-model)))))
    (dotimes (i vocab-size)
      (dotimes (j d-model)
        (setf (aref arr i j)
              (* scale (- (random 2.0) 1.0)))))  ; uniform in [-scale, scale]
    arr))

;;; ─── Core Operation: Token → Vector ────────────────────────────

(defun embed-token (E token-id)
  "Look up row TOKEN-ID in embedding matrix E.
   Returns a fresh 1D vector (copy of that row)."
  (let* ((d-model (array-dimension E 1))
         (vec (make-array d-model :element-type 'single-float)))
    (dotimes (j d-model vec)
      (setf (aref vec j) (aref E token-id j)))))

;;; ─── Core Operation: Sequence → Matrix ─────────────────────────

(defun embed-sequence (E token-ids)
  "Embed a list of TOKEN-IDS using matrix E.
   Returns a [T × d_model] matrix where T = length(token-ids)."
  (let* ((T (length token-ids))
         (d-model (array-dimension E 1))
         (X (make-array (list T d-model) :element-type 'single-float)))
    (loop for id in token-ids
          for t from 0 do
      (let ((row (embed-token E id)))
        (dotimes (j d-model)
          (setf (aref X t j) (aref row j)))))
    X))

;;; ─── Utility: Print a matrix ────────────────────────────────────

(defun print-matrix (M &optional (label "Matrix"))
  (destructuring-bind (rows cols) (array-dimensions M)
    (format t "~%~a [~a × ~a]:~%" label rows cols)
    (dotimes (i rows)
      (format t "  [")
      (dotimes (j cols)
        (format t "~8,3f" (aref M i j)))
      (format t " ]~%"))))

;;; ─── Utility: Dot product ───────────────────────────────────────

(defun dot (u v)
  "Dot product of two 1D vectors U and V."
  (reduce #'+ (map '(vector single-float)
                   (lambda (a b) (* a b)) u v)))

;;; ─── Utility: Cosine similarity ─────────────────────────────────

(defun norm (v)
  "L2 norm of vector V."
  (sqrt (reduce #'+ (map '(vector single-float) (lambda (x) (* x x)) v))))

(defun cosine-similarity (u v)
  "Cosine similarity of U and V: cos θ = (u·v) / (|u| |v|)"
  (/ (dot u v) (* (norm u) (norm v))))

;;; ─── Demo ────────────────────────────────────────────────────────

(defparameter *vocab-size* 100)
(defparameter *d-model* 8)

(let* ((E (make-embedding-matrix *vocab-size* *d-model*))
       ;; Pretend token IDs: "the"=1, "cat"=2, "sat"=3
       (ids '(1 2 3))
       (X (embed-sequence E ids)))
  (print-matrix X "Embedding output X [T=3, d=8]")
  (format t "~%Dot product E[1]·E[2]: ~,4f~%"
          (dot (embed-token E 1) (embed-token E 2)))
  (format t "Cosine similarity E[1]·E[2]: ~,4f~%"
          (cosine-similarity (embed-token E 1) (embed-token E 2))))
```

**Try it:**

```bash
# Run with SBCL:
sbcl --load embeddings.lisp --quit

# Or with CLISP:
clisp embeddings.lisp
```

---

## 2.6 Key Takeaways

- An embedding matrix `E ∈ ℝ^{|V| × d}` maps token IDs to dense vectors.
- The embedding of token `i` is row `i` of `E` — a single **lookup**, no computation.
- For a sequence of `T` tokens, we get a `[T × d]` matrix `X`.
- The vectors are **learned**: training nudges similar tokens closer together.
- Semantic arithmetic (`king − man + woman ≈ queen`) emerges from the training objective.

> **What's next?** We have a matrix `X` of shape `[T × d]`. Each row knows *what* the token is, but nothing about *where* it sits in the sequence. Swapping two tokens would produce the same rows in a different order, which the model would treat identically. We need to inject **position information** — that's Chapter 3.


---

![Embedding matrix lookup — each token ID indexes into a row of the weight matrix.](images/ch02_embedding_lookup.png)

*Embedding matrix lookup — each token ID indexes into a row of the weight matrix.*

![Semantic arithmetic in embedding space — king − man + woman ≈ queen.](images/ch02_semantic_space.png)

*Semantic arithmetic in embedding space — king − man + woman ≈ queen.*

