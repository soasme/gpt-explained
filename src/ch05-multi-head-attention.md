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

Split the model dimension into `H` smaller "sub-spaces." In each sub-space, run independent attention. Concatenate all results. Project back.

Formally, each head `h` has its own weight matrices:

```
Wqʰ ∈ ℝ^{d×dₖ},  Wkʰ ∈ ℝ^{d×dₖ},  Wvʰ ∈ ℝ^{d×dᵥ}
```

And computes:

```
headʰ = Attention(X Wqʰ, X Wkʰ, X Wvʰ)   ∈ ℝ^{T×dᵥ}
```

The standard choice is `dₖ = dᵥ = d/H`, so that the total computation remains comparable to one large attention head.

---

## 5.2 The Math

**Step 1 — Compute each head independently:**

```
For h = 1, …, H:
  Qʰ = X Wqʰ    ∈ ℝ^{T×dₖ}
  Kʰ = X Wkʰ    ∈ ℝ^{T×dₖ}
  Vʰ = X Wvʰ    ∈ ℝ^{T×dᵥ}
  headʰ = softmax( QʰKʰᵀ / √dₖ + M ) Vʰ
```

**Step 2 — Concatenate:**

```
MultiHead = concat(head¹, head², …, headᴴ)   ∈ ℝ^{T × (H·dᵥ)}
```

Since `dᵥ = d/H`, the concatenated output has shape `[T × d]` — the same as the input.

**Step 3 — Output projection:**

```
Wo ∈ ℝ^{d×d}   (learned)

Output = MultiHead · Wo   ∈ ℝ^{T×d}
```

The final output projection mixes information across heads.

**Full formula:**

```
MHA(X) = [head¹ ‖ head² ‖ … ‖ headᴴ] Wo

where headʰ = Attention(X Wqʰ, X Wkʰ, X Wvʰ)
```

---

## 5.3 The Matrix: Worked Example

Let `T = 3`, `d = 4`, `H = 2` heads, so `dₖ = dᵥ = 2`.

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

Scores: `S¹ = Q¹K¹ᵀ / √2`:
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

**Output projection** with `Wo ∈ ℝ^{4×4}` mixes the heads' outputs.

![Multi-head attention: two heads capturing different relationship types](images/ch05-multi-head-attention.png)

---

## 5.4 The Code: Multi-Head Attention in Lisp

```lisp
;;;; multi-head-attention.lisp
;;;; Multi-head attention = H parallel attention heads + output projection

;; (load "attention.lisp")   ; provides scaled-dot-product-attention, mat-mul, etc.

;;; ─── Data Structure ──────────────────────────────────────────────
;;
;; An ATTENTION-HEAD is a plist with keys:
;;   :wq [d × dk]    :wk [d × dk]    :wv [d × dv]
;;
;; MHA-PARAMS is a plist with keys:
;;   :heads — list of ATTENTION-HEAD plists
;;   :wo    — output projection [d × d]

(defun make-mha-params (d num-heads)
  "Create Multi-Head Attention parameters for a D-dimensional model with NUM-HEADS heads.
   Each head operates in dimension dk = dv = d/num-heads."
  (assert (zerop (mod d num-heads))
          () "d (~a) must be divisible by num-heads (~a)" d num-heads)
  (let* ((dk (/ d num-heads))
         (heads (loop repeat num-heads
                      collect (list :wq (rand-weight-matrix d dk)
                                    :wk (rand-weight-matrix d dk)
                                    :wv (rand-weight-matrix d dk)))))
    (list :heads heads
          :wo   (rand-weight-matrix d d))))

;;; ─── Matrix concatenation (column-wise) ─────────────────────────

(defun hstack (matrices)
  "Concatenate matrices column-wise.  All must have same row count.
   E.g., [3×2] [3×2] → [3×4]."
  (let* ((rows  (array-dimension (car matrices) 0))
         (total-cols (reduce #'+ matrices
                             :key (lambda (m) (array-dimension m 1))))
         (result (make-array (list rows total-cols)
                             :element-type 'single-float))
         (col-offset 0))
    (dolist (M matrices result)
      (let ((cols (array-dimension M 1)))
        (dotimes (i rows)
          (dotimes (j cols)
            (setf (aref result i (+ col-offset j))
                  (aref M i j))))
        (incf col-offset cols)))))

;;; ─── Multi-Head Attention ────────────────────────────────────────

(defun multi-head-attention (X params)
  "Compute Multi-Head Attention for input X [T × d] using PARAMS.

   For each head h:
     Qh = X Wqh,  Kh = X Wkh,  Vh = X Wvh
     headh = Attention(Qh, Kh, Vh)

   Output = concat(head1, ..., headH) · Wo
   Returns (values output all-attention-weights)."
  (let* ((heads (getf params :heads))
         (Wo    (getf params :wo))
         (head-outputs '())
         (all-weights  '()))
    ;; Run each head
    (dolist (head heads)
      (let* ((Wq (getf head :wq))
             (Wk (getf head :wk))
             (Wv (getf head :wv))
             (Q  (linear X Wq))
             (K  (linear X Wk))
             (V  (linear X Wv)))
        (multiple-value-bind (out weights)
            (scaled-dot-product-attention Q K V :mask t)
          (push out head-outputs)
          (push weights all-weights))))
    ;; Concatenate head outputs: [T × (H·dv)]
    (let* ((concat-out (hstack (reverse head-outputs)))
           ;; Project: [T × d] · [d × d] = [T × d]
           (output (mat-mul concat-out Wo)))
      (values output (reverse all-weights)))))

;;; ─── Demo ─────────────────────────────────────────────────────────

(defun print-head-weights (weights)
  (loop for w in weights
        for h from 0 do
    (format t "Head ~a attention weights [~a×~a]:~%"
            h (array-dimension w 0) (array-dimension w 1))
    (dotimes (i (array-dimension w 0))
      (format t "  [")
      (dotimes (j (array-dimension w 1))
        (format t "~7,3f" (aref w i j)))
      (format t " ]~%"))))

(let* ((d 8)
       (num-heads 2)
       (T-len 4)
       (params (make-mha-params d num-heads))
       ;; Random input
       (X (let ((m (make-array (list T-len d) :element-type 'single-float)))
            (dotimes (i T-len m)
              (dotimes (j d) (setf (aref m i j) (- (random 2.0) 1.0)))))))
  (multiple-value-bind (output weights)
      (multi-head-attention X params)
    (print-head-weights weights)
    (format t "~%MHA Output [~a × ~a]:~%"
            (array-dimension output 0) (array-dimension output 1))
    (dotimes (i T-len)
      (format t "  [")
      (dotimes (j d) (format t "~7,3f" (aref output i j)))
      (format t " ]~%"))))
```

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

- Multi-head attention runs `H` attention heads **in parallel**, each in a lower-dimensional subspace `dₖ = d/H`.
- Each head has independent `Wq, Wk, Wv` matrices — each learns a different "question to ask."
- Outputs are **concatenated** (not averaged), then projected back to `d` with `Wo`.
- The total parameter count is `3Hd·dₖ + d² = 3d² + d²` — same as one large head.
- Different heads specialize in different linguistic relationships.

> **What's next?** After attention mixes information across tokens, each token's vector goes through a small **feed-forward network** — a two-layer MLP applied identically to every position. This is where most of the model's stored "knowledge" lives. Chapter 6.


---

![Multi-head attention — H parallel heads each attend independently, then concat + project.](images/ch05_mha.png)

*Multi-head attention — H parallel heads each attend independently, then concat + project.*

