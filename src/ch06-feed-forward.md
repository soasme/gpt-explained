# Chapter 6: Feed-Forward Network — The Model's Memory

> *"Attention decides what to look at. The feed-forward network decides what to think about it."*

After multi-head attention has blended information across positions, each token's vector passes through a **feed-forward network (FFN)**. This is a simple two-layer MLP — the same network, applied independently to every position.

While attention does the *routing* (mixing information across tokens), the FFN does the *processing* (transforming each token's representation). Remarkably, experiments by Geva et al. (2021) showed that the FFN layers act as "key-value memories" — the first layer's neurons trigger on specific input patterns, and the second layer retrieves associated information.

---

## 6.1 The Idea

The FFN takes a single vector $x \in \mathbb{R}^d$ and transforms it:

1. **Expand** it to a higher-dimensional space (`d_ff = 4d` typically).
2. **Apply a non-linearity** (activation function).
3. **Project back** to the model dimension.

$$
FFN(x) = \operatorname{GELU}( x W_1 + b_1 ) W_2 + b_2
$$

- $W_1 \in \mathbb{R}^{d\times d_ff}$ — expansion (usually `d_ff = 4d`)
- $W_2 \in \mathbb{R}^{d_ff\times d}$ — contraction
- `GELU` — the activation function (explained below)

The FFN is applied **position-wise**: every token's vector goes through the same weights independently. No mixing across positions happens here — that already happened in attention.

> **Math Minute — ReLU and GELU**
> A **non-linearity** is a function that makes the network capable of learning complex patterns beyond linear mappings.
>
> `ReLU(x) = max(0, x)` — simple: negative values → 0, positive values unchanged.
>
> $\operatorname{GELU}(x) = x \cdot \Phi(x)$ where $\Phi$ is the standard normal CDF. GELU is smooth and probabilistic: instead of a hard cut at zero, it "softly" gates the input. GPT uses GELU. LLaMA and most modern models use `SwiGLU`, a gated variant.
>
> `Sigmoid(x) = 1/(1+e⁻ˣ)` squashes any real number to (0,1). Used in gates.

---

## 6.2 The Math

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

## 6.5 The Code: FFN in Lisp

```lisp
;;;; feed-forward-network.lisp
;;;; Position-wise Feed-Forward Network with GELU activation

;; (load "attention.lisp")   ; provides mat-mul, linear

;;; ─── GELU activation ─────────────────────────────────────────────

;; We use the tanh approximation (same as used in GPT-2)
(defun gelu (x)
  "GELU activation: x · 0.5 · (1 + tanh(√(2/π) · (x + 0.044715·x³)))"
  (let* ((c (sqrt (/ 2.0 pi)))
         (inner (+ x (* 0.044715 x x x)))
         (t-val (tanh (* c inner))))
    (* x 0.5 (+ 1.0 t-val))))

(defun gelu-matrix (M)
  "Apply GELU element-wise to matrix M."
  (destructuring-bind (rows cols) (array-dimensions M)
    (let ((result (make-array (list rows cols) :element-type 'single-float)))
      (dotimes (i rows result)
        (dotimes (j cols)
          (setf (aref result i j) (gelu (aref M i j))))))))

;;; ─── Bias broadcast helper ───────────────────────────────────────

(defun add-bias (M b)
  "Add bias vector B (length cols) to every row of M [T × cols]."
  (destructuring-bind (rows cols) (array-dimensions M)
    (assert (= cols (length b)) () "Bias length mismatch")
    (let ((result (make-array (list rows cols) :element-type 'single-float)))
      (dotimes (i rows result)
        (dotimes (j cols)
          (setf (aref result i j) (+ (aref M i j) (aref b j))))))))

;;; ─── FFN Parameters ──────────────────────────────────────────────

(defun make-ffn-params (d &optional (d-ff-ratio 4))
  "Create FFN parameters.
   d      : model dimension
   d-ff   : feed-forward inner dimension (default 4×d)
   Returns a plist with :w1 :b1 :w2 :b2."
  (let* ((d-ff (* d d-ff-ratio))
         (scale-1 (sqrt (/ 2.0 (+ d d-ff))))
         (scale-2 (sqrt (/ 2.0 (+ d-ff d))))
         (W1 (let ((m (make-array (list d d-ff) :element-type 'single-float)))
               (dotimes (i d m)
                 (dotimes (j d-ff)
                   (setf (aref m i j) (* scale-1 (- (random 2.0) 1.0)))))))
         (b1 (make-array d-ff :element-type 'single-float :initial-element 0.0))
         (W2 (let ((m (make-array (list d-ff d) :element-type 'single-float)))
               (dotimes (i d-ff m)
                 (dotimes (j d)
                   (setf (aref m i j) (* scale-2 (- (random 2.0) 1.0)))))))
         (b2 (make-array d :element-type 'single-float :initial-element 0.0)))
    (list :w1 W1 :b1 b1 :w2 W2 :b2 b2)))

;;; ─── Forward pass ────────────────────────────────────────────────

(defun ffn-forward (X params)
  "FFN(X) = GELU(X W₁ + b₁) W₂ + b₂
   X      : [T × d]
   Returns: [T × d]"
  (let* ((W1 (getf params :w1))
         (b1 (getf params :b1))
         (W2 (getf params :w2))
         (b2 (getf params :b2))
         ;; Step 1: expand with W1 + bias
         (H (add-bias (mat-mul X W1) b1))
         ;; Step 2: GELU activation
         (H-act (gelu-matrix H))
         ;; Step 3: contract with W2 + bias
         (Y (add-bias (mat-mul H-act W2) b2)))
    Y))

;;; ─── Demo ─────────────────────────────────────────────────────────

(let* ((d 8)
       (T-len 3)
       (params (make-ffn-params d 4))  ; 4× expansion → d_ff = 32
       ;; Sample input (output from attention layer)
       (X (let ((m (make-array (list T-len d) :element-type 'single-float)))
            (dotimes (i T-len m)
              (dotimes (j d)
                (setf (aref m i j) (* 0.1 (- (random 20.0) 10.0))))))))
  (format t "FFN Input X [~a × ~a]:~%" T-len d)
  (dotimes (i T-len)
    (format t "  [")
    (dotimes (j d) (format t "~7,3f" (aref X i j)))
    (format t " ]~%"))
  (let ((Y (ffn-forward X params)))
    (format t "~%FFN Output Y [~a × ~a]:~%" T-len d)
    (dotimes (i T-len)
      (format t "  [")
      (dotimes (j d) (format t "~7,3f" (aref Y i j)))
      (format t " ]~%"))))
```

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

