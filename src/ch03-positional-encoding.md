# Chapter 3: Positional Encoding — Giving Order to Meaning

> *"The model sees all tokens at once. Without position, 'dog bites man' and 'man bites dog' are identical."*

After the embedding step we have a matrix `X ∈ ℝ^{T×d}`. Each row `X[t]` is a vector representing token `t`. But here is the problem: attention (coming in Chapter 4) processes all tokens **simultaneously** — there is no inherent notion of "token 3 comes after token 2." If you shuffled the rows, the model would not know.

We fix this by **adding a position vector** to each token's embedding. The combined vector carries both *what* (the token identity) and *where* (the position).

---

## 3.1 The Idea

For position `t` in a sequence, we compute a vector `p_t ∈ ℝᵈ` and add it to the corresponding embedding:

```
x̃_t = x_t + p_t
```

The simplest approach: learned position embeddings (like the token embeddings — just a table of vectors, one per position). GPT-2 uses this.

The original "Attention Is All You Need" paper used a more elegant approach: **sinusoidal positional encoding** — a deterministic formula using sine and cosine functions at different frequencies. We will study both, but sinusoidal encoding teaches more about why it works.

> **Math Minute — Sine and Cosine**
> `sin(θ)` and `cos(θ)` oscillate between -1 and 1 as θ increases. Different frequencies oscillate at different rates. Low frequency → slow oscillation (changes slowly with position). High frequency → fast oscillation (changes rapidly). Think of them as a clock: hour hand (low frequency) and second hand (high frequency) together uniquely identify the time.

---

## 3.2 Sinusoidal Positional Encoding

For position `t` and dimension `i` (where `0 ≤ i < d`):

```
PE(t, 2i)   = sin( t / 10000^(2i/d) )
PE(t, 2i+1) = cos( t / 10000^(2i/d) )
```

Even-indexed dimensions use sine; odd-indexed dimensions use cosine. The denominator `10000^(2i/d)` controls the frequency — it grows exponentially with `i`, so low dimensions have high frequency (vary rapidly with `t`) and high dimensions have low frequency (vary slowly).

**Why this formula works:**

1. **Unique per position:** Every `t` produces a distinct vector.
2. **Bounded:** All values lie in `[−1, 1]`, so they do not dominate the token embeddings.
3. **Smooth:** Nearby positions have similar encodings, so the model can learn "close in position → related in meaning."
4. **Extrapolatable:** The formula works for any `t`, even positions not seen during training.

**A key property:** The encoding for position `t+k` can be expressed as a **linear transformation** of the encoding for position `t`. This means the model can learn to attend to "the token 3 positions back" using a simple linear operation — no memorization of absolute positions required.

---

## 3.3 The Math

Full formula for the positional encoding matrix `PE ∈ ℝ^{T×d}`:

```
PE[t, i] = sin( t · ω_i )    if i is even
PE[t, i] = cos( t · ω_⌊i/2⌋) if i is odd

where ω_k = 1 / 10000^(2k/d)
```

After building `PE`, the position-encoded input is:

```
X̃ = X + PE    (element-wise addition, both [T × d])
```

The model now works with `X̃` instead of `X`.

---

## 3.4 The Matrix: Worked Example

Let `T = 4` (4 tokens), `d = 8` (8-dimensional).

First compute the frequencies `ω_k` for `k = 0, 1, 2, 3`:

```
ω₀ = 1 / 10000^(0/8) = 1.0
ω₁ = 1 / 10000^(2/8) = 1 / 10000^0.25 ≈ 0.1
ω₂ = 1 / 10000^(4/8) = 1 / 100 = 0.01
ω₃ = 1 / 10000^(6/8) = 1 / 10000^0.75 ≈ 0.001
```

Now compute PE row by row (position `t = 0, 1, 2, 3`), column by column:

```
PE[0] = [sin(0·1.0), cos(0·1.0), sin(0·0.1), cos(0·0.1), ...]
      = [0.000,      1.000,       0.000,       1.000,       0.000, 1.000, 0.000, 1.000]

PE[1] = [sin(1·1.0), cos(1·1.0), sin(1·0.1), cos(1·0.1), ...]
      = [0.841,      0.540,       0.0998,      0.995,       0.00999, 0.99995, ...]

PE[2] = [sin(2·1.0), cos(2·1.0), sin(2·0.1), cos(2·0.1), ...]
      = [0.909,     -0.416,       0.198,       0.980,       0.01999, 0.99980, ...]

PE[3] = [sin(3·1.0), cos(3·1.0), sin(3·0.1), cos(3·0.1), ...]
      = [0.141,     -0.990,       0.296,       0.955,       0.02999, 0.99955, ...]
```

Notice: dimension 0-1 (high frequency) changes dramatically between positions. Dimension 6-7 (low frequency) barely changes. Together they form a **unique fingerprint** for each position.

The final input matrix `X̃ = X + PE`:

```
X̃[t] = embed_vector[t] + PE[t]

For t=0: [-0.90, 0.10, 0.20, -0.30, ...]
       + [ 0.00, 1.00, 0.00,  1.00, ...]
       = [-0.90, 1.10, 0.20,  0.70, ...]
```

![Sinusoidal positional encoding heatmap](images/ch03-positional-encoding.png)

---

## 3.5 The Code: Positional Encoding in Lisp

```lisp
;;;; positional-encoding.lisp
;;;; Sinusoidal positional encoding (Vaswani et al. 2017)

(defparameter *pe-base* 10000.0)

;;; ─── Frequency computation ───────────────────────────────────────

(defun omega (k d-model)
  "Compute angular frequency for dimension-pair k in a d-model-dimensional space.
   ω_k = 1 / 10000^(2k/d)"
  (/ 1.0 (expt *pe-base* (/ (* 2.0 k) d-model))))

;;; ─── Single position encoding ────────────────────────────────────

(defun positional-encoding-row (t-pos d-model)
  "Compute the positional encoding vector for position T-POS.
   Returns a 1D vector of length D-MODEL."
  (let ((pe (make-array d-model :element-type 'single-float)))
    (loop for i below d-model
          do (let* ((k (floor i 2))
                    (w (omega k d-model)))
               (setf (aref pe i)
                     (if (evenp i)
                         (sin (* t-pos w))
                         (cos (* t-pos w))))))
    pe))

;;; ─── Full PE matrix ──────────────────────────────────────────────

(defun make-pe-matrix (max-seq-len d-model)
  "Build the PE matrix of shape [max-seq-len × d-model].
   PE[t, i] = sin(t·ω_{i/2}) for even i
              cos(t·ω_{i/2}) for odd i"
  (let ((PE (make-array (list max-seq-len d-model)
                        :element-type 'single-float)))
    (dotimes (t-pos max-seq-len PE)
      (let ((row (positional-encoding-row t-pos d-model)))
        (dotimes (j d-model)
          (setf (aref PE t-pos j) (aref row j)))))))

;;; ─── Add PE to embedding matrix ──────────────────────────────────

(defun add-positional-encoding (X PE)
  "Add positional encoding PE to embedding matrix X.
   Both have shape [T × d]. Returns X̃ = X + PE[:T, :]."
  (destructuring-bind (T-len d-model) (array-dimensions X)
    (let ((X-tilde (make-array (list T-len d-model)
                               :element-type 'single-float)))
      (dotimes (t T-len)
        (dotimes (j d-model)
          (setf (aref X-tilde t j)
                (+ (aref X t j) (aref PE t j)))))
      X-tilde)))

;;; ─── Demonstrate position uniqueness ─────────────────────────────

(defun dot-product-1d (u v)
  "Dot product of two 1D arrays."
  (let ((acc 0.0))
    (dotimes (i (length u) acc)
      (incf acc (* (aref u i) (aref v i))))))

(defun cosine-sim (u v)
  (let ((dot (dot-product-1d u v))
        (nu (sqrt (dot-product-1d u u)))
        (nv (sqrt (dot-product-1d v v))))
    (/ dot (* nu nv))))

;;; ─── Demo ─────────────────────────────────────────────────────────

(let* ((d-model 8)
       (seq-len 5)
       (PE (make-pe-matrix seq-len d-model)))
  (format t "Positional Encoding Matrix [~a × ~a]:~%" seq-len d-model)
  (dotimes (t seq-len)
    (format t "  PE[~a]: [" t)
    (dotimes (j d-model)
      (format t "~7,3f " (aref PE t j)))
    (format t "]~%"))
  ;; Show that nearby positions are more similar than far positions
  (format t "~%Cosine similarities to position 0:~%")
  (dotimes (t seq-len)
    (let ((p0 (positional-encoding-row 0 d-model))
          (pt (positional-encoding-row t d-model)))
      (format t "  sim(PE[0], PE[~a]) = ~,4f~%" t (cosine-sim p0 pt)))))
```

---

## 3.6 Learned vs Sinusoidal Positional Embeddings

GPT-2 and GPT-3 use **learned** positional embeddings — a second embedding matrix `P ∈ ℝ^{T_max × d}`, trained alongside the token embedding matrix. Each position `t` has a learned row `P[t]`.

Pros:
- Can adapt to the specific patterns in the training data.

Cons:
- Cannot generalize to sequences longer than `T_max` (the training context length).

Sinusoidal encoding is deterministic and extrapolates naturally. Modern architectures (LLaMA, Mistral) use a more sophisticated variant called **Rotary Positional Encoding (RoPE)** which applies the positional information inside the attention computation rather than at the input stage.

---

## 3.7 Key Takeaways

- Without position information, the model is **permutation-invariant** — it cannot distinguish `"dog bites man"` from `"man bites dog"`.
- Sinusoidal PE encodes position using sine/cosine at exponentially-spaced frequencies.
- The encoding is **added** to the token embedding: `X̃ = X + PE`.
- Nearby positions have high cosine similarity; distant positions have lower similarity.
- Modern models (GPT-2: learned; LLaMA: RoPE) vary the method but the purpose is the same.

> **What's next?** We now have `X̃ ∈ ℝ^{T×d}` — a matrix that knows both what each token is and where it sits. Now the interesting part begins: **attention**, the mechanism that lets tokens share information with each other. Chapter 4.
