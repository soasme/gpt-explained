# Chapter 3: Positional Encoding — Giving Order to Meaning

> *"The model sees all tokens at once. Without position, 'dog bites man' and 'man bites dog' are identical."*

After the embedding step we have a matrix $X \in \mathbb{R}^{T\times d}$. Each row `X[t]` is a vector representing token `t`. But here is the problem: attention (coming in Chapter 4) processes all tokens **simultaneously** — there is no inherent notion of "token 3 comes after token 2." If you shuffled the rows, the model would not know.

We fix this by **adding a position vector** to each token's embedding. The combined vector carries both *what* (the token identity) and *where* (the position).

---

## 3.1 The Idea

For position `t` in a sequence, we compute a vector $p_t \in \mathbb{R}^d$ and add it to the corresponding embedding:

```
x̃_t = x_t + p_t
```

The simplest approach: learned position embeddings (like the token embeddings — just a table of vectors, one per position). GPT-2 uses this.

The original "Attention Is All You Need" paper used a more elegant approach: **sinusoidal positional encoding** — a deterministic formula using sine and cosine functions at different frequencies. We will study both, but sinusoidal encoding teaches more about why it works.

> **Math Minute — Sine and Cosine**
> `sin(θ)` and `cos(θ)` oscillate between -1 and 1 as θ increases. Different frequencies oscillate at different rates. Low frequency → slow oscillation (changes slowly with position). High frequency → fast oscillation (changes rapidly). Think of them as a clock: hour hand (low frequency) and second hand (high frequency) together uniquely identify the time.

---

## 3.2 Sinusoidal Positional Encoding

For position `t` and dimension `i` (where $0 \leq i < d$):

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

Full formula for the positional encoding matrix $PE \in \mathbb{R}^{T\times d}$:

$$
PE[t, i] = \sin( t \cdot \omega_i )    if i is even
PE[t, i] = \cos( t \cdot \omega_\lfloor i/2\rfloor) if i is odd

where \omega_k = 1 / 10000^(2k/d)
$$

After building `PE`, the position-encoded input is:

$$
\tilde{X} = X + PE    (element-wise addition, both [T \times d])
$$

The model now works with $\tilde{X}$ instead of `X`.

---

## 3.4 The Matrix: Worked Example

Let `T = 4` (4 tokens), `d = 8` (8-dimensional).

First compute the frequencies $\omega_k$ for `k = 0, 1, 2, 3`:

$$
\omega_0 = 1 / 10000^(0/8) = 1.0
\omega_1 = 1 / 10000^(2/8) = 1 / 10000^0.25 \approx 0.1
\omega_2 = 1 / 10000^(4/8) = 1 / 100 = 0.01
\omega_3 = 1 / 10000^(6/8) = 1 / 10000^0.75 \approx 0.001
$$

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

The final input matrix $\tilde{X} = X + PE$:

```
X̃[t] = embed_vector[t] + PE[t]

For t=0: [-0.90, 0.10, 0.20, -0.30, ...]
       + [ 0.00, 1.00, 0.00,  1.00, ...]
       = [-0.90, 1.10, 0.20,  0.70, ...]
```

![Sinusoidal positional encoding heatmap](images/ch03-positional-encoding.png)

---

## 3.5 The Code: Positional Encoding in Scheme

The matrix data abstraction from Chapter 2 carries over unchanged — we repeat it here so each file is self-contained. Every chapter from this point forward assumes these six procedures are present.

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

The frequency $\omega_k = 1/10000^{2k/d}$ controls how fast dimension-pair $k$ oscillates with position. When $k$ is small (low dimensions), $\omega_k$ is close to 1 and the sine/cosine cycle is fast — the value changes dramatically between adjacent positions. When $k$ is large (high dimensions), $\omega_k$ is tiny and the cycle is very slow, nearly constant across short sequences. The key property is exponential spacing: each successive $k$ slows the oscillation by a constant factor, just as a clock's second, minute, and hour hands each turn ten-times slower than the last.

```scheme
(define (angular-frequency k d)
  (/ 1.0 (expt 10000.0 (/ (* 2.0 k) d))))
```

`pe-value` is the atomic formula: one number at position $t$, dimension $i$, in a $d$-dimensional encoding. Even-indexed dimensions receive sine; odd-indexed dimensions receive cosine. Both are evaluated at $t \cdot \omega_{\lfloor i/2 \rfloor}$. The entire positional encoding is this single expression — everything else is just iteration over it.

```scheme
(define (pe-value t-pos dim d)
  (let* ((k     (quotient dim 2))
         (omega (angular-frequency k d)))
    (if (even? dim)
        (sin (* t-pos omega))
        (cos (* t-pos omega)))))
```

A full positional encoding vector for position $t$ is $d$ calls to `pe-value`, one per dimension. This lifts the scalar formula to the level of a row vector. The procedure follows the same pattern as `embed-token` from Chapter 2: allocate a vector, fill it by index, return it.

```scheme
(define (positional-encoding-row t-pos d)
  (let ((v (make-vector d)))
    (let loop ((i 0))
      (when (< i d)
        (vector-set! v i (pe-value t-pos i d))
        (loop (+ i 1))))
    v))
```

The full PE matrix lifts once more: apply `positional-encoding-row` to each of the $T$ positions and store the results row by row into a $[T \times d]$ matrix. This is the same lifting pattern used in Chapter 2 for `embed-sequence` — building a matrix by stacking row operations.

```scheme
(define (make-pe-matrix max-len d)
  (let ((PE (make-matrix max-len d)))
    (let loop ((t 0))
      (when (< t max-len)
        (let ((row (positional-encoding-row t d)))
          (let copy ((j 0))
            (when (< j d)
              (matrix-set! PE t j (vector-ref row j))
              (copy (+ j 1)))))
        (loop (+ t 1))))
    PE))
```

Adding PE to the token embedding matrix $X$ gives $\tilde{X} = X + PE$, element-wise — both are $[T \times d]$. Each token embedding receives its position's fingerprint fused into its values. From this point forward the model works with $\tilde{X}$, which encodes both what each token is and where it sits in the sequence.

```scheme
(define (add-positional-encoding X PE)
  (let* ((T (matrix-rows X))
         (d (matrix-cols X))
         (X-tilde (make-matrix T d)))
    (let loop ((t 0))
      (when (< t T)
        (let col ((j 0))
          (when (< j d)
            (matrix-set! X-tilde t j
              (+ (matrix-ref X t j) (matrix-ref PE t j)))
            (col (+ j 1))))
        (loop (+ t 1))))
    X-tilde))
```

Cosine similarity gives us a concrete way to check the encoding's behavior: nearby positions should produce more similar vectors than distant ones. We define it here for the demo; the same procedure appears in Chapter 2 and will reappear in later chapters.

```scheme
(define (dot u v)
  (let loop ((i 0) (acc 0.0))
    (if (= i (vector-length u)) acc
        (loop (+ i 1) (+ acc (* (vector-ref u i) (vector-ref v i)))))))

(define (cosine-similarity u v)
  (/ (dot u v) (* (sqrt (dot u u)) (sqrt (dot v v)))))
```

**Demo.** We build a PE matrix for five positions with $d=8$, print every row, then measure the cosine similarity between position 0 and each other position. The similarities should decrease monotonically, confirming that the encoding faithfully captures distance.

```scheme
(let* ((d       8)
       (seq-len 5)
       (PE      (make-pe-matrix seq-len d)))
  (display "Positional Encoding Matrix [5x8]:") (newline)
  (let loop ((t 0))
    (when (< t seq-len)
      (display "  PE[") (display t) (display "]: [")
      (let col ((j 0))
        (when (< j d)
          (display (matrix-ref PE t j)) (display " ")
          (col (+ j 1))))
      (display "]") (newline)
      (loop (+ t 1))))
  (newline)
  (display "Cosine similarities to PE[0]:") (newline)
  (let loop ((t 0))
    (when (< t seq-len)
      (let ((sim (cosine-similarity
                  (positional-encoding-row 0 d)
                  (positional-encoding-row t d))))
        (display "  sim(PE[0], PE[") (display t) (display "]) = ")
        (display sim) (newline))
      (loop (+ t 1)))))
```

Run with `mit-scheme --quiet --load positional-encoding.scm`.

---

## 3.6 Learned vs Sinusoidal Positional Embeddings

GPT-2 and GPT-3 use **learned** positional embeddings — a second embedding matrix $P \in \mathbb{R}^{T_max \times d}$, trained alongside the token embedding matrix. Each position `t` has a learned row `P[t]`.

Pros:
- Can adapt to the specific patterns in the training data.

Cons:
- Cannot generalize to sequences longer than `T_max` (the training context length).

Sinusoidal encoding is deterministic and extrapolates naturally. Modern architectures (LLaMA, Mistral) use a more sophisticated variant called **Rotary Positional Encoding (RoPE)** which applies the positional information inside the attention computation rather than at the input stage.

---

## 3.7 Key Takeaways

- Without position information, the model is **permutation-invariant** — it cannot distinguish `"dog bites man"` from `"man bites dog"`.
- Sinusoidal PE encodes position using sine/cosine at exponentially-spaced frequencies.
- The encoding is **added** to the token embedding: $\tilde{X} = X + PE$.
- Nearby positions have high cosine similarity; distant positions have lower similarity.
- Modern models (GPT-2: learned; LLaMA: RoPE) vary the method but the purpose is the same.

> **What's next?** We now have $\tilde{X} \in \mathbb{R}^{T\times d}$ — a matrix that knows both what each token is and where it sits. Now the interesting part begins: **attention**, the mechanism that lets tokens share information with each other. Chapter 4.


---

![Positional encoding matrix — each row is added to the corresponding token embedding.](images/ch03_pe_matrix.png)

*Positional encoding matrix — each row is added to the corresponding token embedding.*

![Sinusoidal waves at different frequencies encode position uniquely across dimensions.](images/ch03_sin_waves.png)

*Sinusoidal waves at different frequencies encode position uniquely across dimensions.*

