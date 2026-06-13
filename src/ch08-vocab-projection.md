# Chapter 8: Vocabulary Projection — From Vectors to Words

> *"After all that computation, the model must answer a simple question: what word comes next?"*

After `N` transformer blocks, we have a matrix `X_final ∈ ℝ^{T×d}`. The final row, `X_final[T-1]`, is a rich contextualized representation of the last token — it "knows" everything in the context window. The last step is to project this vector into a probability distribution over the vocabulary: which token is most likely to come next?

This projection is called the **language model head**, or **unembedding layer**.

---

## 8.1 The Idea

The unembedding is the inverse of the embedding:

- **Embedding:** integer → vector. Look up row `i` in `E ∈ ℝ^{|V|×d}`.
- **Unembedding:** vector → distribution over integers. Project `x ∈ ℝᵈ` to `logits ∈ ℝ^{|V|}`, then softmax.

The projection is done with a weight matrix `Wᵤ ∈ ℝ^{d×|V|}`:

```
logits = x Wᵤ   ∈ ℝ^{|V|}
```

Each `logits[i]` is the **unnormalized score** for vocabulary entry `i`. Higher score = the model thinks token `i` is more likely next.

Most GPT implementations **tie** the unembedding weights to the embedding matrix: `Wᵤ = Eᵀ`. The same matrix is used for both embedding (rows as vectors) and unembedding (columns as classifiers). This **weight tying** reduces parameters and has been shown to improve performance.

---

## 8.2 The Math

Given the final hidden state `hₜ = X_final[T-1] ∈ ℝᵈ`:

**Step 1 — Final Layer Norm:**

```
h̃ₜ = LayerNorm(hₜ)    ∈ ℝᵈ
```

(Applied once more before the unembedding projection.)

**Step 2 — Vocabulary Projection:**

```
logits = h̃ₜ Wᵤ    ∈ ℝ^{|V|}
```

With weight tying: `Wᵤ = Eᵀ`, so:

```
logits[i] = h̃ₜ · Eᵢ   (dot product with the embedding of token i)
```

The logit for token `i` is the dot product of the model's "prediction vector" with token `i`'s embedding. Tokens whose embeddings align with the prediction vector get high logits.

**Step 3 — Softmax → Probabilities:**

```
P(next = i | context) = softmax(logits)[i] = exp(logits[i]) / Σⱼ exp(logits[j])
```

> **Math Minute — Temperature**
> We can control the "sharpness" of the distribution with a **temperature parameter** `T_temp`:
> ```
> P(i) = softmax(logits / T_temp)[i]
> ```
> - `T_temp → 0`: argmax — always pick the highest-probability token (greedy)
> - `T_temp = 1`: standard softmax
> - `T_temp > 1`: flattens the distribution — more randomness/creativity
>
> Temperature is not a model parameter — it's a sampling hyperparameter set at inference time.

---

## 8.3 The Generation Loop

Training and inference use the same forward pass differently.

**Training:** Given a sequence `[t₁, t₂, …, tₙ]`, the model predicts all next tokens simultaneously (thanks to causal masking): `P(t₂|t₁), P(t₃|t₁,t₂), …, P(tₙ|t₁,…,tₙ₋₁)`. The loss is cross-entropy averaged over all positions.

**Inference (generation):**
```
1. Start with prompt tokens [t₁, …, tₙ]
2. Forward pass → logits for position n
3. Sample/argmax → tₙ₊₁
4. Append tₙ₊₁ to sequence
5. Forward pass → logits for position n+1
6. Repeat until stop token or max length
```

This is called **autoregressive generation**: each new token is fed back in as input to generate the next.

---

## 8.4 Loss: Cross-Entropy

During training, for each position `t`, the model produces a probability distribution. The **cross-entropy loss** measures how far the prediction is from the true next token:

```
L(t) = -log P(tₜ₊₁ | t₁, …, tₜ)
```

If the model assigns probability `0.8` to the true next token: `L = -log(0.8) ≈ 0.22`. 
If the model assigns `0.001`: `L = -log(0.001) ≈ 6.9`. 

The total loss is the mean over all positions and all training examples. Minimizing this via gradient descent is exactly what shapes all the weights in the model — the embedding matrix, the attention weight matrices, the FFN weights — everything.

> **Math Minute — Log**
> `log(p)` for `p ∈ (0,1]` is always negative. As `p → 0`, `log(p) → -∞`. As `p → 1`, `log(p) → 0`. The negative log loss is 0 when the prediction is perfect and large when the model is confident about the wrong answer.

---

## 8.5 The Matrix: Worked Example

Let `|V| = 5`, `d = 4`. Final hidden state:

```
h = [0.3, -0.1, 0.8, 0.2]
```

Unembedding matrix `Wᵤ = Eᵀ` (4×5, columns = token embeddings):

```
Eᵀ[:,0] = [0.10, 0.50, -0.90, 0.40]   ← embedding of token 0
Eᵀ[:,1] = [-0.20, 0.60, 0.10, -0.50]  ← embedding of token 1
Eᵀ[:,2] = [0.30, -0.70, 0.20, 0.60]   ← embedding of token 2
Eᵀ[:,3] = [-0.40, 0.80, -0.30, -0.70] ← embedding of token 3
Eᵀ[:,4] = [0.50, -0.40, 0.20, 0.30]   ← embedding of token 4 (hypothetical)
```

Wait — Eᵀ has shape `[d × |V|]`. Projecting `h [1×d]` → `[1×|V|]`:

```
logits[0] = h · E[0] = (0.3)(0.10) + (-0.1)(−0.20) + (0.8)(0.30) + (0.2)(−0.40)
          = 0.030 + 0.020 + 0.240 − 0.080 = 0.210

logits[1] = h · E[1] = (0.3)(0.50) + (-0.1)(0.60) + (0.8)(−0.70) + (0.2)(0.80)
          = 0.150 − 0.060 − 0.560 + 0.160 = −0.310

logits[2] = h · E[2] = (0.3)(−0.90) + (-0.1)(0.10) + (0.8)(0.20) + (0.2)(−0.30)
          = −0.270 − 0.010 + 0.160 − 0.060 = −0.180

logits[3] = h · E[3] = (0.3)(0.40) + (-0.1)(−0.50) + (0.8)(0.60) + (0.2)(−0.70)
          = 0.120 + 0.050 + 0.480 − 0.140 = 0.510

logits[4] = h · E[4] = (0.3)(−0.10) + (-0.1)(0.80) + (0.8)(−0.40) + (0.2)(0.50)
          = −0.030 − 0.080 − 0.320 + 0.100 = −0.330

logits = [0.210, -0.310, -0.180, 0.510, -0.330]
```

**Softmax:**
```
exp(logits) = [1.233, 0.733, 0.835, 1.665, 0.719]
sum         = 5.185

P = [0.238, 0.141, 0.161, 0.321, 0.139]
```

**Token 3 has the highest probability (32.1%)** — that's the model's best guess for next token.

![Vocabulary projection: hidden state → logits → softmax probabilities](images/ch08-vocab-projection.png)

---

## 8.6 The Code: Full microGPT Forward Pass in Lisp

```lisp
;;;; microgpt.lisp
;;;; Complete forward pass: tokens → probabilities
;;;; Assembles all previous chapters into a working microGPT

;; Dependencies (load in order):
;; (load "embeddings.lisp")
;; (load "positional-encoding.lisp")
;; (load "attention.lisp")
;; (load "multi-head-attention.lisp")
;; (load "feed-forward-network.lisp")
;; (load "transformer-block.lisp")

;;; ─── GPT Hyperparameters ─────────────────────────────────────────

(defstruct gpt-config
  vocab-size    ; |V|: number of tokens in vocabulary
  d-model       ; d: hidden dimension (e.g., 768 for GPT-2 small)
  num-heads     ; H: number of attention heads
  num-layers    ; N: number of transformer blocks
  max-seq-len   ; T_max: maximum sequence length
  )

(defparameter *microgpt-config*
  (make-gpt-config :vocab-size 50
                   :d-model 16
                   :num-heads 4
                   :num-layers 2
                   :max-seq-len 32))

;;; ─── GPT Parameters ──────────────────────────────────────────────

(defun make-gpt-params (config)
  "Initialize all parameters for a microGPT model."
  (let* ((V (gpt-config-vocab-size  config))
         (d (gpt-config-d-model     config))
         (H (gpt-config-num-heads   config))
         (N (gpt-config-num-layers  config))
         (T (gpt-config-max-seq-len config)))
    (list
     ;; Token embedding matrix [V × d]
     :wte (make-embedding-matrix V d)
     ;; Position embedding matrix [T × d] (learned, like GPT-2)
     :wpe (make-embedding-matrix T d)
     ;; Stack of N transformer blocks
     :blocks (make-transformer-stack N d H)
     ;; Final layer norm
     :ln-f (make-ln-params d)
     ;; NOTE: unembedding (lm_head) reuses :wte  transposed (weight tying)
     )))

;;; ─── Forward Pass ─────────────────────────────────────────────────

(defun gpt-forward (token-ids params config)
  "Full GPT forward pass.
   token-ids : list of integer token IDs, length T
   params    : from make-gpt-params
   Returns   : logits [|V|] for the LAST position (next-token prediction)"
  (let* ((T-len (length token-ids))
         (d     (gpt-config-d-model config))
         (wte   (getf params :wte))
         (wpe   (getf params :wpe))
         (blocks(getf params :blocks))
         (ln-f  (getf params :ln-f))

         ;; 1. Token embeddings: [T × d]
         (tok-emb (embed-sequence wte token-ids))

         ;; 2. Position embeddings: [T × d]
         (pos-emb (embed-sequence wpe (loop for i below T-len collect i)))

         ;; 3. Add: X = token_emb + pos_emb
         (X (mat-add tok-emb pos-emb))

         ;; 4. Pass through N transformer blocks
         (X-final (forward-transformer-stack X blocks))

         ;; 5. Final LayerNorm on last token
         (last-row (let ((v (make-array d :element-type 'single-float)))
                     (dotimes (j d v) (setf (aref v j) (aref X-final (1- T-len) j)))))
         (h (layer-norm-1d last-row (getf ln-f :gamma) (getf ln-f :beta)))

         ;; 6. Unembedding: logits = h · Eᵀ
         ;; Weight tying: use wte transposed as classifier
         ;; logits[i] = dot(h, wte[i])
         (V-size (gpt-config-vocab-size config))
         (logits (make-array V-size :element-type 'single-float)))

    (dotimes (i V-size logits)
      (setf (aref logits i)
            (let ((sum 0.0))
              (dotimes (j d sum)
                (incf sum (* (aref h j) (aref wte i j))))))))  )

;;; ─── Softmax (1D) ────────────────────────────────────────────────

(defun softmax-1d (v)
  "Apply softmax to a 1D array, returning probabilities."
  (let* ((n   (length v))
         (max-v (reduce #'max v))
         (exps (make-array n :element-type 'single-float))
         (sum  0.0))
    (dotimes (i n)
      (setf (aref exps i) (exp (- (aref v i) max-v)))
      (incf sum (aref exps i)))
    (dotimes (i n exps)
      (setf (aref exps i) (/ (aref exps i) sum)))))

;;; ─── Sampling ─────────────────────────────────────────────────────

(defun sample-token (probs)
  "Sample a token index from probability distribution PROBS."
  (let ((r (random 1.0))
        (cumsum 0.0))
    (dotimes (i (length probs) (1- (length probs)))
      (incf cumsum (aref probs i))
      (when (>= cumsum r) (return i)))))

(defun top-k-logits (logits k)
  "Zero out all but top-K logits (set to -1e9)."
  (let* ((n (length logits))
         (indexed (loop for i below n collect (cons i (aref logits i))))
         (sorted (sort indexed #'> :key #'cdr))
         (result (make-array n :element-type 'single-float
                               :initial-element -1.0e9)))
    (loop for (idx . val) in (subseq sorted 0 (min k n)) do
      (setf (aref result idx) val))
    result))

;;; ─── Text Generation Loop ────────────────────────────────────────

(defun generate (params config start-ids &key
                                           (max-new-tokens 10)
                                           (temperature 1.0)
                                           (top-k 10))
  "Autoregressive text generation.
   start-ids       : initial context token IDs
   max-new-tokens  : how many tokens to generate
   temperature     : sampling temperature (1.0 = standard, <1 = sharper)
   top-k           : only sample from top-K tokens

   Returns list of generated token IDs (excluding prompt)."
  (let ((tokens (copy-list start-ids))
        (generated '()))
    (dotimes (_ max-new-tokens (reverse generated))
      ;; Forward pass on current token sequence
      (let* ((logits (gpt-forward tokens params config))
             ;; Apply temperature
             (scaled (let ((v (copy-seq logits)))
                       (dotimes (i (length v) v)
                         (setf (aref v i) (/ (aref v i) temperature)))))
             ;; Top-K filtering
             (filtered (top-k-logits scaled top-k))
             ;; Softmax → probabilities
             (probs (softmax-1d filtered))
             ;; Sample
             (next-token (sample-token probs)))
        (push next-token generated)
        (setf tokens (append tokens (list next-token)))))))

;;; ─── Parameter count ──────────────────────────────────────────────

(defun count-parameters (config)
  "Compute total number of scalar parameters in the model."
  (let* ((V (gpt-config-vocab-size  config))
         (d (gpt-config-d-model     config))
         (H (gpt-config-num-heads   config))
         (N (gpt-config-num-layers  config))
         (T (gpt-config-max-seq-len config))
         ;; Embeddings
         (wte-params (* V d))
         (wpe-params (* T d))
         ;; Per block: MHA = 4 × d² (Wq,Wk,Wv,Wo), FFN = 8d², LN = 2×2d
         (block-params (* N (+ (* 4 d d) (* 8 d d) (* 4 d))))
         ;; Final LN
         (ln-f-params (* 2 d)))
    (+ wte-params wpe-params block-params ln-f-params)))

;;; ─── Demo ─────────────────────────────────────────────────────────

(let* ((config *microgpt-config*)
       (params (make-gpt-params config))
       (prompt '(3 7 12 5))    ; pretend token IDs for "the cat sat on"
       (generated (generate params config prompt
                             :max-new-tokens 5
                             :temperature 0.8
                             :top-k 5)))
  (format t "microGPT (~a params)~%"
          (count-parameters config))
  (format t "Prompt tokens: ~a~%" prompt)
  (format t "Generated tokens: ~a~%" generated)
  (format t "Full sequence: ~a~%" (append prompt generated)))
```

---

## 8.7 Weight Tying in Detail

Why does weight tying work so well? Consider:

- The embedding `E[i]` is learned to represent token `i` such that tokens that appear in similar contexts have similar embeddings.
- The unembedding `logits[i] = h · E[i]` measures the alignment between the model's prediction vector `h` and token `i`'s embedding.
- If `h` is pointing in the direction of "tokens that follow this context," and `E[i]` represents token `i`'s meaning, then tokens that are semantically appropriate will naturally score higher.

Weight tying forces the model to use a single geometric space for both "what a token means" and "how to predict a token" — an elegant constraint that regularizes learning.

---

## 8.8 Key Takeaways

- The **unembedding** projects the final hidden state to logits: `logits = h Wᵤ ∈ ℝ^{|V|}`.
- **Weight tying** (`Wᵤ = Eᵀ`) reuses the embedding matrix and reduces parameters.
- **Softmax** converts logits to a probability distribution.
- **Temperature** controls sampling sharpness; **top-K** restricts the candidate set.
- **Autoregressive generation**: sample one token at a time, feeding each back as input.
- **Training loss** is cross-entropy: `-log P(true next token | context)`.

---

## 8.9 The Complete Picture

You have now seen every piece:

```
Input text
   ↓ tokenizer
Token IDs: [3, 7, 12, 5, …]
   ↓ embedding matrix E
Token embeddings: [T × d]
   ↓ + positional encoding
X̃ = X + PE: [T × d]
   ↓ transformer block × N
   │   LayerNorm
   │   Multi-Head Attention (Q, K, V projections + causal mask + softmax + weighted sum)
   │   Residual
   │   LayerNorm
   │   Feed-Forward Network (expand → GELU → contract)
   │   Residual
   ↓
X_final: [T × d]
   ↓ final LayerNorm + unembedding
Logits: [|V|]
   ↓ softmax (+ temperature + top-K)
P(next token): [|V|]  →  sample  →  next token ID
```

Every box in that diagram corresponds to one chapter of this book.

> **Continue reading:** The Appendix assembles all Lisp code into a single runnable `microgpt.lisp` and walks through training on a tiny dataset. The Math Primer covers every mathematical prerequisite in one place.
