# Chapter 8: Vocabulary Projection — From Vectors to Words

> *"After all that computation, the model must answer a simple question: what word comes next?"*

After `N` transformer blocks, we have a matrix $X_final \in \mathbb{R}^{T\times d}$. The final row, `X_final[T-1]`, is a rich contextualized representation of the last token — it "knows" everything in the context window. The last step is to project this vector into a probability distribution over the vocabulary: which token is most likely to come next?

This projection is called the **language model head**, or **unembedding layer**.

---

## 8.1 The Idea

The unembedding is the inverse of the embedding:

- **Embedding:** integer → vector. Look up row `i` in $E \in \mathbb{R}^{|V|\times d}$.
- **Unembedding:** vector → distribution over integers. Project $x \in \mathbb{R}^d$ to $logits \in \mathbb{R}^{|V|}$, then softmax.

The projection is done with a weight matrix $W_u \in \mathbb{R}^{d\times |V|}$:

$$
logits = x W_u   \in \mathbb{R}^{|V|}
$$

Each `logits[i]` is the **unnormalized score** for vocabulary entry `i`. Higher score = the model thinks token `i` is more likely next.

Most GPT implementations **tie** the unembedding weights to the embedding matrix: $W_u = E^{\top}$. The same matrix is used for both embedding (rows as vectors) and unembedding (columns as classifiers). This **weight tying** reduces parameters and has been shown to improve performance.

---

## 8.2 The Math

Given the final hidden state $h_t = X_final[T-1] \in \mathbb{R}^d$:

**Step 1 — Final Layer Norm:**

$$
\tilde{h}_t = \operatorname{LayerNorm}(h_t)    \in \mathbb{R}^d
$$

(Applied once more before the unembedding projection.)

**Step 2 — Vocabulary Projection:**

$$
logits = \tilde{h}_t W_u    \in \mathbb{R}^{|V|}
$$

With weight tying: $W_u = E^{\top}$, so:

$$
logits[i] = \tilde{h}_t \cdot E_i   (dot product with the embedding of token i)
$$

The logit for token `i` is the dot product of the model's "prediction vector" with token `i`'s embedding. Tokens whose embeddings align with the prediction vector get high logits.

**Step 3 — Softmax → Probabilities:**

$$
P(next = i | context) = \operatorname{softmax}(logits)[i] = \exp(logits[i]) / Σ_j \exp(logits[j])
$$

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

**Training:** Given a sequence $[t_1, t_2, \ldots, t_n]$, the model predicts all next tokens simultaneously (thanks to causal masking): $P(t_2|t_1), P(t_3|t_1,t_2), \ldots, P(t_n|t_1,\ldots,t_{n-1})$. The loss is cross-entropy averaged over all positions.

**Inference (generation):**
$$
1. Start with prompt tokens [t_1, \ldots, t_n]
2. Forward pass \to logits for position n
3. Sample/\operatorname{argmax} \to t_{n+1}
4. Append t_{n+1} to sequence
5. Forward pass \to logits for position n+1
6. Repeat until stop token or max length
$$

This is called **autoregressive generation**: each new token is fed back in as input to generate the next.

---

> **Training loss** is covered in depth in [Chapter 9: Loss](./ch09-loss.md). The short version: the model is penalized by $-\log P(\text{true next token})$ at every position, and gradient descent nudges all weights to reduce this number.

---

## 8.5 The Matrix: Worked Example

Let `|V| = 5`, `d = 4`. Final hidden state:

```
h = [0.3, -0.1, 0.8, 0.2]
```

Unembedding matrix $W_u = E^{\top}$ (4×5, columns = token embeddings):

```
Eᵀ[:,0] = [0.10, 0.50, -0.90, 0.40]   ← embedding of token 0
Eᵀ[:,1] = [-0.20, 0.60, 0.10, -0.50]  ← embedding of token 1
Eᵀ[:,2] = [0.30, -0.70, 0.20, 0.60]   ← embedding of token 2
Eᵀ[:,3] = [-0.40, 0.80, -0.30, -0.70] ← embedding of token 3
Eᵀ[:,4] = [0.50, -0.40, 0.20, 0.30]   ← embedding of token 4 (hypothetical)
```

Wait — Eᵀ has shape $[d \times |V|]$. Projecting $h [1\times d]$ → $[1\times |V|]$:

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

## 8.6 The Code: Full microGPT Forward Pass in Scheme

Configuration is data. We represent the model's five hyperparameters as a plain list and define selectors once. Nothing in the forward pass hardcodes any dimension — every procedure reads what it needs through these selectors. 

```scheme
(define (make-gpt-config vocab-size d-model num-heads num-layers max-seq-len)
  (list vocab-size d-model num-heads num-layers max-seq-len))

(define (config-vocab-size   c) (car c))
(define (config-d-model      c) (cadr c))
(define (config-num-heads    c) (caddr c))
(define (config-num-layers   c) (cadddr c))
(define (config-max-seq-len  c) (car (cddddr c)))
```

`make-gpt-params` allocates all learnable weights. `wte` is the token embedding matrix $[|V| \times d]$; `wpe` is the position embedding matrix $[\text{max\_seq\_len} \times d]$ — both are random matrices initialized exactly as in Chapter 2. `blocks` is a list of $N$ transformer block parameter lists from Chapter 7. The final layer norm `ln-f` is a pair $(\gamma, \beta)$. Weight tying means the unembedding step will reuse `wte` transposed — no separate parameter is stored.

```scheme
(define (make-gpt-params config)
  (let* ((V (config-vocab-size  config))
         (d (config-d-model     config))
         (H (config-num-heads   config))
         (N (config-num-layers  config))
         (T (config-max-seq-len config))
         (wte    (make-random-matrix V d))
         (wpe    (make-random-matrix T d))
         (blocks (let build ((n N) (acc '()))
                   (if (= n 0) acc
                       (build (- n 1) (cons (make-block-params d H) acc)))))
         (ln-f   (make-ln-params d)))
    (list wte wpe blocks ln-f)))

(define (params-wte    p) (car p))
(define (params-wpe    p) (cadr p))
(define (params-blocks p) (caddr p))
(define (params-ln-f   p) (cadddr p))
```

`gpt-forward` is the full pipeline: embed token IDs, add position embeddings, pass through $N$ transformer blocks, apply the final layer norm to the last token's vector, then compute dot products with every row of `wte` to produce the logit for each vocabulary entry. Each step is a single call to a procedure defined in an earlier chapter. The composition of those calls is the complete model.

```scheme
(define (gpt-forward token-ids params config)
  (let* ((T-len   (length token-ids))
         (d       (config-d-model config))
         (wte     (params-wte    params))
         (wpe     (params-wpe    params))
         (blocks  (params-blocks params))
         (ln-f    (params-ln-f   params))
         (tok-emb (embed-sequence wte token-ids))
         (pos-ids (let build ((i 0) (acc '()))
                    (if (= i T-len) (reverse acc)
                        (build (+ i 1) (cons i acc)))))
         (pos-emb (embed-sequence wpe pos-ids))
         (X       (matrix-add tok-emb pos-emb))
         (X-final (forward-stack X blocks))
         (last-v  (let ((v (make-vector d)))
                    (let loop ((j 0))
                      (when (< j d)
                        (vector-set! v j (matrix-ref X-final (- T-len 1) j))
                        (loop (+ j 1))))
                    v))
         (h       (layer-norm-vec last-v (ln-gamma ln-f) (ln-beta ln-f)))
         (V-size  (config-vocab-size config))
         (logits  (make-vector V-size 0.0)))
    (let loop ((i 0))
      (when (< i V-size)
        (let dot ((j 0) (acc 0.0))
          (if (= j d)
              (vector-set! logits i acc)
              (dot (+ j 1) (+ acc (* (vector-ref h j) (matrix-ref wte i j))))))
        (loop (+ i 1))))
    logits))
```

Three sampling utilities sit between the logit vector and the next token. `softmax-1d` converts logits to probabilities with the usual max-subtraction trick. `top-k-logits` zeroes out all but the $k$ highest-scoring entries — restricting the candidate set prevents the model from putting probability on obviously wrong tokens. `sample-token` draws one token index from the distribution using inverse CDF sampling. Temperature divides the logits before softmax: values below 1 sharpen the distribution (more greedy), values above 1 flatten it (more random).

```scheme
(define (softmax-1d v)
  (let* ((n     (vector-length v))
         (max-v (let loop ((i 1) (m (vector-ref v 0)))
                  (if (= i n) m
                      (loop (+ i 1) (max m (vector-ref v i))))))
         (exps  (make-vector n))
         (total (let loop ((i 0) (s 0.0))
                  (if (= i n) s
                      (let ((e (exp (- (vector-ref v i) max-v))))
                        (vector-set! exps i e)
                        (loop (+ i 1) (+ s e)))))))
    (let loop ((i 0))
      (when (< i n)
        (vector-set! exps i (/ (vector-ref exps i) total))
        (loop (+ i 1))))
    exps))

(define (top-k-logits logits k)
  (let* ((n      (vector-length logits))
         (pairs  (let build ((i 0) (acc '()))
                   (if (= i n) acc
                       (build (+ i 1) (cons (cons i (vector-ref logits i)) acc)))))
         (sorted (sort pairs (lambda (a b) (> (cdr a) (cdr b)))))
         (result (make-vector n -1.0e9)))
    (let loop ((ps sorted) (count 0))
      (when (and (not (null? ps)) (< count k))
        (vector-set! result (caar ps) (cdar ps))
        (loop (cdr ps) (+ count 1))))
    result))

(define (sample-token probs)
  (let ((r (random 1.0)))
    (let loop ((i 0) (cum 0.0))
      (if (= i (vector-length probs))
          (- (vector-length probs) 1)
          (let ((c (+ cum (vector-ref probs i))))
            (if (>= c r) i
                (loop (+ i 1) c)))))))
```

`generate` is the autoregressive loop. Starting from a list of prompt token IDs, it calls `gpt-forward`, applies temperature and top-k, samples the next token, appends it to the sequence, and repeats. The model sees its own output growing with each step — the entire context window is fed back in every iteration. This loop is the inference procedure; training is different (Chapter 9 and 10).

```scheme
(define (generate params config start-ids max-new-tokens temperature top-k)
  (let loop ((tokens start-ids) (n max-new-tokens) (generated '()))
    (if (= n 0)
        (reverse generated)
        (let* ((logits   (gpt-forward tokens params config))
               (scaled   (let* ((len (vector-length logits))
                                (v   (make-vector len)))
                           (let lp ((i 0))
                             (when (< i len)
                               (vector-set! v i (/ (vector-ref logits i) temperature))
                               (lp (+ i 1))))
                           v))
               (filtered (top-k-logits scaled top-k))
               (probs    (softmax-1d filtered))
               (next     (sample-token probs)))
          (loop (append tokens (list next)) (- n 1) (cons next generated))))))
```

**Demo.** The parameter count formula gives us the model's total size before running it. We then run a short generation to confirm the pipeline executes end-to-end.

```scheme
(define (count-parameters config)
  (let* ((V (config-vocab-size  config))
         (d (config-d-model     config))
         (N (config-num-layers  config))
         (T (config-max-seq-len config)))
    (+ (* V d)
       (* T d)
       (* N (+ (* 4 d d)
               (* 8 d d)
               (* 4 d)))
       (* 2 d))))

(let* ((config    (make-gpt-config 50 16 4 2 32))
       (params    (make-gpt-params config))
       (prompt    '(3 7 12 5))
       (generated (generate params config prompt 5 0.8 5)))
  (display "microGPT parameters: ")
  (display (count-parameters config)) (newline)
  (display "Prompt tokens:    ") (display prompt) (newline)
  (display "Generated tokens: ") (display generated) (newline)
  (display "Full sequence:    ")
  (display (append prompt generated)) (newline))
```

Run with `mit-scheme --quiet --load microgpt.scm`.

---

## 8.7 Weight Tying in Detail

Why does weight tying work so well? Consider:

- The embedding `E[i]` is learned to represent token `i` such that tokens that appear in similar contexts have similar embeddings.
- The unembedding $logits[i] = h \cdot E[i]$ measures the alignment between the model's prediction vector `h` and token `i`'s embedding.
- If `h` is pointing in the direction of "tokens that follow this context," and `E[i]` represents token `i`'s meaning, then tokens that are semantically appropriate will naturally score higher.

Weight tying forces the model to use a single geometric space for both "what a token means" and "how to predict a token" — an elegant constraint that regularizes learning.

---

## 8.8 Key Takeaways

- The **unembedding** projects the final hidden state to logits: $logits = h W_u \in \mathbb{R}^{|V|}$.
- **Weight tying** ($W_u = E^{\top}$) reuses the embedding matrix and reduces parameters.
- **Softmax** converts logits to a probability distribution.
- **Temperature** controls sampling sharpness; **top-K** restricts the candidate set.
- **Autoregressive generation**: sample one token at a time, feeding each back as input.
- **Training loss** is cross-entropy: see [Chapter 9](./ch09-loss.md) for full treatment.

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


---

![Vocab projection — the unembedding matrix converts hidden states to logits over 50k tokens.](images/ch08_projection.png)

*Vocab projection — the unembedding matrix converts hidden states to logits over 50k tokens.*

![Autoregressive generation loop — each predicted token is appended and fed back into the model.](images/ch08_generation.png)

*Autoregressive generation loop — each predicted token is appended and fed back into the model.*

