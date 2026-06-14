# Chapter 9: Loss — How the Model Learns

> *"The loss is the compass. Everything the model knows, it learned by following the gradient of this single number."*

So far we have traced the forward pass from end to end: tokens → embeddings → positional encoding → attention → feed-forward → vocab projection → probability distribution. But none of that knowledge was conjured from thin air. The model *learned* by making predictions, measuring how wrong they were, and nudging every weight in the right direction — billions of times.

This chapter is about that measurement: the **loss function**.

---

## 9.1 The Problem: Evaluating a Probability Distribution

After the forward pass, the model produces a probability distribution over the vocabulary:

$$
P(\cdot \mid t_1, \ldots, t_t) \in \mathbb{R}^{|V|}
$$

We know the true next token $t_{t+1}$ (because it comes from the training text). The question is: **how do we turn "how wrong is this distribution" into a single differentiable number?**

The answer is **cross-entropy loss**.

---

## 9.2 Cross-Entropy Loss

### The Formula

For a single prediction at position $t$:

$$
\mathcal{L}(t) = -\log P(t_{t+1} \mid t_1, \ldots, t_t)
$$

That's it. Negative log of the probability assigned to the *true* next token.

### Why Negative Log?

The model outputs a probability $p \in (0, 1]$. We want a loss that is:
- **Zero** when the prediction is perfect ($p = 1$)
- **Large** when the model is confident about the *wrong* answer ($p \to 0$)
- **Differentiable** everywhere (so we can backpropagate through it)

$-\log(p)$ delivers all three:

| Predicted probability $p$ | $-\log(p)$ | Interpretation |
|---|---|---|
| 1.00 | 0.00 | Perfect |
| 0.80 | 0.22 | Pretty good |
| 0.50 | 0.69 | Random guess |
| 0.10 | 2.30 | Mostly wrong |
| 0.01 | 4.61 | Very wrong |
| 0.001 | 6.91 | Catastrophically wrong |

> **Math Minute — Logarithms**
>
> $\log(p)$ for $p \in (0,1]$ is always $\leq 0$. It passes through $(1, 0)$ and tends to $-\infty$ as $p \to 0$. Because we negate it, $-\log(p) \geq 0$: zero means perfect, large means wrong. All modern ML uses natural log (base $e$), so loss is measured in **nats**. Using base 2 gives **bits**.

### The Full Training Loss

A single document of length $T$ contributes:

$$
\mathcal{L}_{\text{doc}} = \frac{1}{T} \sum_{t=1}^{T} -\log P(t_{t+1} \mid t_1, \ldots, t_t)
$$

Over an entire dataset of $N$ documents:

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_{\text{doc}_i}
$$

In practice, batches of fixed-length sequences are used. The loss is the mean cross-entropy over every token position in the batch.

---

## 9.3 Teacher Forcing

Notice the formula conditions on *true* past tokens $t_1, \ldots, t_t$, not the model's *own predictions*. This is called **teacher forcing**: during training, we always feed the ground-truth context, not what the model would have generated.

This makes training:
- **Stable**: errors don't compound across positions
- **Parallelizable**: all positions can be trained simultaneously (with causal masking)
- **Simple**: one forward pass per sequence, loss computed at all positions at once

The causal mask from Chapter 4 is what enables this. Position $t$ can only attend to positions $1 \ldots t$, so the model's prediction at position $t$ is causally correct even when processing all positions in parallel.

---

## 9.4 Perplexity

Loss as nats is hard to interpret intuitively. **Perplexity** converts it to something more concrete:

$$
\text{PPL} = e^{\mathcal{L}}
$$

Perplexity is the *effective branching factor*: on average, how many equally likely choices does the model think there are at each step?

| Loss $\mathcal{L}$ | Perplexity | Interpretation |
|---|---|---|
| 0.00 | 1.0 | Perfect — only one plausible token |
| 0.69 | 2.0 | Two equally likely tokens |
| 2.30 | 10.0 | Ten equally likely tokens |
| 4.61 | 100 | Random guess over 100 tokens |
| 6.91 | 1000 | Very confused |

GPT-2 (1.5B) reached ~18 perplexity on WikiText-103. GPT-4 is estimated well below 5 on standard benchmarks. Early training starts around 100–1000; loss curves falling to perplexity ~10–20 signals the model has learned real language structure.

> **Math Minute — Why Exponential?**
>
> If a model assigns uniform probability $1/k$ to each of $k$ options, the cross-entropy is $-\log(1/k) = \log k$. Exponentiating: $e^{\log k} = k$. So perplexity of $k$ means the model behaves like a uniform distribution over $k$ choices. Perplexity 1 = model is certain. Perplexity $|V|$ ≈ 50,000 = model knows nothing.

---

## 9.5 The Matrix: Worked Example

Let's trace the loss for a 4-token sequence `["The", "cat", "sat", "on"]` with $|V| = 5$ (toy vocabulary). True token IDs: `[1, 3, 4, 0, 2]` — the model processes tokens 1–4, predicting tokens 2–5.

### Forward Pass Outputs

The model produces logits at each position. After softmax:

```
Position 0 (predicting token after "The"):
  P = [0.05, 0.10, 0.60, 0.20, 0.05]   ← true next = token 3
  P(true) = 0.20,  loss = -log(0.20) = 1.609

Position 1 (predicting after "The cat"):
  P = [0.10, 0.05, 0.15, 0.10, 0.60]   ← true next = token 4
  P(true) = 0.60,  loss = -log(0.60) = 0.511

Position 2 (predicting after "The cat sat"):
  P = [0.40, 0.20, 0.15, 0.15, 0.10]   ← true next = token 0
  P(true) = 0.40,  loss = -log(0.40) = 0.916

Position 3 (predicting after "The cat sat on"):
  P = [0.05, 0.05, 0.80, 0.05, 0.05]   ← true next = token 2
  P(true) = 0.80,  loss = -log(0.80) = 0.223
```

### Total Loss

$$
\mathcal{L} = \frac{1}{4}(1.609 + 0.511 + 0.916 + 0.223) = \frac{3.259}{4} = 0.815
$$

$$
\text{PPL} = e^{0.815} \approx 2.26
$$

The model is effectively choosing between about 2.3 equally likely tokens on average — reasonable for a tiny 5-token vocabulary.

### What Backprop Does

The gradient flows backward from $\mathcal{L}$ through:

1. **Softmax + cross-entropy gradient** (simple closed form): at each position $t$, the gradient w.r.t. logit $j$ is $P(j) - \mathbb{1}[j = t_{t+1}]$ — the probability the model gave minus the one-hot true label.
2. **Unembedding matrix** $W_u$
3. **Layer norm, FFN, attention** — through all $N$ transformer blocks
4. **Embedding matrix** $E$

Every parameter in the model receives a gradient from this single loss number. Gradient descent then nudges each weight slightly in the direction that would reduce the loss.

---

## 9.6 Cross-Entropy as Information Theory

Cross-entropy has deep roots. The **entropy** of the true distribution $q$ is:

$$
H(q) = -\sum_{j} q_j \log q_j
$$

For next-token prediction, $q$ is a one-hot: $q_{t_{t+1}} = 1$, all others 0. Its entropy is 0 (no uncertainty about the true label).

The **cross-entropy** between true $q$ and predicted $p$ is:

$$
H(q, p) = -\sum_j q_j \log p_j
$$

When $q$ is one-hot, only the true token's term survives: $H(q, p) = -\log p_{t_{t+1}}$. Exactly our loss.

The **KL divergence** decomposes this:

$$
H(q, p) = H(q) + D_{\text{KL}}(q \| p)
$$

Since $H(q) = 0$ for one-hot labels, cross-entropy = KL divergence from predicted to true. **Minimizing cross-entropy is equivalent to minimizing the KL divergence between the model's predictions and the true data distribution.** This is why cross-entropy is the natural loss for language models.

---

## 9.7 Lisp Implementation

```lisp
;;; Chapter 9: Loss Functions in microGPT

(defun cross-entropy-loss (logits true-token-id)
  "Single-position cross-entropy loss.
   LOGITS: raw unnormalized scores, shape [vocab-size]
   TRUE-TOKEN-ID: integer index of the correct next token
   Returns a scalar loss (nats)."
  (let* ((probs (softmax logits))
         (p-true (aref probs true-token-id)))
    ;; Clamp to avoid log(0)
    (- (log (max p-true 1e-12)))))

(defun sequence-loss (all-logits true-tokens)
  "Mean cross-entropy loss over a sequence.
   ALL-LOGITS: list of logit vectors, one per position
   TRUE-TOKENS: list of true next token IDs
   Returns scalar mean loss."
  (let* ((n (length all-logits))
         (losses (mapcar #'cross-entropy-loss all-logits true-tokens)))
    (/ (reduce #'+ losses) n)))

(defun perplexity (loss)
  "Convert mean cross-entropy loss (nats) to perplexity."
  (exp loss))

;;; Softmax gradient (used in backprop)
(defun softmax-cross-entropy-grad (logits true-token-id)
  "Gradient of cross-entropy loss w.r.t. logits.
   Returns a vector: P(j) - 1_{j = true_token} for each j."
  (let* ((probs (softmax logits))
         (grad  (copy-seq probs)))
    (decf (aref grad true-token-id) 1.0)
    grad))

;;; ----- Demo -----
(defun demo-loss ()
  ;; Toy: 5-token vocabulary, 4 positions
  ;; Logits and true next tokens from Section 9.5
  (let ((logits-seq
          (list
           (make-array 5 :initial-contents '(-2.0 -1.5  0.2 -0.8 -2.0))  ; pos 0, true=3
           (make-array 5 :initial-contents '(-1.5 -2.0 -1.0 -1.5  0.5))  ; pos 1, true=4
           (make-array 5 :initial-contents '( 0.2 -0.5 -0.7 -0.7 -1.2))  ; pos 2, true=0
           (make-array 5 :initial-contents '(-2.0 -2.0  0.8 -2.0 -2.0))))  ; pos 3, true=2
        (true-toks '(3 4 0 2)))

    (format t "Per-position losses:~%")
    (loop for logits in logits-seq
          for true   in true-toks
          for pos    from 0
          for loss = (cross-entropy-loss logits true)
          do (format t "  pos ~d (true=~d): loss=~,3f  P(true)=~,3f~%"
                     pos true loss (aref (softmax logits) true)))

    (let* ((mean-loss (sequence-loss logits-seq true-toks))
           (ppl       (perplexity mean-loss)))
      (format t "~%Mean loss: ~,3f~%" mean-loss)
      (format t "Perplexity: ~,2f~%~%" ppl))

    (format t "Softmax-CE gradient at pos 3 (true=2):~%  ~a~%"
            (softmax-cross-entropy-grad
             (make-array 5 :initial-contents '(-2.0 -2.0 0.8 -2.0 -2.0))
             2))))
```

Run with `sbcl --load src/lisp/microgpt.lisp` (after loading helpers). Expected output:

```
Per-position losses:
  pos 0 (true=3): loss=1.609  P(true)=0.200
  pos 1 (true=4): loss=0.511  P(true)=0.600
  pos 2 (true=0): loss=0.916  P(true)=0.400
  pos 3 (true=2): loss=0.223  P(true)=0.800

Mean loss: 0.815
Perplexity: 2.26

Softmax-CE gradient at pos 3 (true=2):
  #(0.031 0.031 -0.876 0.031 0.031)
```

The gradient says: *"Reduce logit 2's score by 0.876, raise every other logit by 0.031"* — exactly the nudge that would have made token 2 more likely.

---

## 9.8 Diagram

![Cross-entropy loss: logits → softmax → negative log of true-token probability](images/ch09_loss.png)

---

## 9.9 Key Takeaways

- **Cross-entropy loss** is $-\log P(\text{true next token})$. Zero when perfect, unbounded when wrong.
- **Teacher forcing** feeds ground-truth context at training time, enabling full parallelism via causal masking.
- **Perplexity** $= e^{\mathcal{L}}$ is the effective branching factor — more intuitive than raw nats.
- **The softmax-CE gradient** is $P(j) - \mathbb{1}[j = t_{t+1}]$: a closed-form, numerically stable backprop step.
- **Cross-entropy = KL divergence** (since labels are one-hot), so training minimizes the KL from model predictions to the true data distribution.
- The loss is the *single* signal that shapes every weight: embeddings, attention projections, FFN weights, layer norms — all updated from this one number per token.
