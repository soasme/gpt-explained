# Chapter 10: Training — Teaching the Model

> *"Backpropagation is just the chain rule, applied patiently from output to input."*

Chapter 9 gave us a loss: a single number measuring how wrong the model is. But knowing *how wrong* we are is only half the battle. We also need to know *which weights* are responsible for the error, and *by how much to change each one*.

That is the job of **training**: adjusting every weight in the model so that the next forward pass makes a better prediction. This chapter builds the machinery piece by piece — starting from the simplest idea (nudging one number) and working up to a complete backward pass through the transformer.

---

## 10.1 The Goal: Move the Loss Down

The model has millions of parameters. Call them collectively $\theta$. The loss $\mathcal{L}(\theta)$ is a function of all of them. We want to find $\theta$ that makes $\mathcal{L}$ small.

The strategy is **gradient descent**: repeatedly take a small step downhill on the loss surface.

$$
\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}
$$

$\eta > 0$ is the **learning rate** — how big a step to take. $\nabla_\theta \mathcal{L}$ is the **gradient**: a collection of partial derivatives, one per parameter, each saying *"if you increase this weight, does the loss go up or down, and by how much?"*

> **Math Minute — Partial Derivatives**
>
> A partial derivative $\frac{\partial \mathcal{L}}{\partial w}$ answers: *if we increase $w$ by a tiny $\epsilon$ while holding every other weight fixed, by how much does $\mathcal{L}$ change?*
>
> A large positive value means $w$ is pushing loss up — decrease it. A negative value means the opposite. Zero means $w$ is momentarily irrelevant to the loss.
>
> Notation: $\partial$ (curly d) instead of $d$ signals "partial" — other variables are held constant.

The question is: how do we compute every $\frac{\partial \mathcal{L}}{\partial w}$ efficiently, given that the loss depends on weights through hundreds of nested operations?

The answer is **backpropagation**.

---

## 10.2 The Chain Rule

Backpropagation is the chain rule from calculus, applied to a computation graph.

Suppose the loss depends on weight $w$ through a chain of intermediate values:

$$
w \xrightarrow{f} z \xrightarrow{g} \mathcal{L}
$$

The chain rule says:

$$
\frac{d\mathcal{L}}{dw} = \frac{d\mathcal{L}}{dz} \cdot \frac{dz}{dw}
$$

**If we know how $\mathcal{L}$ varies with $z$, and how $z$ varies with $w$, we multiply to get how $\mathcal{L}$ varies with $w$.**

In a transformer, the chain has hundreds of layers. Backprop starts at the loss and works backward, multiplying local derivatives as it goes. Each layer only needs two things: the gradient arriving from the layer above (called $\delta$, "delta"), and its own local derivative.

Let's build this machinery step by step.

### Step 1: A gradient update for one weight

The simplest gradient descent imaginable: one weight, one step.

```lisp
;;; The most basic gradient descent: one weight, one step downhill.
(defun sgd-update-scalar (weight gradient learning-rate)
  "Move WEIGHT one step downhill along GRADIENT."
  (- weight (* learning-rate gradient)))
```

If the gradient is positive (loss rises when $w$ increases), we subtract — decreasing $w$. If negative, we add. The learning rate $\eta$ scales how large the step is.

This is the entire idea. Everything that follows is this rule, applied to millions of weights simultaneously.

---

## 10.3 Backprop Through a Linear Layer

The most common operation in a transformer is a **linear layer**: $y = Wx$.

Given the upstream gradient $\delta = \frac{\partial \mathcal{L}}{\partial y}$ (arriving from the layer above), the chain rule gives:

$$
\frac{\partial \mathcal{L}}{\partial W} = \delta \cdot x^\top
\qquad
\frac{\partial \mathcal{L}}{\partial x} = W^\top \cdot \delta
$$

The first equation tells us how to update $W$. The second passes the gradient backward to whatever fed $x$ into this layer.

> **Concrete example.** Suppose $W$ is $2 \times 3$, $x = [1, 2, 3]^\top$, and the upstream gradient is $\delta = [0.5, {-0.3}]^\top$. Then:
>
> $$\frac{\partial \mathcal{L}}{\partial W} = \begin{bmatrix} 0.5 \\ -0.3 \end{bmatrix} \begin{bmatrix} 1 & 2 & 3 \end{bmatrix} = \begin{bmatrix} 0.50 & 1.00 & 1.50 \\ -0.30 & -0.60 & -0.90 \end{bmatrix}$$
>
> The gradient w.r.t. $W$ has the same shape as $W$ — one entry per weight, each saying how much the loss would change if that weight increased slightly.

### Step 2: Linear layer backward pass

```lisp
;;; Backward pass through y = W x.
;;; DELTA: upstream gradient, shape [out-dim]
;;; W:     weight matrix used in the forward pass, shape [out-dim x in-dim]
;;; X:     input used in the forward pass, shape [in-dim]
;;; Returns: grad-W (same shape as W) and grad-x (same shape as x).
(defun linear-backward (delta W x)
  (let* ((out-dim (array-dimension W 0))
         (in-dim  (array-dimension W 1))
         (grad-W  (make-array (list out-dim in-dim)
                               :element-type 'single-float
                               :initial-element 0.0))
         (grad-x  (make-array in-dim
                               :element-type 'single-float
                               :initial-element 0.0)))
    ;; grad-W[i,j] = delta[i] * x[j]   (outer product)
    (dotimes (i out-dim)
      (dotimes (j in-dim)
        (setf (aref grad-W i j)
              (* (aref delta i) (aref x j)))))
    ;; grad-x[j] = sum_i  W[i,j] * delta[i]   (W^T delta)
    (dotimes (j in-dim)
      (dotimes (i out-dim)
        (incf (aref grad-x j)
              (* (aref W i j) (aref delta i)))))
    (values grad-W grad-x)))
```

`linear-backward` mirrors `linear-forward` in shape: same sizes flow through, but in the reverse direction.

---

## 10.4 Backprop Through Softmax and Cross-Entropy

Chapter 9 stated the softmax-CE gradient without proof:

$$
\frac{\partial \mathcal{L}}{\partial z_j} = P(j) - \mathbb{1}[j = t]
$$

where $z_j$ are the logits and $t$ is the true token. Let us derive it now.

The forward pass chains two operations:

1. **Softmax:** $P_j = \dfrac{e^{z_j}}{\sum_k e^{z_k}}$
2. **Loss:** $\mathcal{L} = -\log P_t$

Applying the chain rule through softmax to the loss — accounting for the fact that every logit $z_j$ influences every probability $P_k$ — the algebra simplifies beautifully to:

$$
\frac{\partial \mathcal{L}}{\partial z_j} = P_j - \mathbb{1}[j = t]
$$

For the true token $j = t$: gradient is $P_t - 1$ (negative, so we push $z_t$ up). For every other token: gradient is $P_j$ (positive, so we push those logits down). The model is nudged to be more confident about the correct answer.

### Step 3: Softmax-cross-entropy backward

```lisp
;;; Gradient of cross-entropy loss w.r.t. logits z.
;;; LOGITS: raw scores before softmax, shape [vocab-size]
;;; TRUE-ID: index of the correct next token
;;; Returns: gradient vector, shape [vocab-size]
(defun softmax-ce-backward (logits true-id)
  (let* ((probs (softmax logits))
         (grad  (copy-seq probs)))       ; start: P(j) for every j
    (decf (aref grad true-id) 1.0)       ; subtract 1 at the true token
    grad))
```

Two lines of logic. The entire gradient of the output layer is: predicted probabilities, with $1$ subtracted from the correct class.

---

## 10.5 Accumulating Gradients Across the Sequence

We compute the softmax-CE gradient independently at each position $t$. But all positions share the same unembedding matrix $W_u$ (and the same transformer weights). Their gradient contributions must be *summed* before updating any weight.

This is how shared parameters learn from the whole sequence at once.

### Step 4: Summing gradient contributions

```lisp
;;; Add together per-position gradient matrices for a shared weight.
;;; GRAD-LIST: list of gradient matrices (one per sequence position),
;;;            all the same shape.
;;; Returns: element-wise sum, same shape.
(defun accumulate-gradients (grad-list)
  (let* ((rows  (array-dimension (first grad-list) 0))
         (cols  (array-dimension (first grad-list) 1))
         (total (make-array (list rows cols)
                             :element-type 'single-float
                             :initial-element 0.0)))
    (dolist (g grad-list total)
      (dotimes (i rows)
        (dotimes (j cols)
          (incf (aref total i j) (aref g i j)))))))
```

If a weight was used at positions 0, 1, and 2, the gradient entering the weight update is the sum of all three positions' contributions. The more a weight influenced the output at many positions, the larger (and more reliable) its accumulated gradient.

---

## 10.6 The Gradient Descent Step

With gradients accumulated for every parameter, the update rule is:

$$
W \leftarrow W - \eta \cdot \frac{\partial \mathcal{L}}{\partial W}
$$

### Step 5: Updating a weight matrix in place

```lisp
;;; Gradient descent update for a 2-D weight matrix (in place).
;;; W:    weight matrix to update
;;; GRAD: gradient matrix, same shape as W
;;; LR:   learning rate η
(defun sgd-update! (W grad lr)
  (let ((rows (array-dimension W 0))
        (cols (array-dimension W 1)))
    (dotimes (i rows)
      (dotimes (j cols)
        (decf (aref W i j)
              (* lr (aref grad i j)))))))
```

The `!` suffix signals mutation — we modify the weight matrix in place, as is conventional for numerical update routines. Every entry $W_{ij}$ moves a small step in the direction that reduces the loss.

---

## 10.7 One Training Step

Now we can assemble the full cycle: forward pass to get predictions and loss, backward pass to get gradients, update pass to improve every weight.

### Step 6: The training step

```lisp
;;; One complete forward + backward + update cycle.
;;; MODEL:  struct holding all weight matrices
;;; TOKENS: input token IDs (list of integers)
;;; LABELS: true next-token IDs (list of integers, same length)
;;; LR:     learning rate
;;; Returns: scalar loss for this step.
(defun training-step (model tokens labels lr)
  ;; 1. Forward pass — collect logit vectors at every position
  (let* ((logits-seq (forward-pass model tokens))

         ;; 2. Loss — mean cross-entropy over the sequence
         (loss (sequence-loss logits-seq labels))

         ;; 3. Backward pass — each layer receives the upstream delta,
         ;;    returns its own grad-W and the downstream delta.
         ;;    Layers are visited in reverse order.
         (grads (backward-pass model tokens logits-seq labels)))

    ;; 4. Update every parameter with its gradient
    (dolist (param-grad grads)
      (destructuring-bind (W grad) param-grad
        (sgd-update! W grad lr)))

    ;; Return the loss so callers can plot training progress
    loss))
```

`forward-pass` and `backward-pass` are themselves composed of the smaller pieces we built above: `softmax-ce-backward` → `linear-backward` → ... chained from output to input. `training-step` is the loop that ties them together.

---

## 10.8 Watching the Loss Fall

Running thousands of training steps, the loss curve looks roughly like this:

| Step | Loss | Perplexity | Interpretation |
|------|------|------------|----------------|
| 0 | 10.82 | 50,000 | Random — model knows nothing |
| 100 | 6.91 | 1,000 | Ruling out most tokens |
| 1,000 | 4.61 | 100 | Learned basic frequency |
| 10,000 | 2.30 | 10 | Has rough grammar |
| 100,000 | 0.92 | 2.5 | Strong language model |

The curve drops steeply at first (obvious mistakes are easy to fix) then more slowly and noisily (subtle patterns are harder and data points disagree).

> **Why does loss get noisy later?**
>
> Early on, large gradients correct glaring errors — the model hasn't even learned common word frequencies. Later, gradients are smaller and point in slightly different directions for different training examples. The noise is not a bug: random variation in gradient direction helps the model escape poor local minima and find flatter, better-generalizing regions of the loss surface. This is the stochastic in **stochastic gradient descent** (SGD).

---

## 10.9 Diagram

![Backpropagation: gradient flowing from loss backward through each layer, updating weights](images/ch10_training.png)

---

## 10.10 Key Takeaways

- **Gradient descent** moves every weight downhill: $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$.
- **Partial derivatives** measure the sensitivity of the loss to each individual weight, holding all others fixed.
- **Backpropagation** is the chain rule applied layer by layer from loss to inputs. Each layer only needs the upstream gradient $\delta$ and its own local derivative — no global information required.
- **Linear layer backward:** $\frac{\partial \mathcal{L}}{\partial W} = \delta x^\top$, $\frac{\partial \mathcal{L}}{\partial x} = W^\top \delta$. The same two matrices as the forward pass, just transposed and multiplied in a different order.
- **Softmax-CE gradient** is $P(j) - \mathbb{1}[j = t]$: the model's predicted probability minus the true label. Nearly zero when correct and confident; large when wrong.
- **Shared weights accumulate gradients** from every position in the sequence before any update is applied.
- The six-step cycle — forward, loss, backward, accumulate, update, repeat — is all of training. Every weight in the model, over billions of tokens, updated by exactly this loop.
