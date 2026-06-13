# Chapter 1: Tokens — Text to Numbers

> *"Before a model can think about words, it must forget that words exist."*

A language model does not read text. It reads **integers**. The first job of any GPT pipeline is to convert a string of characters into a sequence of integers. That conversion is called **tokenization**, and the integers it produces are called **token IDs**.

This chapter explains what tokens are, how they are produced, and what they look like as data structures. We will implement a tiny byte-pair encoding tokenizer in Lisp.

---

## 1.1 The Idea

Consider the sentence:

```
the cat sat on the mat
```

A human reads six words. A GPT model reads something like:

```
[1the] [2cat] [3sat] [4on] [1the] [5mat]
```

Where each bracketed item is a token and the number is its ID in a vocabulary table. The model never sees the string `"the"` — it only ever sees the integer `1`.

**Why not just use characters?** You could map each letter to an integer: `a→1, b→2, …`. The problem is that the model would need to learn that `c`, `a`, `t` together mean something different from `c`, `a`, `r`. It would have to rediscover every word from scratch. This makes learning extremely sample-inefficient.

**Why not use whole words?** The opposite extreme. Map every word to an integer. This works for common words, but English has hundreds of thousands of words, and new ones appear constantly. The vocabulary would be enormous, and rare words would have almost no training data.

**Subword tokenization** finds the middle ground. Common whole words become single tokens. Rare words get split into recognizable pieces. The word `"tokenization"` might become `["token", "ization"]`. The word `"running"` might become `["run", "ning"]`.

> **Math Minute — Sets and Mappings**
> A vocabulary `V` is a finite set of strings. A tokenizer is a function `T: String → List[ℤ]` mapping a string to a sequence of integers from `{0, 1, …, |V|−1}`. Each integer is an index into `V`.

---

## 1.2 Byte Pair Encoding (BPE)

The most common tokenization algorithm used in GPT models is **Byte Pair Encoding**. The algorithm has two phases:

**Training phase** (happens once, offline):

1. Start with a vocabulary of individual characters (or bytes).
2. Count every adjacent pair of tokens in the training corpus.
3. Merge the most frequent pair into a single new token.
4. Repeat until the vocabulary reaches the target size (e.g., 50,000).

**Encoding phase** (happens at inference time):

1. Start with the input string split into individual characters.
2. Greedily merge adjacent pairs according to the trained merge rules, highest-priority first.
3. Return the final token IDs.

### The Math

Let `V₀ = {c₁, c₂, …}` be the initial character vocabulary. After each merge step `k`:

```
V_{k+1} = V_k ∪ {(a, b) | freq(a, b) is max over all adjacent pairs}
```

The merge order defines a priority list `M = [(a₁,b₁), (a₂,b₂), …]`. To encode a string:

```
tokens = characters(string)
for (a, b) in M:
    tokens = apply_merge(tokens, a, b)
return [vocab_id(t) for t in tokens]
```

Where `apply_merge` replaces every adjacent occurrence of `(a, b)` with the merged token `ab`.

---

## 1.3 The Matrix: A Tiny Vocabulary Table

Let's trace through a micro-example. Suppose our entire training corpus is:

```
low lower newest
```

**Step 0 — Character vocabulary:**

| ID | Token |
|----|-------|
| 0  | `l`   |
| 1  | `o`   |
| 2  | `w`   |
| 3  | `e`   |
| 4  | `r`   |
| 5  | `n`   |
| 6  | `s`   |
| 7  | `t`   |
| 8  | `_`   |  ← end-of-word marker

Initial tokenized corpus: `[l,o,w,_] [l,o,w,e,r,_] [n,e,w,e,s,t,_]`

**Step 1 — Count pairs:**

```
(l,o): 2    ← most frequent!
(o,w): 2
(w,_): 1
(w,e): 1
...
```

Merge `(l,o)` → `lo`. Vocabulary now has 9 entries: `lo` gets ID 9.

**Step 2 — Corpus becomes:**

`[lo,w,_] [lo,w,e,r,_] [n,e,w,e,s,t,_]`

Most frequent pair now: `(lo,w)` with count 2. Merge → `low` (ID 10).

After a few more merges, encoding `"low"` returns `[10]` — a single token.

![Tokenization diagram showing BPE merges](images/ch01-bpe-merges.png)

---

## 1.4 The Code: A Micro-BPE in Lisp

```lisp
;;;; micro-bpe.lisp
;;;; A minimal implementation of Byte Pair Encoding
;;;; Deps: none (pure Common Lisp)

;;; ─── Data Structures ───────────────────────────────────────────

;; A VOCAB is an alist: (string . integer)
;; Example: (("l" . 0) ("o" . 1) ("lo" . 8) ...)

;; A CORPUS is a list of token-sequences, where each token is a string.
;; Example: (("l" "o" "w" "_") ("l" "o" "w" "e" "r" "_"))

;;; ─── Step 1: Build character vocabulary ────────────────────────

(defun string->chars (s)
  "Split string S into a list of single-character strings."
  (loop for c across s collect (string c)))

(defun initial-corpus (words)
  "Given a list of WORDS, return a corpus of character sequences with '_' EOW marker."
  (mapcar (lambda (w) (append (string->chars w) (list "_"))) words))

;;; ─── Step 2: Count adjacent pairs ──────────────────────────────

(defun count-pairs (corpus)
  "Return an alist of (pair . count) for all adjacent token pairs in CORPUS."
  (let ((counts '()))
    (dolist (seq corpus counts)
      (loop for (a b) on seq while b do
        (let* ((pair (list a b))
               (cell (assoc pair counts :test #'equal)))
          (if cell
              (incf (cdr cell))
              (push (cons pair 1) counts)))))))

(defun best-pair (counts)
  "Return the pair with the highest count."
  (car (reduce (lambda (best entry)
                 (if (> (cdr entry) (cdr best)) entry best))
               counts)))

;;; ─── Step 3: Apply a merge ──────────────────────────────────────

(defun apply-merge (seq a b)
  "In SEQ, replace every adjacent (A B) with the merged token AB."
  (let ((merged (concatenate 'string a b)))
    (loop with result = '()
          while seq
          do (if (and (string= (car seq) a)
                      (cdr seq)
                      (string= (cadr seq) b))
                 (progn (push merged result) (setf seq (cddr seq)))
                 (progn (push (car seq) result) (setf seq (cdr seq))))
          finally (return (reverse result)))))

(defun merge-corpus (corpus a b)
  "Apply merge of (A B) to every sequence in CORPUS."
  (mapcar (lambda (seq) (apply-merge seq a b)) corpus))

;;; ─── Step 4: Train BPE ──────────────────────────────────────────

(defun train-bpe (words vocab-size)
  "Train BPE on WORDS until vocabulary reaches VOCAB-SIZE.
   Returns (vocab . merge-rules) where vocab is an alist and
   merge-rules is an ordered list of (a b) pairs."
  (let* ((corpus (initial-corpus words))
         ;; Build initial vocab from all unique characters
         (chars (remove-duplicates
                 (apply #'append corpus) :test #'string=))
         (vocab (loop for c in chars for i from 0
                      collect (cons c i)))
         (merges '()))
    ;; BPE merge loop
    (loop while (< (length vocab) vocab-size)
          do (let ((counts (count-pairs corpus)))
               (when (null counts) (return))
               (let* ((pair (best-pair counts))
                      (a (car pair))
                      (b (cadr pair))
                      (new-tok (concatenate 'string a b))
                      (new-id (length vocab)))
                 (push (cons new-tok new-id) vocab)
                 (push pair merges)
                 (setf corpus (merge-corpus corpus a b)))))
    (cons (reverse vocab) (reverse merges))))

;;; ─── Step 5: Encode a string ────────────────────────────────────

(defun encode (string merges vocab)
  "Encode STRING to a list of token IDs using trained MERGES and VOCAB."
  (let ((tokens (append (string->chars string) (list "_"))))
    ;; Apply each merge rule in order
    (dolist (rule merges)
      (setf tokens (apply-merge tokens (car rule) (cadr rule))))
    ;; Look up IDs
    (mapcar (lambda (tok)
              (cdr (assoc tok vocab :test #'string=)))
            tokens)))

;;; ─── Demo ───────────────────────────────────────────────────────

(let* ((corpus-words '("low" "lower" "newest" "widest"))
       (result (train-bpe corpus-words 20))
       (vocab  (car result))
       (merges (cdr result)))
  (format t "Vocabulary (~a entries):~%" (length vocab))
  (dolist (entry vocab)
    (format t "  ~3a -> ~s~%" (cdr entry) (car entry)))
  (format t "~%Encoding 'low':  ~a~%" (encode "low" merges vocab))
  (format t "Encoding 'lower': ~a~%" (encode "lower" merges vocab)))
```

**Expected output:**

```
Vocabulary (16 entries):
  0  -> "l"
  1  -> "o"
  2  -> "w"
  ...
  9  -> "lo"
  10 -> "low"
  ...

Encoding 'low':  (10)
Encoding 'lower': (10 3 4 8)
```

---

## 1.5 Key Takeaways

- Text enters the model as a **sequence of integers** (token IDs).
- Tokens are **subword units**, not characters or whole words.
- The **vocabulary** is the fixed lookup table mapping tokens to IDs.
- BPE builds the vocabulary by **greedily merging** the most frequent pairs.
- The tokenizer is trained **once**; at inference time it only applies the stored merge rules.

> **What's next?** We have a sequence of integers: `[10, 3, 4, 8, …]`. These integers are just row indices. To give them geometric meaning — so the model can do math on them — we need an **embedding matrix**. That's Chapter 2.
