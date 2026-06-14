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

Let $V_0 = {c_1, c_2, \ldots}$ be the initial character vocabulary. After each merge step `k`:

$$
V_{k+1} = V_k \cup \{(a, b) \mid \text{freq}(a, b) \text{ is max over all adjacent pairs}\}
$$

The merge order defines a priority list $M = [(a_1,b_1), (a_2,b_2), \ldots]$. To encode a string:

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

## 1.4 The Code: A Micro-BPE in Scheme

We use Scheme throughout this book — it strips away syntactic ceremony so the *structure of computation* is visible. We build from named abstractions upward, one chunk at a time.

**The vocabulary abstraction.** Before writing any BPE logic, we name what a vocabulary *is*. A vocab maps string tokens to integer IDs. We expose three operations — create, extend, lookup — and never touch the underlying list structure from outside:

```scheme
(define (make-vocab) '())

(define (vocab-extend vocab token id)
  (cons (cons token id) vocab))

(define (vocab-lookup vocab token)
  (cond ((null? vocab) #f)
        ((string=? (caar vocab) token) (cdar vocab))
        (else (vocab-lookup (cdr vocab) token))))
```

The idea: name what you need, implement it once, and never revisit the representation again. Every line of BPE code above the `define` calls treats vocab as an opaque object.

**Characters and corpus.** A word enters BPE as a list of single-character strings, followed by `"_"` as an end-of-word marker. The marker lets BPE later discover that `"low"` is a complete unit while `"lower"` extends past it. `words->corpus` is simply `map` over words.

```scheme
(define (word->tokens word)
  (append (map string (string->list word)) (list "_")))

(define (words->corpus words)
  (map word->tokens words))
```

**Counting pairs.** Every adjacent pair `(a b)` in the corpus is a merge candidate. `adjacent-pairs` enumerates them recursively. `count-pairs` walks the entire corpus, tallying each pair in an association list. `most-frequent` then scans that list once, accumulating the best entry seen so far.

```scheme
(define (adjacent-pairs seq)
  (if (or (null? seq) (null? (cdr seq)))
      '()
      (cons (list (car seq) (cadr seq))
            (adjacent-pairs (cdr seq)))))

(define (tally! pair counts)
  (let ((cell (assoc pair counts equal?)))
    (if cell
        (begin (set-cdr! cell (+ (cdr cell) 1)) counts)
        (cons (cons pair 1) counts))))

(define (count-pairs corpus)
  (let tally-all ((seqs corpus) (counts '()))
    (if (null? seqs)
        counts
        (let tally-seq ((pairs (adjacent-pairs (car seqs))) (acc counts))
          (if (null? pairs)
              (tally-all (cdr seqs) acc)
              (tally-seq (cdr pairs) (tally! (car pairs) acc)))))))

(define (most-frequent counts)
  (let loop ((remaining (cdr counts)) (best (car counts)))
    (cond ((null? remaining) (car best))
          ((> (cdar remaining) (cdr best)) (loop (cdr remaining) (car remaining)))
          (else (loop (cdr remaining) best)))))
```

**Applying a merge.** Replacing every `(a b)` pair in a token sequence with the concatenated string `"ab"` has a natural recursive structure: if the head is `a` and the next element is `b`, emit `"ab"` and skip both; otherwise emit the head and continue. There are no indices, no mutation — just a list going in and a shorter list coming out.

```scheme
(define (merge-pair seq a b)
  (cond ((null? seq) '())
        ((null? (cdr seq)) seq)
        ((and (string=? (car seq) a) (string=? (cadr seq) b))
         (cons (string-append a b) (merge-pair (cddr seq) a b)))
        (else
         (cons (car seq) (merge-pair (cdr seq) a b)))))

(define (merge-corpus corpus a b)
  (map (lambda (seq) (merge-pair seq a b)) corpus))
```

**The training loop.** `train-bpe` is a single named-let with three accumulators: the current corpus, the growing vocabulary, and the ordered list of merges. Each recursive call is one BPE step: find the best pair, merge it, extend the vocab. The base case is reaching the target size, at which point both lists are reversed (since we prepend) and returned as a pair.

```scheme
(define (unique-tokens corpus)
  (let loop ((seqs corpus) (seen '()))
    (if (null? seqs) seen
        (let add ((tokens (car seqs)) (acc seen))
          (if (null? tokens) (loop (cdr seqs) acc)
              (add (cdr tokens)
                   (if (member (car tokens) acc) acc
                       (cons (car tokens) acc))))))))

(define (train-bpe words target-size)
  (let* ((corpus     (words->corpus words))
         (init-vocab (let number ((tokens (unique-tokens corpus)) (id 0) (v '()))
                       (if (null? tokens) (reverse v)
                           (number (cdr tokens) (+ id 1)
                                   (cons (cons (car tokens) id) v))))))
    (let bpe ((corpus corpus) (vocab init-vocab) (merges '()))
      (if (>= (length vocab) target-size)
          (cons (reverse vocab) (reverse merges))
          (let ((counts (count-pairs corpus)))
            (if (null? counts)
                (cons (reverse vocab) (reverse merges))
                (let* ((pair    (most-frequent counts))
                       (a       (car pair))
                       (b       (cadr pair))
                       (new-tok (string-append a b))
                       (new-id  (length vocab)))
                  (bpe (merge-corpus corpus a b)
                       (cons (cons new-tok new-id) vocab)
                       (cons pair merges)))))))))
```

**Encoding.** Given trained merge rules, encoding applies them in order to a fresh character sequence. After all rules have been applied, `map` converts each remaining token to its ID. Encoding is a fold over the merge list, with `merge-pair` as the step.

```scheme
(define (encode word merges vocab)
  (let apply-all ((tokens (word->tokens word)) (rules merges))
    (if (null? rules)
        (map (lambda (tok) (vocab-lookup vocab tok)) tokens)
        (apply-all (merge-pair tokens (caar rules) (cadar rules))
                   (cdr rules)))))
```

**Demo.** Run this with `mit-scheme --quiet --load micro-bpe.scm`:

```scheme
(let* ((words  '("low" "lower" "newest" "widest"))
       (result (train-bpe words 20))
       (vocab  (car result))
       (merges (cdr result)))
  (display "Vocabulary:") (newline)
  (for-each (lambda (e)
              (display "  ") (display (cdr e))
              (display " -> ") (display (car e)) (newline))
            vocab)
  (newline)
  (display "Encoding \"low\":   ") (display (encode "low"   merges vocab)) (newline)
  (display "Encoding \"lower\": ") (display (encode "lower" merges vocab)) (newline))
```

Expected output:

```
Vocabulary:
  0 -> l
  1 -> o
  ...
  9 -> lo
  10 -> low
  ...

Encoding "low":   (10)
Encoding "lower": (10 3 4 8)
```

---

## 1.5 Key Takeaways

- Text enters the model as a **sequence of integers** (token IDs).
- Tokens are **subword units**, not characters or whole words.
- The **vocabulary** is the fixed lookup table mapping tokens to IDs.
- BPE builds the vocabulary by **greedily merging** the most frequent pairs.
- The tokenizer is trained **once**; at inference time it only applies the stored merge rules.

> **What's next?** We have a sequence of integers: `[10, 3, 4, 8, …]`. These integers are just row indices. To give them geometric meaning — so the model can do math on them — we need an **embedding matrix**. That's Chapter 2.


---

![BPE merge sequence — each step replaces the most frequent byte pair with a new token.](images/ch01_bpe.png)

*BPE merge sequence — each step replaces the most frequent byte pair with a new token.*

![The full tokenization pipeline from raw text to integer IDs.](images/ch01_pipeline.png)

*The full tokenization pipeline from raw text to integer IDs.*

