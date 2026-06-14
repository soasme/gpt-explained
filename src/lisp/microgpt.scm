;;;; microgpt.scm
;;;; Complete microGPT implementation in Scheme
;;;; Assembles Chapters 2–9 of "GPT Explained" into one runnable file.
;;;;
;;;; Run:  mit-scheme --quiet --load microgpt.scm
;;;; Deps: None — pure R5RS Scheme, no external libraries

;;; ══════════════════════════════════════════════════════════════════
;;; §1  MATRIX PRIMITIVES
;;;
;;; A matrix is a list (rows cols data) where data is a flat vector
;;; stored in row-major order.  All higher-level code uses only the
;;; five procedures below — nothing touches the representation directly.
;;; ══════════════════════════════════════════════════════════════════

(define (make-matrix rows cols)
  (list rows cols (make-vector (* rows cols) 0.0)))

(define (matrix-rows M) (car M))
(define (matrix-cols M) (cadr M))
(define (matrix-data M) (caddr M))

(define (matrix-ref M i j)
  (vector-ref (matrix-data M) (+ (* i (matrix-cols M)) j)))

(define (matrix-set! M i j v)
  (vector-set! (matrix-data M) (+ (* i (matrix-cols M)) j) v))

(define (make-random-matrix rows cols)
  (let ((M     (make-matrix rows cols))
        (scale (sqrt (/ 2.0 (+ rows cols)))))
    (let row-loop ((i 0))
      (when (< i rows)
        (let col-loop ((j 0))
          (when (< j cols)
            (matrix-set! M i j (* scale (- (* 2.0 (random 1.0)) 1.0)))
            (col-loop (+ j 1))))
        (row-loop (+ i 1))))
    M))

(define (matrix-multiply A B)
  (let* ((m (matrix-rows A)) (k (matrix-cols A)) (n (matrix-cols B))
         (C (make-matrix m n)))
    (let loop-i ((i 0))
      (when (< i m)
        (let loop-j ((j 0))
          (when (< j n)
            (let loop-k ((p 0) (acc 0.0))
              (if (= p k)
                  (matrix-set! C i j acc)
                  (loop-k (+ p 1)
                          (+ acc (* (matrix-ref A i p) (matrix-ref B p j))))))
            (loop-j (+ j 1))))
        (loop-i (+ i 1))))
    C))

(define (matrix-transpose A)
  (let* ((m (matrix-rows A)) (n (matrix-cols A))
         (AT (make-matrix n m)))
    (let loop-i ((i 0))
      (when (< i m)
        (let loop-j ((j 0))
          (when (< j n)
            (matrix-set! AT j i (matrix-ref A i j))
            (loop-j (+ j 1))))
        (loop-i (+ i 1))))
    AT))

(define (matrix-scale A s)
  (let* ((m (matrix-rows A)) (n (matrix-cols A))
         (B (make-matrix m n)))
    (let loop-i ((i 0))
      (when (< i m)
        (let loop-j ((j 0))
          (when (< j n)
            (matrix-set! B i j (* s (matrix-ref A i j)))
            (loop-j (+ j 1))))
        (loop-i (+ i 1))))
    B))

(define (matrix-add A B)
  (let* ((m (matrix-rows A)) (n (matrix-cols A))
         (C (make-matrix m n)))
    (let loop-i ((i 0))
      (when (< i m)
        (let loop-j ((j 0))
          (when (< j n)
            (matrix-set! C i j (+ (matrix-ref A i j) (matrix-ref B i j)))
            (loop-j (+ j 1))))
        (loop-i (+ i 1))))
    C))

(define (hstack matrices)
  (let* ((T    (matrix-rows (car matrices)))
         (d    (apply + (map matrix-cols matrices)))
         (R    (make-matrix T d)))
    (let outer ((mats matrices) (offset 0))
      (unless (null? mats)
        (let* ((M  (car mats))
               (nc (matrix-cols M)))
          (let loop-i ((i 0))
            (when (< i T)
              (let loop-j ((j 0))
                (when (< j nc)
                  (matrix-set! R i (+ offset j) (matrix-ref M i j))
                  (loop-j (+ j 1))))
              (loop-i (+ i 1))))
          (outer (cdr mats) (+ offset nc)))))
    R))

(define (matrix-row M i)
  (let* ((d (matrix-cols M))
         (v (make-vector d)))
    (let loop ((j 0))
      (when (< j d)
        (vector-set! v j (matrix-ref M i j))
        (loop (+ j 1))))
    v))

;;; ══════════════════════════════════════════════════════════════════
;;; §2  ACTIVATIONS
;;; ══════════════════════════════════════════════════════════════════

(define (softmax v)
  (let* ((n     (vector-length v))
         (max-v (let loop ((i 1) (m (vector-ref v 0)))
                  (if (= i n) m (loop (+ i 1) (max m (vector-ref v i))))))
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

(define (softmax-each-row M)
  (let* ((T (matrix-rows M)) (n (matrix-cols M))
         (A (make-matrix T n)))
    (let loop ((i 0))
      (when (< i T)
        (let ((row (make-vector n)))
          (let copy ((j 0))
            (when (< j n)
              (vector-set! row j (matrix-ref M i j))
              (copy (+ j 1))))
          (let ((soft (softmax row)))
            (let copy-back ((j 0))
              (when (< j n)
                (matrix-set! A i j (vector-ref soft j))
                (copy-back (+ j 1))))))
        (loop (+ i 1))))
    A))

(define (gelu x)
  (let* ((c  (sqrt (/ 2.0 (* 4 (atan 1)))))
         (t  (tanh (* c (+ x (* 0.044715 x x x))))))
    (* x 0.5 (+ 1.0 t))))

(define (gelu-matrix M)
  (let* ((m (matrix-rows M)) (n (matrix-cols M))
         (R (make-matrix m n)))
    (let loop-i ((i 0))
      (when (< i m)
        (let loop-j ((j 0))
          (when (< j n)
            (matrix-set! R i j (gelu (matrix-ref M i j)))
            (loop-j (+ j 1))))
        (loop-i (+ i 1))))
    R))

;;; ══════════════════════════════════════════════════════════════════
;;; §3  EMBEDDINGS  (Chapter 2)
;;;
;;; embed-sequence looks up rows of the embedding matrix E for each
;;; token ID in the list and stacks them into a [T × d] matrix.
;;; ══════════════════════════════════════════════════════════════════

(define (embed-sequence E ids)
  (let* ((T (length ids))
         (d (matrix-cols E))
         (X (make-matrix T d)))
    (let loop ((ids ids) (t 0))
      (unless (null? ids)
        (let copy ((j 0))
          (when (< j d)
            (matrix-set! X t j (matrix-ref E (car ids) j))
            (copy (+ j 1))))
        (loop (cdr ids) (+ t 1))))
    X))

;;; ══════════════════════════════════════════════════════════════════
;;; §4  POSITIONAL ENCODING  (Chapter 3)
;;;
;;; PE[t, 2k]   = sin(t · ωₖ)    where ωₖ = 1 / 10000^(2k/d)
;;; PE[t, 2k+1] = cos(t · ωₖ)
;;; ══════════════════════════════════════════════════════════════════

(define (make-sinusoidal-pe max-len d)
  (let ((PE (make-matrix max-len d)))
    (let loop-t ((t 0))
      (when (< t max-len)
        (let loop-i ((i 0))
          (when (< i d)
            (let* ((k     (quotient i 2))
                   (omega (/ 1.0 (expt 10000.0 (/ (* 2.0 k) d)))))
              (matrix-set! PE t i
                           (if (even? i)
                               (sin (* t omega))
                               (cos (* t omega)))))
            (loop-i (+ i 1))))
        (loop-t (+ t 1))))
    PE))

;;; ══════════════════════════════════════════════════════════════════
;;; §5  LAYER NORMALIZATION  (Chapter 7)
;;;
;;; LayerNorm(x) = γ ⊙ (x − μ)/σ + β
;;; Applied independently to each row of a matrix.
;;; ══════════════════════════════════════════════════════════════════

(define (layer-norm-vec v gamma beta)
  (let* ((n   (vector-length v))
         (eps 1e-5)
         (mu  (let loop ((i 0) (s 0.0))
                (if (= i n) (/ s n)
                    (loop (+ i 1) (+ s (vector-ref v i))))))
         (var (let loop ((i 0) (s 0.0))
                (if (= i n) (/ s n)
                    (let ((d (- (vector-ref v i) mu)))
                      (loop (+ i 1) (+ s (* d d)))))))
         (std (sqrt (+ var eps)))
         (out (make-vector n)))
    (let loop ((i 0))
      (when (< i n)
        (vector-set! out i
                     (+ (* (vector-ref gamma i)
                           (/ (- (vector-ref v i) mu) std))
                        (vector-ref beta i)))
        (loop (+ i 1))))
    out))

(define (layer-norm-matrix M gamma beta)
  (let* ((T (matrix-rows M)) (d (matrix-cols M))
         (R (make-matrix T d)))
    (let loop ((i 0))
      (when (< i T)
        (let* ((row    (matrix-row M i))
               (normed (layer-norm-vec row gamma beta)))
          (let copy ((j 0))
            (when (< j d)
              (matrix-set! R i j (vector-ref normed j))
              (copy (+ j 1)))))
        (loop (+ i 1))))
    R))

;;; ══════════════════════════════════════════════════════════════════
;;; §6  CAUSAL MASK + SCALED DOT-PRODUCT ATTENTION  (Chapter 4)
;;;
;;; Attention(Q,K,V) = softmax( QKᵀ/√dₖ + mask ) · V
;;; Returns (cons output attention-weights).
;;; ══════════════════════════════════════════════════════════════════

(define (causal-mask T)
  (let ((M (make-matrix T T)))
    (let loop-i ((i 0))
      (when (< i T)
        (let loop-j ((j (+ i 1)))
          (when (< j T)
            (matrix-set! M i j -1.0e9)
            (loop-j (+ j 1))))
        (loop-i (+ i 1))))
    M))

(define (scaled-dot-product-attention Q K V)
  (let* ((dk     (exact->inexact (matrix-cols Q)))
         (T      (matrix-rows Q))
         (S      (matrix-scale (matrix-multiply Q (matrix-transpose K))
                               (/ 1.0 (sqrt dk))))
         (S-mask (matrix-add S (causal-mask T)))
         (A      (softmax-each-row S-mask))
         (output (matrix-multiply A V)))
    (cons output A)))

;;; ══════════════════════════════════════════════════════════════════
;;; §7  MULTI-HEAD ATTENTION  (Chapter 5)
;;;
;;; Each head: (wq wk wv), each [d × dk].  Output projection: wo [d × d].
;;; MHA(X) = concat(head₁, …, headₕ) · Wo
;;; ══════════════════════════════════════════════════════════════════

(define (make-head d dk)
  (list (make-random-matrix d dk)
        (make-random-matrix d dk)
        (make-random-matrix d dk)))

(define (head-wq h) (car h))
(define (head-wk h) (cadr h))
(define (head-wv h) (caddr h))

(define (make-mha-params d num-heads)
  (let* ((dk    (/ d num-heads))
         (heads (let build ((n num-heads) (acc '()))
                  (if (= n 0) acc
                      (build (- n 1) (cons (make-head d dk) acc))))))
    (list heads (make-random-matrix d d))))

(define (mha-heads p) (car p))
(define (mha-wo    p) (cadr p))

(define (multi-head-attention X params)
  (let* ((results (map (lambda (h)
                         (scaled-dot-product-attention
                          (matrix-multiply X (head-wq h))
                          (matrix-multiply X (head-wk h))
                          (matrix-multiply X (head-wv h))))
                       (mha-heads params)))
         (concat-out (hstack (map car results)))
         (output     (matrix-multiply concat-out (mha-wo params))))
    (cons output (map cdr results))))

;;; ══════════════════════════════════════════════════════════════════
;;; §8  FEED-FORWARD NETWORK  (Chapter 6)
;;;
;;; FFN(X) = GELU(X·W₁ + b₁)·W₂ + b₂
;;; W₁: [d × 4d] expands;  W₂: [4d × d] contracts.
;;; ══════════════════════════════════════════════════════════════════

(define (make-ffn-params d)
  (let ((dff (* d 4)))
    (list (make-random-matrix d dff)
          (make-vector dff 0.0)
          (make-random-matrix dff d)
          (make-vector d 0.0))))

(define (ffn-w1 p) (car p))
(define (ffn-b1 p) (cadr p))
(define (ffn-w2 p) (caddr p))
(define (ffn-b2 p) (cadddr p))

(define (add-bias M b)
  (let* ((m (matrix-rows M)) (n (matrix-cols M))
         (R (make-matrix m n)))
    (let loop-i ((i 0))
      (when (< i m)
        (let loop-j ((j 0))
          (when (< j n)
            (matrix-set! R i j (+ (matrix-ref M i j) (vector-ref b j)))
            (loop-j (+ j 1))))
        (loop-i (+ i 1))))
    R))

(define (ffn-forward X params)
  (let* ((H  (add-bias (matrix-multiply X (ffn-w1 params)) (ffn-b1 params)))
         (H' (gelu-matrix H)))
    (add-bias (matrix-multiply H' (ffn-w2 params)) (ffn-b2 params))))

;;; ══════════════════════════════════════════════════════════════════
;;; §9  TRANSFORMER BLOCK  (Chapter 7)
;;;
;;; Pre-LN block:
;;;   x₁ = x  + MHA(LayerNorm₁(x))
;;;   x₂ = x₁ + FFN(LayerNorm₂(x₁))
;;; ══════════════════════════════════════════════════════════════════

(define (make-ln-params d)
  (list (make-vector d 1.0)   ; gamma
        (make-vector d 0.0))) ; beta

(define (ln-gamma p) (car p))
(define (ln-beta  p) (cadr p))

(define (make-block-params d num-heads)
  (list (make-mha-params d num-heads)
        (make-ffn-params d)
        (make-ln-params d)
        (make-ln-params d)))

(define (block-mha  p) (car p))
(define (block-ffn  p) (cadr p))
(define (block-ln1  p) (caddr p))
(define (block-ln2  p) (cadddr p))

(define (transformer-block-forward X params)
  (let* ((ln1       (block-ln1 params))
         (ln2       (block-ln2 params))
         (X-norm1   (layer-norm-matrix X (ln-gamma ln1) (ln-beta ln1)))
         (mha-out   (car (multi-head-attention X-norm1 (block-mha params))))
         (X1        (matrix-add X mha-out))
         (X-norm2   (layer-norm-matrix X1 (ln-gamma ln2) (ln-beta ln2)))
         (ffn-out   (ffn-forward X-norm2 (block-ffn params))))
    (matrix-add X1 ffn-out)))

;;; ══════════════════════════════════════════════════════════════════
;;; §10  GPT MODEL  (Chapter 8)
;;;
;;; Full forward pass:
;;;   1. Token embeddings + position embeddings
;;;   2. N transformer blocks
;;;   3. Final layer norm on last token's vector
;;;   4. Unembedding: logits[i] = h · wte[i]  (weight tying)
;;; ══════════════════════════════════════════════════════════════════

(define (make-gpt-config vocab-size d-model num-heads num-layers max-seq-len)
  (list vocab-size d-model num-heads num-layers max-seq-len))

(define (config-vocab-size  c) (car c))
(define (config-d-model     c) (cadr c))
(define (config-num-heads   c) (caddr c))
(define (config-num-layers  c) (cadddr c))
(define (config-max-seq-len c) (car (cddddr c)))

(define (make-gpt-params config)
  (let ((V (config-vocab-size  config))
        (d (config-d-model     config))
        (H (config-num-heads   config))
        (N (config-num-layers  config))
        (T (config-max-seq-len config)))
    (list
     (make-random-matrix V d)  ; wte: token embeddings [V × d]
     (make-random-matrix T d)  ; wpe: position embeddings [T × d]
     (let build ((n N) (acc '()))
       (if (= n 0) acc
           (build (- n 1) (cons (make-block-params d H) acc))))
     (make-ln-params d))))     ; final layer norm

(define (gpt-wte    p) (car p))
(define (gpt-wpe    p) (cadr p))
(define (gpt-blocks p) (caddr p))
(define (gpt-ln-f   p) (cadddr p))

(define (gpt-forward ids params config)
  (let* ((T     (length ids))
         (d     (config-d-model     config))
         (V     (config-vocab-size  config))
         (wte   (gpt-wte    params))
         (wpe   (gpt-wpe    params))
         (ln-f  (gpt-ln-f   params))
         ;; 1. Embeddings
         (pos-ids (let build ((i 0) (acc '()))
                    (if (= i T) (reverse acc) (build (+ i 1) (cons i acc)))))
         (X    (matrix-add (embed-sequence wte ids)
                           (embed-sequence wpe pos-ids)))
         ;; 2. Transformer blocks
         (X-out (let loop ((X X) (blocks (gpt-blocks params)))
                  (if (null? blocks) X
                      (loop (transformer-block-forward X (car blocks))
                            (cdr blocks)))))
         ;; 3. Final layer norm on last token
         (last-row (matrix-row X-out (- T 1)))
         (h        (layer-norm-vec last-row (ln-gamma ln-f) (ln-beta ln-f)))
         ;; 4. Unembedding (weight tying: logits[i] = h · wte[i])
         (logits (make-vector V)))
    (let loop ((i 0))
      (when (< i V)
        (let dot ((j 0) (acc 0.0))
          (if (= j d)
              (vector-set! logits i acc)
              (dot (+ j 1) (+ acc (* (vector-ref h j) (matrix-ref wte i j))))))
        (loop (+ i 1))))
    logits))

;;; ══════════════════════════════════════════════════════════════════
;;; §11  SAMPLING + GENERATION
;;; ══════════════════════════════════════════════════════════════════

(define (temperature-scale logits temp)
  (let* ((n (vector-length logits))
         (v (make-vector n)))
    (let loop ((i 0))
      (when (< i n)
        (vector-set! v i (/ (vector-ref logits i) temp))
        (loop (+ i 1))))
    v))

(define (top-k-filter logits k)
  (let* ((n       (vector-length logits))
         (indexed (let build ((i 0) (acc '()))
                    (if (= i n) acc
                        (build (+ i 1)
                               (cons (cons i (vector-ref logits i)) acc)))))
         (sorted  (sort indexed (lambda (a b) (> (cdr a) (cdr b)))))
         (result  (make-vector n -1.0e9)))
    (let loop ((entries sorted) (count 0))
      (when (and (not (null? entries)) (< count k))
        (vector-set! result (caar entries) (vector-ref logits (caar entries)))
        (loop (cdr entries) (+ count 1))))
    result))

(define (sample-token probs)
  (let loop ((i 0) (r (random 1.0)))
    (if (= i (vector-length probs))
        (- (vector-length probs) 1)
        (let ((r-next (- r (vector-ref probs i))))
          (if (<= r-next 0.0) i
              (loop (+ i 1) r-next))))))

(define (generate params config start-ids max-new temperature top-k)
  (let loop ((tokens start-ids) (new '()) (count 0))
    (if (= count max-new)
        (reverse new)
        (let* ((logits   (gpt-forward tokens params config))
               (scaled   (temperature-scale logits temperature))
               (filtered (top-k-filter scaled top-k))
               (probs    (softmax filtered))
               (next     (sample-token probs)))
          (loop (append tokens (list next))
                (cons next new)
                (+ count 1))))))

;;; ══════════════════════════════════════════════════════════════════
;;; §12  PARAMETER COUNT
;;; ══════════════════════════════════════════════════════════════════

(define (count-parameters config)
  (let ((V (config-vocab-size  config))
        (d (config-d-model     config))
        (N (config-num-layers  config))
        (T (config-max-seq-len config)))
    (+  (* V d)               ; wte
        (* T d)               ; wpe
        (* N (+
              (* 4 d d)       ; MHA: Wq Wk Wv Wo
              (* 8 d d)       ; FFN: W1 W2 (4d each)
              (* 4 d)))       ; LayerNorm gamma+beta × 2
        (* 2 d))))            ; final LayerNorm

;;; ══════════════════════════════════════════════════════════════════
;;; §13  DEMO
;;; ══════════════════════════════════════════════════════════════════

(define *config*
  (make-gpt-config 100    ; vocab-size
                   32     ; d-model
                   4      ; num-heads
                   2      ; num-layers
                   64))   ; max-seq-len

(display "microGPT in Scheme") (newline)
(display "══════════════════════════════════════") (newline)
(display "vocab-size  : ") (display (config-vocab-size  *config*)) (newline)
(display "d-model     : ") (display (config-d-model     *config*)) (newline)
(display "num-heads   : ") (display (config-num-heads   *config*)) (newline)
(display "num-layers  : ") (display (config-num-layers  *config*)) (newline)
(display "max-seq-len : ") (display (config-max-seq-len *config*)) (newline)
(display "parameters  : ") (display (count-parameters   *config*)) (newline)
(newline)

(display "Building model...") (newline)
(define *model* (make-gpt-params *config*))
(display "Done.") (newline)
(newline)

(display "Forward pass on prompt (1 2 3 4 5)...") (newline)
(let* ((prompt '(1 2 3 4 5))
       (logits (gpt-forward prompt *model* *config*))
       (probs  (softmax logits)))
  (display "Logits (first 10): ")
  (let loop ((i 0))
    (when (< i 10)
      (display (vector-ref logits i)) (display " ")
      (loop (+ i 1))))
  (newline)
  (let best ((i 0) (best-i 0) (best-p 0.0))
    (if (= i (vector-length probs))
        (begin
          (display "Top predicted token: ") (display best-i)
          (display " (prob ") (display best-p) (display ")") (newline))
        (let ((p (vector-ref probs i)))
          (if (> p best-p)
              (best (+ i 1) i p)
              (best (+ i 1) best-i best-p))))))
(newline)

(display "Generating 10 tokens from prompt (1 2 3)...") (newline)
(let ((generated (generate *model* *config* '(1 2 3) 10 0.8 10)))
  (display "Generated: ") (display generated) (newline)
  (display "Full sequence: ")
  (display (append '(1 2 3) generated)) (newline))
(newline)
(display "microGPT demo complete.") (newline)
