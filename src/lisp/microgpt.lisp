;;;; microgpt.lisp
;;;; Complete microGPT implementation in Common Lisp
;;;; Chapters 1-8 of "GPT Explained" assembled into one file.
;;;;
;;;; Run:  sbcl --load microgpt.lisp
;;;; Deps: None (pure ANSI Common Lisp, no external libraries)

(declaim (optimize (speed 2) (safety 1)))

;;; ════════════════════════════════════════════════════════════════
;;; §1 MATRIX PRIMITIVES
;;; ════════════════════════════════════════════════════════════════

(defun make-matrix (rows cols &optional (init 0.0))
  (make-array (list rows cols) :element-type 'single-float
              :initial-element (float init 1.0)))

(defun mat-ref  (M i j)      (aref M i j))
(defun mat-set! (M i j v)    (setf (aref M i j) v))
(defun mat-rows (M)          (array-dimension M 0))
(defun mat-cols (M)          (array-dimension M 1))

(defun mat-mul (A B)
  "Matrix multiplication: A [m×k] · B [k×n] → [m×n]"
  (let* ((m (mat-rows A)) (k (mat-cols A)) (n (mat-cols B))
         (C (make-matrix m n)))
    (dotimes (i m C)
      (dotimes (j n)
        (dotimes (p k)
          (incf (aref C i j) (* (aref A i p) (aref B p j))))))))

(defun mat-add (A B)
  "Element-wise addition, same shape."
  (let* ((m (mat-rows A)) (n (mat-cols A))
         (C (make-matrix m n)))
    (dotimes (i m C)
      (dotimes (j n)
        (setf (aref C i j) (+ (aref A i j) (aref B i j)))))))

(defun mat-transpose (A)
  (let* ((m (mat-rows A)) (n (mat-cols A))
         (AT (make-matrix n m)))
    (dotimes (i m AT)
      (dotimes (j n)
        (setf (aref AT j i) (aref A i j))))))

(defun mat-scale (A scalar)
  (let* ((m (mat-rows A)) (n (mat-cols A))
         (B (make-matrix m n)))
    (dotimes (i m B)
      (dotimes (j n)
        (setf (aref B i j) (* scalar (aref A i j)))))))

(defun hstack (matrices)
  "Concatenate 2D matrices column-wise."
  (let* ((rows (mat-rows (car matrices)))
         (total-cols (reduce #'+ matrices :key #'mat-cols))
         (result (make-matrix rows total-cols))
         (col-offset 0))
    (dolist (M matrices result)
      (let ((cols (mat-cols M)))
        (dotimes (i rows)
          (dotimes (j cols)
            (setf (aref result i (+ col-offset j)) (aref M i j))))
        (incf col-offset cols)))))

(defun rand-matrix (rows cols)
  "Xavier uniform init."
  (let* ((scale (sqrt (/ 2.0 (+ rows cols))))
         (M (make-matrix rows cols)))
    (dotimes (i rows M)
      (dotimes (j cols)
        (setf (aref M i j) (* scale (- (random 2.0) 1.0)))))))

(defun extract-row (M i)
  "Return row I of matrix M as a 1D array."
  (let* ((cols (mat-cols M))
         (v (make-array cols :element-type 'single-float)))
    (dotimes (j cols v)
      (setf (aref v j) (aref M i j)))))

(defun print-matrix (M &optional (label "M"))
  (format t "~%~a [~a×~a]:~%" label (mat-rows M) (mat-cols M))
  (dotimes (i (mat-rows M))
    (format t "  [")
    (dotimes (j (mat-cols M))
      (format t "~8,3f" (aref M i j)))
    (format t " ]~%")))

;;; ════════════════════════════════════════════════════════════════
;;; §2 ACTIVATIONS
;;; ════════════════════════════════════════════════════════════════

(defun softmax-vec (v)
  "Softmax over a 1D float array."
  (let* ((n (length v))
         (max-v (reduce #'max v))
         (exps (make-array n :element-type 'single-float))
         (sum 0.0))
    (dotimes (i n)
      (let ((e (exp (- (aref v i) max-v))))
        (setf (aref exps i) e)
        (incf sum e)))
    (dotimes (i n exps)
      (setf (aref exps i) (/ (aref exps i) sum)))))

(defun softmax-rows (M)
  "Apply softmax to each row of M."
  (let* ((T-len (mat-rows M))
         (cols  (mat-cols M))
         (A     (make-matrix T-len cols)))
    (dotimes (i T-len A)
      (let* ((row (make-array cols :element-type 'single-float))
             (sr nil))
        (dotimes (j cols) (setf (aref row j) (aref M i j)))
        (setf sr (softmax-vec row))
        (dotimes (j cols) (setf (aref A i j) (aref sr j)))))))

(defun gelu (x)
  "GELU activation (tanh approximation)."
  (let* ((c  (coerce (sqrt (/ 2.0 pi)) 'single-float))
         (x3 (* x x x))
         (t1 (tanh (* c (+ x (* 0.044715 x3))))))
    (* x 0.5 (+ 1.0 t1))))

(defun gelu-matrix (M)
  (let* ((m (mat-rows M)) (n (mat-cols M))
         (R (make-matrix m n)))
    (dotimes (i m R)
      (dotimes (j n)
        (setf (aref R i j) (gelu (aref M i j)))))))

;;; ════════════════════════════════════════════════════════════════
;;; §3 EMBEDDINGS  (Chapter 2)
;;; ════════════════════════════════════════════════════════════════

(defun embed-sequence (E ids)
  "Look up rows of E for each id in IDS. Returns [T × d] matrix."
  (let* ((T-len (length ids))
         (d     (mat-cols E))
         (X     (make-matrix T-len d)))
    (loop for id in ids for t from 0 do
      (dotimes (j d)
        (setf (aref X t j) (aref E id j))))
    X))

;;; ════════════════════════════════════════════════════════════════
;;; §4 POSITIONAL ENCODING  (Chapter 3)
;;; ════════════════════════════════════════════════════════════════

(defun make-sinusoidal-pe (max-len d)
  "Sinusoidal PE matrix [max-len × d]."
  (let ((PE (make-matrix max-len d)))
    (dotimes (t-pos max-len PE)
      (dotimes (i d)
        (let* ((k (floor i 2))
               (omega (/ 1.0 (expt 10000.0 (/ (* 2.0 k) d)))))
          (setf (aref PE t-pos i)
                (if (evenp i)
                    (sin (* t-pos omega))
                    (cos (* t-pos omega)))))))))

;;; ════════════════════════════════════════════════════════════════
;;; §5 LAYER NORMALIZATION  (Chapter 7)
;;; ════════════════════════════════════════════════════════════════

(defun layer-norm-vec (v gamma beta &optional (eps 1e-5))
  (let* ((n   (length v))
         (mu  (/ (reduce #'+ v) n))
         (var (/ (reduce #'+ (map 'vector (lambda (x) (expt (- x mu) 2)) v)) n))
         (std (sqrt (+ var eps)))
         (out (make-array n :element-type 'single-float)))
    (dotimes (i n out)
      (setf (aref out i)
            (+ (* (aref gamma i) (/ (- (aref v i) mu) std))
               (aref beta i))))))

(defun layer-norm-matrix (M gamma beta &optional (eps 1e-5))
  (let* ((T-len (mat-rows M)) (d (mat-cols M))
         (R (make-matrix T-len d)))
    (dotimes (i T-len R)
      (let ((row (extract-row M i)))
        (let ((normed (layer-norm-vec row gamma beta eps)))
          (dotimes (j d) (setf (aref R i j) (aref normed j))))))))

;;; ════════════════════════════════════════════════════════════════
;;; §6 CAUSAL MASK + ATTENTION  (Chapter 4)
;;; ════════════════════════════════════════════════════════════════

(defun causal-mask (T-len)
  (let ((M (make-matrix T-len T-len)))
    (dotimes (i T-len M)
      (loop for j from (1+ i) below T-len do
        (setf (aref M i j) -1.0e9)))))

(defun sdpa (Q K V &key (mask t))
  "Scaled dot-product attention.
   Q,K : [T×dk],  V : [T×dv] → returns ([T×dv], [T×T])"
  (let* ((dk (float (mat-cols Q)))
         (T-len (mat-rows Q))
         (S (mat-scale (mat-mul Q (mat-transpose K)) (/ 1.0 (sqrt dk))))
         (S-m (if mask (mat-add S (causal-mask T-len)) S))
         (A (softmax-rows S-m))
         (out (mat-mul A V)))
    (values out A)))

;;; ════════════════════════════════════════════════════════════════
;;; §7 MULTI-HEAD ATTENTION  (Chapter 5)
;;; ════════════════════════════════════════════════════════════════

(defstruct mha-params heads wo)

(defun make-mha (d num-heads)
  (assert (zerop (mod d num-heads)))
  (let ((dk (/ d num-heads)))
    (make-mha-params
     :heads (loop repeat num-heads
                  collect (list :wq (rand-matrix d dk)
                                :wk (rand-matrix d dk)
                                :wv (rand-matrix d dk)))
     :wo (rand-matrix d d))))

(defun mha-forward (X params)
  (let ((head-outs '()))
    (dolist (h (mha-params-heads params))
      (let* ((Q (mat-mul X (getf h :wq)))
             (K (mat-mul X (getf h :wk)))
             (V (mat-mul X (getf h :wv))))
        (push (sdpa Q K V) head-outs)))
    (mat-mul (hstack (reverse head-outs)) (mha-params-wo params))))

;;; ════════════════════════════════════════════════════════════════
;;; §8 FEED-FORWARD NETWORK  (Chapter 6)
;;; ════════════════════════════════════════════════════════════════

(defstruct ffn-params w1 b1 w2 b2)

(defun make-ffn (d &optional (ratio 4))
  (let ((dff (* d ratio)))
    (make-ffn-params
     :w1 (rand-matrix d dff)
     :b1 (make-array dff :element-type 'single-float :initial-element 0.0)
     :w2 (rand-matrix dff d)
     :b2 (make-array d :element-type 'single-float :initial-element 0.0))))

(defun add-bias (M b)
  (let* ((m (mat-rows M)) (n (mat-cols M))
         (R (make-matrix m n)))
    (dotimes (i m R)
      (dotimes (j n)
        (setf (aref R i j) (+ (aref M i j) (aref b j)))))))

(defun ffn-forward (X params)
  (let* ((H  (add-bias (mat-mul X (ffn-params-w1 params)) (ffn-params-b1 params)))
         (H' (gelu-matrix H)))
    (add-bias (mat-mul H' (ffn-params-w2 params)) (ffn-params-b2 params))))

;;; ════════════════════════════════════════════════════════════════
;;; §9 TRANSFORMER BLOCK  (Chapter 7)
;;; ════════════════════════════════════════════════════════════════

(defstruct ln-params gamma beta)

(defun make-ln (d)
  (make-ln-params
   :gamma (make-array d :element-type 'single-float :initial-element 1.0)
   :beta  (make-array d :element-type 'single-float :initial-element 0.0)))

(defstruct block-params mha ffn ln1 ln2)

(defun make-block (d num-heads)
  (make-block-params :mha (make-mha d num-heads)
                     :ffn (make-ffn d)
                     :ln1 (make-ln d)
                     :ln2 (make-ln d)))

(defun block-forward (X params)
  "Pre-LN transformer block:
   x1 = x  + MHA(LN1(x))
   x2 = x1 + FFN(LN2(x1))"
  (let* ((ln1 (block-params-ln1 params))
         (ln2 (block-params-ln2 params))
         (X-normed-1 (layer-norm-matrix X (ln-params-gamma ln1) (ln-params-beta ln1)))
         (mha-out    (mha-forward X-normed-1 (block-params-mha params)))
         (X1         (mat-add X mha-out))
         (X-normed-2 (layer-norm-matrix X1 (ln-params-gamma ln2) (ln-params-beta ln2)))
         (ffn-out    (ffn-forward X-normed-2 (block-params-ffn params))))
    (mat-add X1 ffn-out)))

;;; ════════════════════════════════════════════════════════════════
;;; §10 GPT MODEL  (Chapter 8)
;;; ════════════════════════════════════════════════════════════════

(defstruct gpt-config
  vocab-size d-model num-heads num-layers max-seq-len)

(defstruct gpt-params
  wte  ; token embedding  [V × d]
  wpe  ; position embed   [T_max × d]
  blocks
  ln-f ; final layer norm
  )

(defun make-gpt (config)
  (let ((V (gpt-config-vocab-size  config))
        (d (gpt-config-d-model     config))
        (H (gpt-config-num-heads   config))
        (N (gpt-config-num-layers  config))
        (T (gpt-config-max-seq-len config)))
    (make-gpt-params
     :wte    (rand-matrix V d)
     :wpe    (rand-matrix T d)
     :blocks (loop repeat N collect (make-block d H))
     :ln-f   (make-ln d))))

(defun gpt-forward (ids params config)
  "Full GPT forward pass. Returns logits [|V|] for last position."
  (let* ((T-len (length ids))
         (d     (gpt-config-d-model config))
         (V     (gpt-config-vocab-size config))
         (wte   (gpt-params-wte params))
         (wpe   (gpt-params-wpe params))
         (ln-f  (gpt-params-ln-f params))

         ;; 1. token + position embeddings
         (X (mat-add
             (embed-sequence wte ids)
             (embed-sequence wpe (loop for i below T-len collect i))))

         ;; 2. N transformer blocks
         (X-out (reduce #'block-forward (gpt-params-blocks params)
                        :initial-value X))

         ;; 3. Final LN on last row
         (last-row (extract-row X-out (1- T-len)))
         (h (layer-norm-vec last-row
                            (ln-params-gamma ln-f)
                            (ln-params-beta  ln-f)))

         ;; 4. Unembedding: logits[i] = h · wte[i]  (weight tying)
         (logits (make-array V :element-type 'single-float)))

    (dotimes (i V logits)
      (let ((s 0.0))
        (dotimes (j d)
          (incf s (* (aref h j) (aref wte i j))))
        (setf (aref logits i) s)))))

;;; ════════════════════════════════════════════════════════════════
;;; §11 SAMPLING + GENERATION
;;; ════════════════════════════════════════════════════════════════

(defun temperature-scale (logits temp)
  (let ((v (copy-seq logits)))
    (dotimes (i (length v) v)
      (setf (aref v i) (/ (aref v i) temp)))))

(defun top-k-filter (logits k)
  "Set all but top-K logits to -1e9."
  (let* ((n (length logits))
         (indexed (loop for i below n collect (cons i (aref logits i))))
         (sorted  (sort indexed #'> :key #'cdr))
         (result  (make-array n :element-type 'single-float
                                :initial-element -1.0e9)))
    (loop for (idx . _) in (subseq sorted 0 (min k n)) do
      (setf (aref result idx) (aref logits idx)))
    result))

(defun sample (probs)
  "Sample a token index from a probability array."
  (let ((r (random 1.0)) (cum 0.0))
    (dotimes (i (length probs) (1- (length probs)))
      (incf cum (aref probs i))
      (when (>= cum r) (return i)))))

(defun generate (params config start-ids
                 &key (max-new 20) (temperature 1.0) (top-k 40))
  "Autoregressive generation. Returns new token IDs."
  (let ((tokens (copy-list start-ids))
        (new-tokens '()))
    (dotimes (_ max-new (reverse new-tokens))
      (let* ((logits  (gpt-forward tokens params config))
             (scaled  (temperature-scale logits temperature))
             (filtered (top-k-filter scaled top-k))
             (probs   (softmax-vec filtered))
             (next    (sample probs)))
        (push next new-tokens)
        (setf tokens (append tokens (list next)))))))

;;; ════════════════════════════════════════════════════════════════
;;; §12 PARAMETER COUNT
;;; ════════════════════════════════════════════════════════════════

(defun count-params (config)
  (let* ((V (gpt-config-vocab-size  config))
         (d (gpt-config-d-model     config))
         (N (gpt-config-num-layers  config))
         (T (gpt-config-max-seq-len config)))
    (+ (* V d)        ; wte
       (* T d)        ; wpe
       (* N (+ (* 4 d d)    ; MHA: Wq Wk Wv Wo
               (* 8 d d)    ; FFN: W1 W2 (4d each)
               (* 4 d)))    ; LN params
       (* 2 d))))     ; final LN

;;; ════════════════════════════════════════════════════════════════
;;; §13 DEMO
;;; ════════════════════════════════════════════════════════════════

(defparameter *config*
  (make-gpt-config :vocab-size 100
                   :d-model 32
                   :num-heads 4
                   :num-layers 2
                   :max-seq-len 64))

(format t "~%╔══════════════════════════════════════╗~%")
(format t "║        microGPT in Lisp              ║~%")
(format t "║  GPT Explained — github.com/soasme   ║~%")
(format t "╚══════════════════════════════════════╝~%")
(format t "~%Config:~%")
(format t "  vocab-size : ~a~%" (gpt-config-vocab-size *config*))
(format t "  d-model    : ~a~%" (gpt-config-d-model *config*))
(format t "  num-heads  : ~a~%" (gpt-config-num-heads *config*))
(format t "  num-layers : ~a~%" (gpt-config-num-layers *config*))
(format t "  max-seq-len: ~a~%" (gpt-config-max-seq-len *config*))
(format t "  parameters : ~a~%" (count-params *config*))

(format t "~%Building model...~%")
(defparameter *model* (make-gpt *config*))
(format t "Done.~%")

(format t "~%Running forward pass on prompt [1 2 3 4 5]...~%")
(let* ((prompt '(1 2 3 4 5))
       (logits (gpt-forward prompt *model* *config*))
       (probs  (softmax-vec logits)))
  (format t "Logits (first 10): ~{~,3f ~}...~%"
          (loop for i below 10 collect (aref logits i)))
  (format t "Top predicted token: ~a (prob ~,4f)~%"
          (position (reduce #'max probs) probs)
          (reduce #'max probs)))

(format t "~%Generating 10 new tokens from prompt [1 2 3]...~%")
(let ((generated (generate *model* *config* '(1 2 3)
                            :max-new 10 :temperature 0.8 :top-k 10)))
  (format t "Generated: ~a~%" generated)
  (format t "Full sequence: ~a~%" (append '(1 2 3) generated)))

(format t "~%microGPT demo complete.~%")
