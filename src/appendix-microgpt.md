# Appendix A: microGPT in Lisp — Complete Runnable Code

This appendix assembles every Lisp snippet from the book into a single file, `src/lisp/microgpt.lisp`, with all dependencies resolved. You can run it with any ANSI Common Lisp implementation (SBCL recommended).

## Installation & Running

```bash
# Install SBCL (Ubuntu/Debian)
apt-get install sbcl

# Run the full demo
sbcl --load src/lisp/microgpt.lisp

# Or with CLISP
clisp src/lisp/microgpt.lisp
```

## Annotated Assembly

The complete `microgpt.lisp` is in `src/lisp/microgpt.lisp`. It is organized in dependency order:

1. **Matrix primitives** — `make-array`, `mat-mul`, `mat-add`, `mat-scale`, `mat-transpose`, `hstack`
2. **Activations** — `softmax-row`, `softmax-rows`, `gelu`, `sigmoid`
3. **Embeddings** — `make-embedding-matrix`, `embed-sequence`
4. **Positional encoding** — `make-pe-matrix`, `add-positional-encoding`
5. **Attention** — `scaled-dot-product-attention`, `causal-mask`
6. **Multi-head attention** — `multi-head-attention`
7. **FFN** — `ffn-forward`
8. **Layer norm** — `layer-norm-matrix`
9. **Transformer block** — `transformer-block-forward`
10. **GPT model** — `gpt-forward`, `generate`

## The Tiny Shakespeare Demo

```lisp
;; Run after loading microgpt.lisp:
(tiny-shakespeare-demo)  ; trains for 100 steps on a 500-char corpus
```

See `src/lisp/microgpt.lisp` for the full training loop.

---

## microGPT Architecture Summary (Reference)

| Component | Input | Output | Parameters |
|-----------|-------|--------|------------|
| Token Embedding | `[T]` ints | $[T \times d]$ | $|V| \times d$ |
| Positional Embedding | `[T]` positions | $[T \times d]$ | $T_max \times d$ |
| × N Transformer Blocks | | | |
| — LayerNorm 1 | $[T \times d]$ | $[T \times d]$ | `2d` |
| — Multi-Head Attn | $[T \times d]$ | $[T \times d]$ | $4d^2$ |
| — Residual | $[T \times d]$ | $[T \times d]$ | 0 |
| — LayerNorm 2 | $[T \times d]$ | $[T \times d]$ | `2d` |
| — FFN | $[T \times d]$ | $[T \times d]$ | $8d^2$ |
| — Residual | $[T \times d]$ | $[T \times d]$ | 0 |
| Final LayerNorm | $[T \times d]$ | $[T \times d]$ | `2d` |
| Unembedding | `[d]` | `[|V|]` | $d \times |V|$ (tied) |

**Total parameters:** $2|V|d + T_max\cdot d + N(12d^2 + 4d) + 2d$

For GPT-2 small: `d=768, N=12, |V|=50257, T_max=1024` → ~117M params.
