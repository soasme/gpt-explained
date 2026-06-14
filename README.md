# GPT Explained

> *A hands-on guide to transformer architecture — from tokens to text generation, with math, matrices, Scheme, and animations.*

**Read online:** https://www.soasme.com/gpt-explained/

---

## Contents

| Chapter | Topic |
|---------|-------|
| 1 | **Tokens** — Text to Numbers (BPE tokenization) |
| 2 | **Embeddings** — Numbers to Meaning |
| 3 | **Positional Encoding** — Giving Order to Meaning |
| 4 | **Attention** — Tokens Talking to Each Other |
| 5 | **Multi-Head Attention** — Many Conversations at Once |
| 6 | **Feed-Forward Network** — The Model's Memory |
| 7 | **The Transformer Block** — Putting It Together |
| 8 | **Vocabulary Projection** — From Vectors to Words |
| 9 | **Loss** — Are we adjusting the model weights the right way? |
| 10 | **Training** — Backpropagation to adjust weights |
| A | **microGPT in Scheme** — Complete Runnable Code |
| B | **Math Primer** — Every prerequisite in one place |

---

## Building Locally

### Prerequisites

```bash
# macOS — install system dependencies
brew bundle

# Install Ruby gem dependencies
bundle install
```

The `Brewfile` installs Ruby, cmake, bison, flex, cairo, pango, gdk-pixbuf, and fontconfig — all required to compile the `mathematical` gem for LaTeX rendering in PDF output.

### Build all formats

```bash
bundle exec rake book:build
```

Output files:

| File | Format |
|------|--------|
| `gpt-explained.html` | Single-page HTML (self-contained) |
| `gpt-explained.epub` | EPUB 3 |
| `gpt-explained.fb2.zip` | FictionBook 2 |
| `gpt-explained.pdf` | PDF with rendered LaTeX |

### Generate diagrams

Diagrams are animated Manim scenes. To regenerate:

```bash
cd src/manim
python -m manim <scene>.py <SceneName> --format=png -ql   # fast preview
python -m manim <scene>.py <SceneName> --format=png -qh   # production
```

CI rebuilds all images on each push before building the book.

---

## Running microGPT

```bash
# With Guile Scheme
guile book/A-microgpt/microgpt.scm

# With Racket (in R5RS mode)
racket --language r5rs book/A-microgpt/microgpt.scm
```

The model is randomly initialized — for non-trivial outputs, train it on text.

---

## CI / Deployment

Every push to `main`:

1. Installs system and Ruby dependencies
2. Builds all book formats via `bundle exec rake book:build`
3. Creates a versioned GitHub Release with HTML, EPUB, FB2, and PDF as assets
4. Publishes HTML, PDF, and EPUB to GitHub Pages

---

## License

Book text and code: [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported](https://creativecommons.org/licenses/by-nc-sa/3.0) (CC BY-NC-SA 3.0).
