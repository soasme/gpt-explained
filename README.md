# GPT Explained

> *A hands-on guide to transformer architecture — from tokens to text generation, with math, matrices, Python, and animations.*

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
| A | **microGPT in Python** — Complete Runnable Code |
| B | **Math Primer** — Every prerequisite in one place |

---

## Building Locally

### Prerequisites

```bash
# macOS — install system dependencies
brew bundle

# Install Ruby gem dependencies
bundle install

# Python 3 is required for the runnable book code.
python3 --version
```

The `Brewfile` installs Ruby, cmake, bison, flex, cairo, pango, gdk-pixbuf, and fontconfig — all required to compile the `mathematical` gem for LaTeX rendering in PDF output.
The book code uses only the Python standard library.

### Run code examples

The Python files in `src/python/` are the source of truth for code shown in the book.
AsciiDoc listings include tagged regions from those files, so update the Python first and let the book reference it.

```bash
# Run the complete microGPT demo
python3 src/python/microgpt.py

# Run the tiny BPE tokenizer demo
python3 src/python/micro_bpe.py

# Run every chapter's end-to-end code check
python3 src/python/run_book_code.py

# Same check through Rake
bundle exec rake book:run_code
```

If Bundler is not available in your shell, `rake book:run_code` runs the same Python check.

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

Diagrams are generated with Matplotlib scripts. To regenerate:

```bash
src/matplotlib/render_all.sh
```

CI rebuilds all images on each push before building the book.

---

## Python Source Layout

| File | Purpose |
|------|---------|
| `src/python/microgpt.py` | Core matrix, transformer, loss, training, and generation code |
| `src/python/micro_bpe.py` | Chapter 3 byte-pair encoding tokenizer |
| `src/python/chapter_demos.py` | Runnable chapter demos used by the end-to-end check |
| `src/python/run_book_code.py` | End-to-end runner for all chapter code |

The demo model is randomly initialized with fixed seeds for reproducible checks.
For non-trivial outputs, it would need real training data and a training loop beyond the book demo.

---

## CI / Deployment

Every push to `main`:

1. Installs system and Ruby dependencies
2. Runs the Python chapter code check via `bundle exec rake book:run_code`
3. Builds all book formats via `bundle exec rake book:build`
4. Creates a versioned GitHub Release with HTML, EPUB, FB2, and PDF as assets
5. Publishes HTML, PDF, and EPUB to GitHub Pages

---

## License

Book text and code: [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported](https://creativecommons.org/licenses/by-nc-sa/3.0) (CC BY-NC-SA 3.0).
