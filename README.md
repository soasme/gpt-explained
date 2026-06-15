# GPT Explained

> *A hands-on guide to transformer architecture, from tokens to text generation, with math, matrices, Python, and diagrams.*

**Read online:** https://www.soasme.com/gpt-explained/

## Contents

| Chapter | Topic |
|---------|-------|
| 1 | **Introduction** |
| 2 | **Notation and Definitions** |
| 3 | **Tokens** — Text to Numbers |
| 4 | **Embeddings** — Numbers to Meaning |
| 5 | **Positional Encoding** — Giving Order to Meaning |
| 6 | **Attention** — Tokens Talking to Each Other |
| 7 | **Multi-Head Attention** — Many Conversations at Once |
| 8 | **Feed-Forward Network** — The Model's Memory |
| 9 | **The Transformer Block** — Putting It Together |
| 10 | **Vocabulary Projection** — From Vectors to Words |
| 11 | **Loss** — How the Model Learns |
| 12 | **Training** — Teaching the Model |
| 13 | **Modern GPT** |
| A | **microGPT in Python** — Complete Runnable Code |

## Building Locally

### Prerequisites

```bash
# macOS
brew bundle

# Python 3 is required for the runnable book code.
python3 --version
```

The `Brewfile` installs Quarto and Graphviz. Quarto renders the Markdown book sources in `.qmd` files, and Graphviz renders the DOT diagrams embedded in the book.

The book code uses only the Python standard library. Matplotlib and Pillow are needed only when regenerating images from `src/matplotlib/`.

### Run code examples

The Python files in `src/python/` are the source of truth for code shown in the book. Markdown listings are refreshed from tagged regions before each Quarto render.

```bash
# Run the complete GPT demo
python3 src/python/inference.py

# Run the tiny BPE tokenizer demo
python3 src/python/micro_bpe.py

# Run every chapter's end-to-end code check
python3 src/python/run_book_code.py

# Same check through Make
make check
```

### Build the book

```bash
quarto render
```

or:

```bash
make book
```

Output files are written under `_book/`.

| File | Format |
|------|--------|
| `_book/index.html` | HTML book |
| `_book/gpt-explained.epub` | EPUB 3 |
| `_book/gpt-explained.pdf` | PDF |

### Generate diagrams

Plot, heatmap, matrix, and neural-network diagrams are generated with Matplotlib scripts. To regenerate:

```bash
python3 -m pip install matplotlib pillow
bash src/matplotlib/render_all.sh
```

DOT diagrams are embedded directly in `.qmd` files and rendered by Quarto through Graphviz.

## Python Source Layout

| File | Purpose |
|------|---------|
| `src/python/common.py` | Forward-pass building blocks |
| `src/python/train.py` | Loss functions and gradient descent |
| `src/python/inference.py` | Generation and end-to-end demo |
| `src/python/micro_bpe.py` | Chapter 3 byte-pair encoding tokenizer |
| `src/python/chapter_demos.py` | Runnable chapter demos used by the end-to-end check |
| `src/python/run_book_code.py` | End-to-end runner for all chapter code |

## CI / Deployment

Every push to `main`:

1. Installs Quarto, Graphviz, Python, and image-generation dependencies
2. Regenerates images
3. Runs the Python chapter code check
4. Builds the Quarto book
5. Creates a versioned GitHub Release with EPUB and PDF assets
6. Publishes the HTML book, PDF, and EPUB to GitHub Pages

## License

Book text and code: [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported](https://creativecommons.org/licenses/by-nc-sa/3.0) (CC BY-NC-SA 3.0).
