#!/usr/bin/env bash
# Render all Matplotlib images in one Python process.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOK_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$BOOK_ROOT"

python3 src/matplotlib/render_all.py
