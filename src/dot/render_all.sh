#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOK_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$BOOK_ROOT"

mkdir -p images
for f in src/dot/*.dot; do
    name=$(basename "$f" .dot)
    dot -Tpng -Gdpi=150 -o "images/${name}.png" "$f"
    echo "Rendered: images/${name}.png"
done
