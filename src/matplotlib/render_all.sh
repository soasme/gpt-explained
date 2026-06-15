#!/usr/bin/env bash
# Render all matplotlib diagram scripts to images/
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOK_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$BOOK_ROOT"

for py in src/matplotlib/ch*.py; do
  echo "Running $py ..."
  python3 "$py"
done

echo ""
echo "Done! PNGs in images/:"
ls images/*.png 2>/dev/null | wc -l
