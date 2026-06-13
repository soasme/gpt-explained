#!/usr/bin/env bash
# render_all.sh — render all chapter Manim scenes and copy PNGs to src/images/
# Usage: bash src/manim/render_all.sh [--quality ql|qm|qh]

set -euo pipefail
QUALITY=${1:---ql}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGES_DIR="$(dirname "$SCRIPT_DIR")/images"
mkdir -p "$IMAGES_DIR"

declare -A SCENES=(
  ["ch01_tokens.py"]="Ch01BPEMerges Ch01TokenizationPipeline"
  ["ch02_embeddings.py"]="Ch02EmbeddingLookup Ch02SemanticSpace"
  ["ch03_positional_encoding.py"]="Ch03PositionalEncoding Ch03SinWaves"
  ["ch04_attention.py"]="Ch04AttentionWeights Ch04QKVDiagram"
  ["ch05_multi_head_attention.py"]="Ch05MultiHeadAttention"
  ["ch06_ffn.py"]="Ch06FFN Ch06MemoryInterpretation"
  ["ch07_transformer_block.py"]="Ch07TransformerBlock Ch07ResidualStream"
  ["ch08_vocab_projection.py"]="Ch08VocabProjection Ch08GenerationLoop"
)

OUTPUT_NAMES=(
  "ch01-bpe-merges"
  "ch01-tokenization-pipeline"
  "ch02-embedding-lookup"
  "ch02-semantic-space"
  "ch03-positional-encoding"
  "ch03-sin-waves"
  "ch04-attention-weights"
  "ch04-qkv-diagram"
  "ch05-multi-head-attention"
  "ch06-ffn"
  "ch06-memory"
  "ch07-transformer-block"
  "ch07-residual-stream"
  "ch08-vocab-projection"
  "ch08-generation-loop"
)

echo "=== Rendering Manim scenes (quality: $QUALITY) ==="
cd "$SCRIPT_DIR"

for script in "${!SCENES[@]}"; do
  echo ""
  echo "--- $script ---"
  for scene in ${SCENES[$script]}; do
    echo "  Rendering $scene ..."
    manim "$QUALITY" --format=png -s "$script" "$scene" 2>&1 | grep -E "(Error|Warning|Rendered)" || true
  done
done

echo ""
echo "=== Copying PNGs to $IMAGES_DIR ==="

# Manim saves stills to: media/images/<script_stem>/<SceneName>_ManimCE_v*.png
find "$SCRIPT_DIR/media/images" -name "*.png" | while read -r f; do
  scene_name=$(basename "$(dirname "$f")")
  filename=$(basename "$f" | sed 's/_ManimCE_v[0-9.]*//')
  # Convert CamelCase scene name to kebab-case
  kebab=$(echo "$scene_name" | sed 's/\([A-Z]\)/-\1/g' | sed 's/^-//' | tr '[:upper:]' '[:lower:]')
  dest="$IMAGES_DIR/${kebab}.png"
  cp "$f" "$dest"
  echo "  $f → $dest"
done

echo ""
echo "Done. Images written to $IMAGES_DIR"
