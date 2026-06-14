#!/usr/bin/env bash
# Render all Manim scenes to static PNGs in images/
# Usage: bash src/manim/render_all.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOK_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMG_DIR="$BOOK_ROOT/images"
MEDIA_DIR="/tmp/manim_render"

mkdir -p "$IMG_DIR"

trim_vertical_whitespace() {
  local file="$1"
  python3 - "$file" <<'PY'
import sys
from PIL import Image, ImageChops

path = sys.argv[1]
img = Image.open(path).convert("RGBA")
bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
diff = ImageChops.difference(img, bg).convert("L")
bbox = diff.getbbox()

if bbox is None:
    img.save(path)
    raise SystemExit

left, top, right, bottom = bbox
pad = 24
top = max(0, top - pad)
bottom = min(img.height, bottom + pad)
img.crop((0, top, img.width, bottom)).save(path)
PY
}

run_scene() {
  local py_file="$1"
  local scene_name="$2"
  local out_name="$3"
  local scene_media="$MEDIA_DIR/$scene_name"

  echo "Rendering $scene_name → $out_name.png"
  rm -rf "$scene_media"
  manim "$SCRIPT_DIR/$py_file" "$scene_name" \
    --format png --save_last_frame \
    --media_dir "$scene_media" \
    --quality h \
    -v WARNING || { echo "WARN: $scene_name failed, skipping"; return 0; }

  local png
  png=$(find "$scene_media" -name "*.png" 2>/dev/null | tail -1)
  if [[ -f "$png" ]]; then
    cp "$png" "$IMG_DIR/${out_name}.png"
    trim_vertical_whitespace "$IMG_DIR/${out_name}.png"
    echo "  → copied to images/${out_name}.png"
  else
    echo "  WARN: could not find output PNG for $scene_name"
  fi
}

run_scene ch01_tokens.py          Ch01BPEMerges            ch01_bpe
run_scene ch01_tokens.py          Ch01TokenizationPipeline ch01_pipeline
run_scene ch02_embeddings.py      Ch02EmbeddingLookup      ch02_embedding_lookup
run_scene ch02_embeddings.py      Ch02SemanticSpace        ch02_semantic_space
run_scene ch03_positional_encoding.py Ch03PositionalEncoding ch03_pe_matrix
run_scene ch03_positional_encoding.py Ch03SinWaves          ch03_sin_waves
run_scene ch04_attention.py       Ch04QKVDiagram           ch04_qkv
run_scene ch04_attention.py       Ch04AttentionWeights     ch04_attn_weights
run_scene ch05_multi_head_attention.py Ch05MultiHeadAttention ch05_mha
run_scene ch06_ffn.py             Ch06FFN                  ch06_ffn
run_scene ch06_ffn.py             Ch06MemoryInterpretation ch06_memory
run_scene ch07_transformer_block.py Ch07TransformerBlock   ch07_block
run_scene ch07_transformer_block.py Ch07ResidualStream     ch07_residual
run_scene ch08_vocab_projection.py Ch08VocabProjection     ch08_projection
run_scene ch08_vocab_projection.py Ch08GenerationLoop      ch08_generation
run_scene ch09_loss.py             Ch09LossScene           ch09_loss
run_scene ch10_training.py         Ch10TrainingDiagram     ch10_training

echo ""
echo "Done! Images in $IMG_DIR:"
ls "$IMG_DIR"/*.png 2>/dev/null | wc -l
echo "PNGs generated."
