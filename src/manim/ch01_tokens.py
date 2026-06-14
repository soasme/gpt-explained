"""
ch01_tokens.py
Chapter 1: Tokenization — BPE Merge Diagram
Generates: images/ch01-bpe-merges.png
"""
from manim import *

BG     = "#FFFFFF"
DARK   = "#222222"
BLUE   = "#1177BB"
GREEN  = "#228811"
YELLOW = "#CC8800"
ORANGE = "#DD6600"
RED    = "#CC2222"
MONO   = "Monospace"

class Ch01BPEMerges(Scene):
    def construct(self):
        self.camera.background_color = BG

        # --- Show the original word as characters ---
        chars = list("tokenization")
        char_boxes = VGroup()
        for i, c in enumerate(chars):
            box = VGroup(
                Square(side_length=0.55, color=BLUE, fill_opacity=0.15),
                Text(c, font=MONO, font_size=24, color=DARK)
            )
            char_boxes.add(box)

        char_boxes.arrange(RIGHT, buff=0.08)
        char_boxes.move_to(UP * 1.6)

        label_chars = Text("Step 0: characters", font=MONO, font_size=20, color=GREEN)
        label_chars.next_to(char_boxes, DOWN, buff=0.3)

        self.play(Create(char_boxes), run_time=1.5)
        self.play(FadeIn(label_chars))
        self.wait(1.0)

        # --- Step 1: merge "iz" -> "iz" ---
        def highlight_pair(boxes, i1, i2, color=YELLOW):
            return AnimationGroup(
                boxes[i1][0].animate.set_color(color).set_fill(color, opacity=0.35),
                boxes[i2][0].animate.set_color(color).set_fill(color, opacity=0.35),
            )

        # Highlight "iz" (indices 5,6 in "tokenization": t-o-k-e-n-i-z-a-t-i-o-n)
        merge_pairs = [(5, 6, "i+z→iz"), (3, 4, "e+n→en"), (0, 1, "t+o→to")]

        current_boxes = char_boxes.copy()
        for pair_i1, pair_i2, label_text in merge_pairs[:1]:
            self.play(highlight_pair(current_boxes, pair_i1, pair_i2))
            merge_label = Text(label_text, font=MONO, font_size=20, color=YELLOW)
            merge_label.next_to(current_boxes, DOWN, buff=0.6)
            self.play(Write(merge_label))
            self.wait(1.0)
            self.play(FadeOut(merge_label))

        self.wait(0.5)
        self.play(FadeOut(current_boxes), FadeOut(label_chars))

        # --- Show final tokenized form ---
        tokens = ["token", "ization"]
        token_boxes = VGroup()
        colors_for_tokens = [GREEN, ORANGE]
        for tok, col in zip(tokens, colors_for_tokens):
            box = VGroup(
                RoundedRectangle(corner_radius=0.1, width=len(tok)*0.35+0.4,
                                 height=0.65, color=col, fill_opacity=0.2,
                                 stroke_width=2),
                Text(tok, font=MONO, font_size=26, color=col)
            )
            token_boxes.add(box)
        token_boxes.arrange(RIGHT, buff=0.25)
        token_boxes.move_to(ORIGIN)

        label_final = Text("After BPE: 2 tokens", font=MONO, font_size=20, color=GREEN)
        label_final.next_to(token_boxes, DOWN, buff=0.4)

        id_labels = VGroup()
        for i, box in enumerate(token_boxes):
            lbl = Text(f"ID {i}", font=MONO, font_size=18, color=YELLOW)
            lbl.next_to(box, UP, buff=0.2)
            id_labels.add(lbl)

        self.play(Create(token_boxes), run_time=1.5)
        self.play(FadeIn(label_final), FadeIn(id_labels))
        self.wait(2.0)

        # --- Vocabulary size note ---
        note = Text("Vocabulary: 50,257 entries (GPT-2)", font=MONO,
                    font_size=18, color=BLUE)
        note.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(note))
        self.wait(2.0)

class Ch01TokenizationPipeline(Scene):
    """Shows the full text-to-IDs pipeline."""
    def construct(self):
        self.camera.background_color = BG

        # Source text
        src_text = Text('"the cat sat on the mat"', font=MONO,
                        font_size=28, color=DARK)
        src_text.move_to(UP * 1.5)

        # Arrow down
        arrow = Arrow(src_text.get_bottom(), src_text.get_bottom() + DOWN*1.2,
                      color=YELLOW, buff=0.1)

        # Token IDs
        ids_text = Text("[1, 5, 12, 8, 1, 7]", font=MONO,
                        font_size=28, color=GREEN)
        ids_text.next_to(arrow, DOWN, buff=0.2)

        label_tokenizer = Text("tokenizer", font=MONO, font_size=20,
                               color=YELLOW, slant=ITALIC)
        label_tokenizer.next_to(arrow, RIGHT, buff=0.2)

        self.play(Write(src_text))
        self.wait(0.5)
        self.play(GrowArrow(arrow), FadeIn(label_tokenizer))
        self.play(Write(ids_text))
        self.wait(2.0)

        # Highlight that "the" appears twice with same ID
        brace1 = Brace(src_text[1:4], UP, color=ORANGE)
        brace2 = Brace(src_text[16:19], UP, color=ORANGE)
        same_id_note = Text("same ID = 1", font=MONO, font_size=18, color=ORANGE)
        same_id_note.next_to(brace1, UP, buff=0.1)

        self.play(Create(brace1), Create(brace2))
        self.play(Write(same_id_note))
        self.wait(2.0)
