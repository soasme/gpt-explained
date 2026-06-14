"""
ch04_attention.py
Chapter 4: Attention Weight Heatmap + QKV Diagram
Generates: images/ch04-attention-weights.png
"""
from manim import *
import numpy as np

BG     = "#FFFFFF"
DARK   = "#222222"
BLUE   = "#1177BB"
GREEN  = "#228811"
YELLOW = "#CC8800"
ORANGE = "#DD6600"
RED    = "#CC2222"
MONO   = "Monospace"

def softmax(v):
    e = np.exp(v - np.max(v))
    return e / e.sum()


class Ch04AttentionWeights(Scene):
    """Attention weight heatmap for a small example."""

    def construct(self):
        self.camera.background_color = BG

        # Example attention weights (T=5 tokens, causal)
        tokens = ["the", "cat", "sat", "on", "mat"]
        T = len(tokens)

        # Compute a plausible weight matrix
        raw = np.array([
            [2.0, -9, -9, -9, -9],
            [0.5, 1.5, -9, -9, -9],
            [1.0, 0.3, 1.8, -9, -9],
            [0.2, 0.8, 0.5, 1.2, -9],
            [0.3, 0.6, 1.4, 0.9, 2.1],
        ])
        A = np.array([softmax(raw[i]) for i in range(T)])

        cell_size = 0.72
        grid = VGroup()

        for i in range(T):
            for j in range(T):
                val = A[i, j]
                color = interpolate_color(ManimColor("#000000"), ManimColor(BLUE), min(val * 2.5, 1.0))
                sq = Square(side_length=cell_size,
                            fill_color=color, fill_opacity=0.95,
                            stroke_color=BG, stroke_width=0.5)
                sq.move_to(RIGHT * j * cell_size + DOWN * i * cell_size)

                # Value label
                lbl = Text(f"{val:.2f}", font=MONO, font_size=12, color=DARK)
                lbl.move_to(sq)
                grid.add(VGroup(sq, lbl))

        grid.center().shift(LEFT * 0.5 + DOWN * 0.3)

        # Row/col labels
        row_labels = VGroup()
        col_labels = VGroup()
        for idx, tok in enumerate(tokens):
            rl = Text(tok, font=MONO, font_size=16, color=YELLOW)
            rl.next_to(grid[idx * T], LEFT, buff=0.25)
            row_labels.add(rl)
            cl = Text(tok, font=MONO, font_size=16, color=YELLOW)
            cl.next_to(grid[idx], UP, buff=0.2)
            col_labels.add(cl)

        query_lbl = Text("query →", font=MONO, font_size=16, color=GREEN)
        query_lbl.rotate(PI / 2)
        query_lbl.next_to(grid, LEFT, buff=0.9)
        key_lbl = Text("← key", font=MONO, font_size=16, color=GREEN)
        key_lbl.next_to(grid, UP, buff=0.7)

        self.play(Create(grid), run_time=2.0)
        self.play(FadeIn(row_labels), FadeIn(col_labels),
                  FadeIn(query_lbl), FadeIn(key_lbl))
        self.wait(0.5)

        # Highlight: token 4 "mat" attends to token 2 "sat" most
        highlight = SurroundingRectangle(grid[4*T + 2], color=ORANGE, buff=0.05)
        note = Text('"mat" attends mostly\nto "sat" (1.40 raw score)',
                    font=MONO, font_size=16, color=ORANGE)
        note.to_edge(RIGHT, buff=0.4)
        note.shift(DOWN * 1.0)
        self.play(Create(highlight), Write(note))
        self.wait(2.5)

        # Show the masked region
        mask_box = Polygon(
            grid[0 * T + 1][0].get_corner(UL),
            grid[0 * T + (T-1)][0].get_corner(UR),
            grid[(T-2) * T + (T-1)][0].get_corner(DR),
            color=RED, stroke_width=2, fill_opacity=0.08, fill_color=RED)
        mask_note = Text("masked (future)", font=MONO, font_size=14, color=RED)
        mask_note.next_to(mask_box, RIGHT, buff=0.1)
        self.play(Create(mask_box), FadeIn(mask_note))
        self.wait(2.5)

class Ch04QKVDiagram(Scene):
    """Q, K, V projection diagram."""
    def construct(self):
        self.camera.background_color = BG

        # Input
        X_box = RoundedRectangle(corner_radius=0.15, width=2.0, height=1.0,
                                 color=GREEN, fill_opacity=0.2)
        X_lbl = Text("X\n[T × d]", font=MONO, font_size=22, color=GREEN)
        X_lbl.move_to(X_box)
        X = VGroup(X_box, X_lbl)
        X.move_to(UP * 1.5)

        # Three projections
        proj_data = [
            ("Wq", "Q\n[T × dk]", YELLOW, LEFT * 3.5 + DOWN * 0.5),
            ("Wk", "K\n[T × dk]", ORANGE, ORIGIN + DOWN * 0.5),
            ("Wv", "V\n[T × dv]", BLUE,   RIGHT * 3.5 + DOWN * 0.5),
        ]

        projections = []
        for w_lbl, out_lbl, col, pos in proj_data:
            w_box = RoundedRectangle(corner_radius=0.12, width=1.4, height=0.7,
                                     color=col, fill_opacity=0.15)
            w_text = Text(w_lbl, font=MONO, font_size=20, color=col)
            w_text.move_to(w_box)
            w = VGroup(w_box, w_text)
            w.move_to(pos)
            projections.append((w, col, out_lbl, pos))

        for w, col, out_lbl, pos in projections:
            arrow = Arrow(X.get_bottom(), w.get_top(), color=col,
                          buff=0.1, stroke_width=2)
            self.play(GrowArrow(arrow), Create(w), run_time=0.6)

        # Output boxes
        for w, col, out_lbl, pos in projections:
            out_box = RoundedRectangle(corner_radius=0.12, width=1.6, height=0.8,
                                       color=col, fill_opacity=0.2)
            out_text = Text(out_lbl, font=MONO, font_size=18, color=col)
            out_text.move_to(out_box)
            out = VGroup(out_box, out_text)
            out.next_to(w[0], DOWN, buff=0.5)
            arr = Arrow(w.get_bottom(), out.get_top(),
                        color=col, buff=0.05, stroke_width=2)
            self.play(GrowArrow(arr), Create(out), run_time=0.5)

        formula = Text("Attn(Q,K,V) = softmax( QKᵀ / √dk ) · V",
                       font=MONO, font_size=22, color=YELLOW)
        formula.to_edge(DOWN, buff=0.5)
        self.play(Write(formula))
        self.wait(3.0)
