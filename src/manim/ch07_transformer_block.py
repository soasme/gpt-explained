"""
ch07_transformer_block.py
Chapter 7: Full transformer block wiring diagram
Generates: images/ch07-transformer-block.png
"""
from manim import *

BG     = "#FFFFFF"
DARK   = "#222222"
BLUE   = "#1177BB"
GREEN  = "#228811"
YELLOW = "#CC8800"
ORANGE = "#DD6600"
PURPLE = "#6633AA"
MONO   = "Monospace"


class Ch07TransformerBlock(Scene):
    def construct(self):
        self.camera.background_color = BG

        # Vertical layout: Input → LN → MHA → + → LN → FFN → + → Output
        box_w, box_h = 3.2, 0.65
        gap = 0.7

        def make_box(label, color, y):
            box = RoundedRectangle(corner_radius=0.12, width=box_w, height=box_h,
                                   color=color, fill_opacity=0.18)
            lbl = Text(label, font=MONO, font_size=20, color=color)
            lbl.move_to(box)
            g = VGroup(box, lbl)
            g.move_to(RIGHT * 0 + UP * y)
            return g

        components = [
            ("Input  x", GREEN,  3.0),
            ("LayerNorm₁", YELLOW, 2.2),
            ("Multi-Head\nAttention", BLUE,  1.1),
            ("⊕  residual", ORANGE, 0.2),
            ("LayerNorm₂", YELLOW,-0.65),
            ("Feed-Forward\nNetwork", PURPLE,-1.75),
            ("⊕  residual", ORANGE,-2.65),
            ("Output  x₂", GREEN, -3.4),
        ]

        boxes = []
        for label, color, y in components:
            b = make_box(label, color, y)
            boxes.append(b)

        self.play(*[Create(b) for b in boxes], run_time=1.5)
        self.wait(0.3)

        # Main arrows (straight down)
        for i in range(len(boxes) - 1):
            arr = Arrow(boxes[i].get_bottom(), boxes[i+1].get_top(),
                        color=DARK, buff=0.05, stroke_width=1.8)
            self.play(GrowArrow(arr), run_time=0.25)

        # Residual skip 1: from "Input" to first residual add (jump over LN+MHA)
        skip1_start = boxes[0].get_right() + RIGHT * 0.0
        skip1_end   = boxes[3].get_right()
        skip1 = CurvedArrow(
            skip1_start, skip1_end,
            angle=-TAU/6, color=ORANGE, stroke_width=2.5)
        skip1_lbl = Text("skip", font=MONO, font_size=14, color=ORANGE)
        skip1_lbl.next_to(skip1, RIGHT, buff=0.05)
        self.play(Create(skip1), FadeIn(skip1_lbl))

        # Residual skip 2: from first residual output to second residual add
        skip2_start = boxes[3].get_right()
        skip2_end   = boxes[6].get_right()
        skip2 = CurvedArrow(
            skip2_start, skip2_end,
            angle=-TAU/6, color=ORANGE, stroke_width=2.5)
        self.play(Create(skip2))
        self.wait(0.5)

        # Equations
        eq1 = Text("x₁ = x + MHA(LN₁(x))", font=MONO, font_size=20,
                   color=BLUE)
        eq2 = Text("x₂ = x₁ + FFN(LN₂(x₁))", font=MONO, font_size=20,
                   color=PURPLE)
        equations = VGroup(eq1, eq2).arrange(DOWN, buff=0.25)
        equations.to_edge(RIGHT, buff=0.5)

        self.play(Write(equations))
        self.wait(3.0)

class Ch07ResidualStream(Scene):
    """Residual stream concept: N blocks adding to a shared stream."""
    def construct(self):
        self.camera.background_color = BG

        # Horizontal stream
        stream = Line(LEFT * 5.5, RIGHT * 5.5, color=GREEN, stroke_width=3)
        stream.move_to(ORIGIN)
        stream_lbl = Text("stream  [T × d]", font=MONO, font_size=18, color=GREEN)
        stream_lbl.next_to(stream, DOWN, buff=0.35)
        self.play(Create(stream), FadeIn(stream_lbl))

        # Blocks that write to the stream
        block_xs = [-4.0, -1.5, 1.0, 3.5]
        block_labels = ["Block 1\n(MHA+FFN)", "Block 2\n(MHA+FFN)",
                        "Block N-1\n(MHA+FFN)", "Block N\n(MHA+FFN)"]
        colors = [YELLOW, ORANGE, BLUE, PURPLE]

        for x, lbl, col in zip(block_xs, block_labels, colors):
            # Block above stream
            bx = RoundedRectangle(corner_radius=0.12, width=1.8, height=0.9,
                                   color=col, fill_opacity=0.15)
            bt = Text(lbl, font=MONO, font_size=14, color=col)
            bt.move_to(bx)
            b = VGroup(bx, bt)
            b.move_to(RIGHT * x + UP * 1.3)

            # Read arrow (down)
            arr_read = Arrow(stream.point_from_proportion((x + 5.5) / 11.0),
                             b.get_bottom(), color=col, buff=0.05,
                             stroke_width=1.8)
            # Write arrow (back down, curved)
            arr_write = CurvedArrow(b.get_bottom() + RIGHT * 0.3,
                                    stream.point_from_proportion((x + 5.5) / 11.0)
                                    + RIGHT * 0.4,
                                    angle=TAU / 8, color=col, stroke_width=1.8)

            self.play(Create(b), GrowArrow(arr_read),
                      Create(arr_write), run_time=0.6)

        note = Text(
            "Each block reads from and writes to the same stream via residual adds",
            font=MONO, font_size=16, color=ORANGE)
        note.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(note))
        self.wait(3.0)
