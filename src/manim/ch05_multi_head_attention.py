"""
ch05_multi_head_attention.py
Chapter 5: Multi-Head Attention parallel heads diagram
Generates: images/ch05-multi-head-attention.png
"""
from manim import *

BG     = "#FFFFFF"
BLUE   = "#1177BB"
GREEN  = "#228811"
YELLOW = "#CC8800"
ORANGE = "#DD6600"
PURPLE = "#6633AA"
MONO   = "Monospace"


class Ch05MultiHeadAttention(Scene):
    def construct(self):
        self.camera.background_color = BG

        # Input X
        x_box = RoundedRectangle(corner_radius=0.15, width=2.5, height=0.75,
                                  color=GREEN, fill_opacity=0.18)
        x_lbl = Text("X  [T × d]", font=MONO, font_size=22, color=GREEN)
        x_lbl.move_to(x_box)
        X = VGroup(x_box, x_lbl)
        X.move_to(UP * 2.5)
        self.play(Create(X))
        self.wait(0.3)

        # H attention heads
        H = 4
        head_colors = [YELLOW, ORANGE, BLUE, PURPLE]
        head_groups = []

        for h in range(H):
            x_pos = (h - (H-1)/2) * 2.6
            col = head_colors[h]

            # Head box
            hbox = RoundedRectangle(corner_radius=0.12, width=2.2, height=0.9,
                                     color=col, fill_opacity=0.15)
            hlbl = Text(f"Head {h+1}\nAttn(Q{h+1},K{h+1},V{h+1})",
                        font=MONO, font_size=14, color=col)
            hlbl.move_to(hbox)
            head = VGroup(hbox, hlbl)
            head.move_to(RIGHT * x_pos + UP * 0.6)

            # Arrow from X to head
            arr_in = Arrow(X.get_bottom(), head.get_top(), color=col,
                           buff=0.08, stroke_width=2)

            # Output arrow
            arr_out = Arrow(head.get_bottom(),
                            head.get_bottom() + DOWN * 0.85,
                            color=col, buff=0.05, stroke_width=2)

            head_groups.append((head, arr_in, arr_out, col, x_pos))
            self.play(GrowArrow(arr_in), Create(head), run_time=0.5)

        self.wait(0.3)

        # Draw output arrows
        out_positions = []
        for head, arr_in, arr_out, col, x_pos in head_groups:
            self.play(GrowArrow(arr_out), run_time=0.3)
            out_positions.append(arr_out.get_end())

        # Concat box
        concat_box = RoundedRectangle(corner_radius=0.12, width=5.5, height=0.75,
                                       color=BLUE, fill_opacity=0.15)
        concat_lbl = Text("Concat(head1, …, head4)  [T × d]",
                          font=MONO, font_size=18, color=BLUE)
        concat_lbl.move_to(concat_box)
        concat = VGroup(concat_box, concat_lbl)
        concat.move_to(DOWN * 1.1)

        # Convergence arrows
        for pos in out_positions:
            arr = Arrow(pos, concat.get_top() + RIGHT * (pos[0] * 0.15),
                        color=BLUE, buff=0.05, stroke_width=1.5)
            self.play(GrowArrow(arr), run_time=0.3)

        self.play(Create(concat))
        self.wait(0.3)

        # Output projection
        wo_arrow = Arrow(concat.get_bottom(), concat.get_bottom() + DOWN * 0.9,
                         color=GREEN, buff=0.05, stroke_width=2)
        wo_lbl = Text("× Wo", font=MONO, font_size=18, color=YELLOW)
        wo_lbl.next_to(wo_arrow, RIGHT, buff=0.1)

        out_box = RoundedRectangle(corner_radius=0.12, width=3.5, height=0.7,
                                    color=GREEN, fill_opacity=0.18)
        out_lbl = Text("Output  [T × d]", font=MONO, font_size=20, color=GREEN)
        out_lbl.move_to(out_box)
        output = VGroup(out_box, out_lbl)
        output.next_to(wo_arrow, DOWN, buff=0.05)

        self.play(GrowArrow(wo_arrow), FadeIn(wo_lbl))
        self.play(Create(output))
        self.wait(0.5)

        # Note: each head uses d/H dimensions
        note = Text("Each head: dk = dv = d/H  →  same total cost as one big head",
                    font=MONO, font_size=15, color=ORANGE)
        note.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(note))
        self.wait(3.0)
