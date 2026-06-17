import matplotlib.pyplot as plt
import numpy as np

DARK='#222222'; BLUE='#1177BB'; GREEN='#228811'; MONO='monospace'

def ffn_diagram():
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
    ax.set_facecolor('white'); ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)

    layers = [('x  [d]', 2.0, 4, DARK, '#DDEEFF'),
              ('GELU(x*W1+b1)  [4d]', 5.0, 8, GREEN, '#DDFFD4'),
              ('output  [d]', 8.0, 4, BLUE, '#DDEEFF')]

    node_ys = {}
    for lbl, cx, n, col, fc in layers:
        spacing = 3.5 / (n+1)
        ys = [0.75 + (i+1)*spacing for i in range(n)]
        node_ys[cx] = ys
        for y in ys:
            c = plt.Circle((cx, y), 0.18, color=col, zorder=3)
            ax.add_patch(c)
        ax.text(cx, 4.5, lbl, ha='center', fontsize=9,
                fontfamily=MONO, color=col)

    items = list(node_ys.items())
    for (x1, ys1), (x2, ys2) in [(items[0], items[1]), (items[1], items[2])]:
        for y1 in ys1[:3]:
            for y2 in ys2[:4]:
                ax.plot([x1, x2], [y1, y2], color='#AAAAAA', lw=0.5, zorder=1)

    ax.text(3.5, 0.3, 'W_1', ha='center', fontsize=11,
            fontfamily=MONO, color=GREEN, fontstyle='italic')
    ax.text(6.5, 0.3, 'W_2', ha='center', fontsize=11,
            fontfamily=MONO, color=BLUE, fontstyle='italic')

    plt.tight_layout()
    plt.savefig('images/ch09_ffn.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()

def ffn_memory():
    np.random.seed(7)
    data = (np.random.rand(6, 8) > 0.55).astype(float)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor='white')
    ax.imshow(data, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax.set_yticks(range(6))
    ax.set_yticklabels([f'fact {i}' for i in range(6)],
                       fontfamily=MONO, fontsize=10)
    ax.set_xticks(range(8))
    ax.set_xticklabels([f'd{i}' for i in range(8)],
                       fontfamily=MONO, fontsize=10)
    ax.set_title('FFN as associative memory (active neurons = blue)',
                 fontsize=11, fontfamily=MONO, color=DARK)
    plt.tight_layout()
    plt.savefig('images/ch09_memory.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    ffn_diagram()
    ffn_memory()
    print('ch06 done')
