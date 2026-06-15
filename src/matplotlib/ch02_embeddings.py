import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DARK='#222222'; BLUE='#1177BB'; GREEN='#228811'
YELLOW='#CC8800'; ORANGE='#DD6600'; MONO='monospace'

def embedding_lookup():
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    ax.set_facecolor('white'); ax.axis('off')
    ax.set_xlim(-1, 13); ax.set_ylim(-0.5, 7)

    V, D = 6, 4
    cw, ch_ = 1.0, 0.7
    highlights = {2: ORANGE, 4: GREEN, 0: YELLOW}

    for i in range(V):
        col = highlights.get(i, BLUE)
        alpha = 0.5 if i in highlights else 0.12
        for j in range(D):
            r = mpatches.Rectangle((j*cw+1, (V-1-i)*ch_), cw-0.05, ch_-0.05,
                lw=1.2, ec=col, fc=col, alpha=alpha)
            ax.add_patch(r)
        ax.text(0.8, (V-1-i)*ch_+ch_/2, f'tok {i}', ha='right', va='center',
                fontsize=10, fontfamily=MONO, color=DARK)

    ax.text(3, V*ch_+0.3, 'E  [6x4]', ha='center', fontsize=13,
            fontfamily=MONO, color=BLUE, fontweight='bold')

    ids = [2, 4, 0]
    out_colors = [ORANGE, GREEN, YELLOW]
    for k, (tok_id, col) in enumerate(zip(ids, out_colors)):
        y_src = (V-1-tok_id)*ch_ + ch_/2
        y_dst = 4.5 - k*1.2
        ax.annotate('', xy=(8.5, y_dst), xytext=(5.1, y_src),
            arrowprops=dict(arrowstyle='->', color=col, lw=1.8))
        ax.text(9.0, y_dst, f'x[{k}] = E[{tok_id}]  [ ... ]',
                va='center', fontsize=10, fontfamily=MONO, color=col)

    ax.text(1, -0.3, f'Input IDs: {ids}', fontsize=11, fontfamily=MONO, color=GREEN)

    plt.tight_layout()
    plt.savefig('images/ch02_embedding_lookup.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()

def semantic_space():
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    ax.set_facecolor('white')
    ax.set_xlim(-3, 3); ax.set_ylim(-2, 3)
    ax.axhline(0, color=DARK, lw=0.5, alpha=0.3)
    ax.axvline(0, color=DARK, lw=0.5, alpha=0.3)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

    pts = {'king': (2.0, 1.5, YELLOW), 'queen': (2.0, -0.5, ORANGE),
           'man': (-1.5, 1.5, GREEN), 'woman': (-1.5, -0.5, BLUE)}
    for word, (x, y, col) in pts.items():
        ax.scatter(x, y, s=120, color=col, zorder=3)
        ax.text(x+0.15, y+0.15, word, fontsize=13, fontfamily=MONO, color=col)

    ax.annotate('', xy=(2.0, -0.5), xytext=(2.0, 1.5),
        arrowprops=dict(arrowstyle='->', color=YELLOW, lw=2.5))
    ax.text(-2.5, -1.5, 'king - man + woman  =~  queen',
            fontsize=12, fontfamily=MONO, color=YELLOW)

    plt.tight_layout()
    plt.savefig('images/ch02_semantic_space.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    embedding_lookup()
    semantic_space()
    print('ch02 done')
