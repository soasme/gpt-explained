import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DARK='#222222'; BLUE='#1177BB'; GREEN='#228811'
YELLOW='#CC8800'; RED='#CC2222'; MONO='monospace'

def qkv_diagram():
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
    ax.set_facecolor('white'); ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)

    T = 4
    cols = [('Q', 2.5, BLUE, 'Q = X * Wq'),
            ('K', 5.0, GREEN, 'K = X * Wk'),
            ('V', 7.5, RED,   'V = X * Wv')]

    for t in range(T):
        r = mpatches.FancyBboxPatch((0.3, 3.5 - t*0.7), 1.5, 0.55,
            boxstyle='round,pad=0.04', lw=1.2, ec=DARK, fc='#EEEEEE')
        ax.add_patch(r)
        ax.text(1.05, 3.775 - t*0.7, f't{t}', ha='center', va='center',
                fontsize=10, fontfamily=MONO, color=DARK)
    ax.text(1.05, 4.2, 'X (input)', ha='center', fontsize=10,
            fontfamily=MONO, color=DARK)

    for name, cx, col, lbl in cols:
        ax.text(cx, 4.2, lbl, ha='center', fontsize=9, fontfamily=MONO, color=col)
        for t in range(T):
            r = mpatches.FancyBboxPatch((cx-0.65, 3.5-t*0.7), 1.3, 0.55,
                boxstyle='round,pad=0.04', lw=1.5, ec=col, fc='white')
            ax.add_patch(r)
            ax.text(cx, 3.775-t*0.7, f'{name}[t{t}]', ha='center', va='center',
                    fontsize=9, fontfamily=MONO, color=col)
        ax.annotate('', xy=(cx-0.65, 3.775), xytext=(1.8, 3.775),
            arrowprops=dict(arrowstyle='->', color=col, lw=1.5))

    ax.text(5.0, 0.4, 'score = softmax( Q * K^T / sqrt(d) ) * V',
            ha='center', fontsize=11, fontfamily=MONO, color=DARK)

    plt.tight_layout()
    plt.savefig('images/ch04_qkv.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()

def attn_weights():
    np.random.seed(42)
    T = 4
    raw = np.random.randn(T, T).astype(float)
    mask = np.triu(np.ones((T, T)), k=1)
    raw[mask == 1] = -1e9
    e = np.exp(raw - raw.max(axis=1, keepdims=True))
    W = e / e.sum(axis=1, keepdims=True)
    W[mask == 1] = 0.0

    fig, ax = plt.subplots(figsize=(5, 4), facecolor='white')
    im = ax.imshow(W, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)
    tks = [f't{i}' for i in range(T)]
    ax.set_xticks(range(T)); ax.set_xticklabels(tks, fontfamily=MONO, fontsize=10)
    ax.set_yticks(range(T)); ax.set_yticklabels(tks, fontfamily=MONO, fontsize=10)
    ax.set_xlabel('key position', fontsize=10, fontfamily=MONO, color=DARK)
    ax.set_ylabel('query position', fontsize=10, fontfamily=MONO, color=DARK)
    ax.set_title('Attention weights (causal masked)', fontsize=11,
                 fontfamily=MONO, color=DARK)
    for i in range(T):
        for j in range(T):
            if mask[i,j] == 0:
                ax.text(j, i, f'{W[i,j]:.2f}', ha='center', va='center',
                        fontsize=9, fontfamily=MONO,
                        color='white' if W[i,j] > 0.6 else DARK)
    plt.tight_layout()
    plt.savefig('images/ch04_attn_weights.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    qkv_diagram()
    attn_weights()
    print('ch04 done')
