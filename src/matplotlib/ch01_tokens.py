import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DARK='#222222'; BLUE='#1177BB'; GREEN='#228811'
YELLOW='#CC8800'; ORANGE='#DD6600'; MONO='monospace'

def bpe_merges():
    fig, ax = plt.subplots(figsize=(13, 4), facecolor='white')
    ax.set_facecolor('white'); ax.axis('off')
    ax.set_xlim(0, 13); ax.set_ylim(0, 4)

    chars = list("tokenization")
    for i, c in enumerate(chars):
        r = mpatches.FancyBboxPatch((i*0.95+0.1, 2.5), 0.75, 0.7,
            boxstyle='round,pad=0.05', lw=1.5, ec=BLUE, fc='#DDEEFF')
        ax.add_patch(r)
        ax.text(i*0.95+0.475, 2.85, c, ha='center', va='center',
                fontsize=12, fontfamily=MONO, color=DARK)
    ax.text(6, 3.5, 'Step 0: individual characters', ha='center',
            fontsize=11, fontfamily=MONO, color=GREEN)

    ax.annotate('', xy=(6, 1.8), xytext=(6, 2.4),
        arrowprops=dict(arrowstyle='->', color=YELLOW, lw=2))
    ax.text(7.2, 2.1, 'BPE merges', fontsize=10, fontfamily=MONO, color=YELLOW)

    tokens = [('token', GREEN, 2.5, 1.5), ('ization', ORANGE, 6.0, 1.5)]
    for tok, col, x, y in tokens:
        w = len(tok)*0.45 + 0.4
        r = mpatches.FancyBboxPatch((x, y), w, 0.65,
            boxstyle='round,pad=0.08', lw=2, ec=col, fc='white')
        ax.add_patch(r)
        ax.text(x+w/2, y+0.325, tok, ha='center', va='center',
                fontsize=13, fontfamily=MONO, color=col, fontweight='bold')

    ax.text(6, 0.9, 'After BPE: 2 tokens   |   Vocabulary: 50,257 entries (GPT-2)',
            ha='center', fontsize=10, fontfamily=MONO, color=BLUE)

    plt.tight_layout()
    plt.savefig('images/ch01_bpe.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def tokenization_pipeline():
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
    ax.set_facecolor('white'); ax.axis('off')
    ax.set_xlim(0, 8); ax.set_ylim(0, 4)

    ax.text(4, 3.2, '"the cat sat on the mat"', ha='center', va='center',
            fontsize=14, fontfamily=MONO, color=DARK)
    ax.annotate('', xy=(4, 2.1), xytext=(4, 2.9),
        arrowprops=dict(arrowstyle='->', color=YELLOW, lw=2.5))
    ax.text(5.0, 2.5, 'tokenizer', fontsize=10, fontfamily=MONO, color=YELLOW)
    ax.text(4, 1.6, '[1,  5,  12,  8,  1,  7]', ha='center', va='center',
            fontsize=14, fontfamily=MONO, color=GREEN)
    ax.text(4, 0.9, '"the" appears twice -> same ID = 1',
            ha='center', fontsize=10, fontfamily=MONO, color=ORANGE)

    plt.tight_layout()
    plt.savefig('images/ch01_pipeline.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    bpe_merges()
    tokenization_pipeline()
    print('ch01 done')
