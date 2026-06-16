import matplotlib.pyplot as plt

DARK='#222222'; BLUE='#1177BB'; GREEN='#228811'; RED='#CC2222'; MONO='monospace'

def vector_addition():
    fig, ax = plt.subplots(figsize=(5, 4), facecolor='white')
    ax.set_facecolor('white')
    ax.axhline(0, color=DARK, lw=0.5, alpha=0.3)
    ax.axvline(0, color=DARK, lw=0.5, alpha=0.3)
    ax.set_xlim(-0.5, 5); ax.set_ylim(-3, 5)
    ax.set_xticks(range(0, 6)); ax.set_yticks(range(-3, 6))
    ax.tick_params(labelsize=9)
    for sp in ['top','right']:
        ax.spines[sp].set_visible(False)

    ax.annotate('', xy=(3,4), xytext=(0,0),
        arrowprops=dict(arrowstyle='->', color=BLUE, lw=2.5))
    ax.text(2.15, 4.15, 'u = [3, 4]', fontsize=10, fontfamily=MONO, color=BLUE)

    ax.annotate('', xy=(1,-2), xytext=(0,0),
        arrowprops=dict(arrowstyle='->', color=GREEN, lw=2.5))
    ax.text(1.1, -2.15, 'v = [1, -2]', fontsize=10, fontfamily=MONO, color=GREEN)

    ax.annotate('', xy=(4,2), xytext=(0,0),
        arrowprops=dict(arrowstyle='->', color=RED, lw=2.5))
    ax.text(4.1, 2.1, 'u+v = [4, 2]', fontsize=10, fontfamily=MONO, color=RED)

    ax.plot([3, 4], [4, 2], color=GREEN, lw=1.2, linestyle='dashed', alpha=0.8)
    ax.plot([1, 4], [-2, 2], color=BLUE, lw=1.2, linestyle='dashed', alpha=0.8)

    plt.tight_layout()
    plt.savefig('images/ch02_vector_addition.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()

def l2_norm():
    fig, ax = plt.subplots(figsize=(5, 4), facecolor='white')
    ax.set_facecolor('white')
    ax.axhline(0, color=DARK, lw=0.5, alpha=0.3)
    ax.axvline(0, color=DARK, lw=0.5, alpha=0.3)
    ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.5, 5)
    ax.set_xticks(range(0, 5)); ax.set_yticks(range(0, 6))
    ax.tick_params(labelsize=9)
    for sp in ['top','right']:
        ax.spines[sp].set_visible(False)

    ax.annotate('', xy=(3, 4), xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color=BLUE, lw=2.5))
    ax.plot([3, 3], [0, 4], color='#777777', lw=1.2, linestyle='dashed')
    ax.plot([0, 3], [4, 4], color='#777777', lw=1.2, linestyle='dashed')

    ax.text(3.12, 4.08, '[3, 4]', fontsize=10, fontfamily=MONO, color=BLUE)
    ax.text(1.25, -0.35, 'x = 3', fontsize=10, fontfamily=MONO, color=DARK)
    ax.text(3.12, 1.9, 'y = 4', fontsize=10, fontfamily=MONO, color=DARK)
    ax.text(1.15, 2.0, '||[3, 4]|| = 5', fontsize=10, fontfamily=MONO,
            color=BLUE, rotation=53.13, rotation_mode='anchor')

    plt.tight_layout()
    plt.savefig('images/ch02_l2_norm.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    vector_addition()
    l2_norm()
    print('ch02_vectors done')
