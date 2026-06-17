import matplotlib.pyplot as plt
import numpy as np

DARK='#222222'; BLUE='#1177BB'; GREEN='#228811'; MONO='monospace'

def loss_diagram():
    probs = np.array([0.05, 0.08, 0.62, 0.12, 0.07, 0.06])
    true_idx = 2
    colors = [GREEN if i == true_idx else BLUE for i in range(len(probs))]

    fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
    ax.set_facecolor('white')
    ax.bar(range(len(probs)), probs, color=colors, edgecolor=DARK, lw=0.8)
    ax.set_xticks(range(len(probs)))
    ax.set_xticklabels([f'tok {i}' for i in range(len(probs))],
                       fontfamily=MONO, fontsize=10)
    ax.set_ylabel('P(token)', fontfamily=MONO, fontsize=10, color=DARK)
    ax.set_ylim(0, 0.85)
    ax.set_title('Cross-entropy loss: L = -log( P(true token) )',
                 fontfamily=MONO, fontsize=11, color=DARK)

    p = probs[true_idx]
    loss = -np.log(p)
    ax.annotate(f'true token (tok {true_idx})\np = {p:.2f}\nloss = -log({p:.2f}) = {loss:.2f}',
                xy=(true_idx, p), xytext=(true_idx+1.2, p+0.15),
                fontsize=9, fontfamily=MONO, color=GREEN,
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5))

    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    plt.savefig('images/ch12_loss.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    loss_diagram()
    print('ch09 done')
