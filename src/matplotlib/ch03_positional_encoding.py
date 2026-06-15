import matplotlib.pyplot as plt
import numpy as np

DARK='#222222'; BLUE='#1177BB'; MONO='monospace'

def pe_matrix():
    T, d = 12, 16
    PE = np.zeros((T, d))
    for pos in range(T):
        for i in range(0, d, 2):
            w = 1.0 / (10000 ** (i / d))
            PE[pos, i]   = np.sin(pos * w)
            if i+1 < d:
                PE[pos, i+1] = np.cos(pos * w)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor='white')
    im = ax.imshow(PE, aspect='auto', cmap='RdYlBu', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_xlabel('dimension', fontsize=11, fontfamily=MONO, color=DARK)
    ax.set_ylabel('position', fontsize=11, fontfamily=MONO, color=DARK)
    ax.set_title('Sinusoidal PE matrix (T=12, d=16)', fontsize=12,
                 fontfamily=MONO, color=DARK)
    plt.tight_layout()
    plt.savefig('images/ch03_pe_matrix.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()

def sin_waves():
    positions = np.arange(50)
    fig, ax = plt.subplots(figsize=(9, 4), facecolor='white')
    ax.set_facecolor('white')
    colors = [BLUE, '#228811', '#CC2222']
    for k, i in enumerate([0, 2, 4]):
        w = 1.0 / (10000 ** (i / 16))
        ax.plot(positions, np.sin(positions * w), color=colors[k],
                lw=2, label=f'dim {i}')
    ax.axhline(0, color=DARK, lw=0.5, alpha=0.4)
    ax.set_xlabel('position', fontsize=11, fontfamily=MONO, color=DARK)
    ax.set_ylabel('sin value', fontsize=11, fontfamily=MONO, color=DARK)
    ax.set_title('Sinusoidal waves at different frequencies',
                 fontsize=12, fontfamily=MONO, color=DARK)
    ax.legend(prop={'family': MONO, 'size': 10})
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    plt.savefig('images/ch03_sin_waves.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    pe_matrix()
    sin_waves()
    print('ch03 done')
