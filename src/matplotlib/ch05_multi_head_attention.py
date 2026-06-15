import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DARK='#222222'; MONO='monospace'
HEAD_COLORS = ['#1177BB', '#228811', '#CC8800', '#CC2222']

def mha():
    fig, ax = plt.subplots(figsize=(11, 5), facecolor='white')
    ax.set_facecolor('white'); ax.axis('off')
    ax.set_xlim(0, 11); ax.set_ylim(0, 5)

    r = mpatches.FancyBboxPatch((0.5, 4.1), 10, 0.5,
        boxstyle='round,pad=0.05', lw=1.5, ec=DARK, fc='#EEEEEE')
    ax.add_patch(r)
    ax.text(5.5, 4.35, 'Input X', ha='center', va='center',
            fontsize=12, fontfamily=MONO, color=DARK)

    H = 4
    box_w = 1.6; gap = 0.5; start_x = 1.0
    for h in range(H):
        col = HEAD_COLORS[h]
        cx = start_x + h*(box_w + gap)
        ax.annotate('', xy=(cx+box_w/2, 3.5), xytext=(cx+box_w/2, 4.1),
            arrowprops=dict(arrowstyle='->', color=col, lw=1.5))
        r = mpatches.FancyBboxPatch((cx, 2.3), box_w, 1.1,
            boxstyle='round,pad=0.06', lw=2, ec=col, fc='white')
        ax.add_patch(r)
        ax.text(cx+box_w/2, 3.25, f'Head {h}', ha='center', va='center',
                fontsize=10, fontfamily=MONO, color=col, fontweight='bold')
        ax.text(cx+box_w/2, 2.75, f'Q{h} K{h} V{h}', ha='center', va='center',
                fontsize=9, fontfamily=MONO, color=col)
        ax.annotate('', xy=(cx+box_w/2, 1.8), xytext=(cx+box_w/2, 2.3),
            arrowprops=dict(arrowstyle='->', color=col, lw=1.5))

    r = mpatches.FancyBboxPatch((0.8, 0.8), 9.4, 0.9,
        boxstyle='round,pad=0.06', lw=2, ec=DARK, fc='#EEF4FF')
    ax.add_patch(r)
    ax.text(5.5, 1.25, 'Concat heads  +  project (W_o)  ->  output',
            ha='center', va='center', fontsize=11, fontfamily=MONO, color=DARK)

    plt.tight_layout()
    plt.savefig('images/ch05_mha.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    mha()
    print('ch05 done')
