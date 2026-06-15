import matplotlib.pyplot as plt
import numpy as np

DARK='#222222'; BLUE='#1177BB'; GREEN='#228811'; RED='#CC2222'; MONO='monospace'

def rope_plot():
    fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
    ax.set_facecolor('white')
    ax.axhline(0, color=DARK, lw=0.5, alpha=0.3)
    ax.axvline(0, color=DARK, lw=0.5, alpha=0.3)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)

    circle = plt.Circle((0,0), 1.0, color=DARK, fill=False, lw=1.0, alpha=0.3, linestyle='--')
    ax.add_patch(circle)

    theta_m = np.radians(50)
    theta_n = np.radians(20)
    qm = np.array([np.cos(theta_m), np.sin(theta_m)])
    kn = np.array([np.cos(theta_n), np.sin(theta_n)])

    ax.annotate('', xy=qm, xytext=(0,0),
        arrowprops=dict(arrowstyle='->', color=BLUE, lw=2.5))
    ax.text(qm[0]+0.07, qm[1]+0.05, 'R(m)*q  (pos m)', fontfamily=MONO, fontsize=10, color=BLUE)

    ax.annotate('', xy=kn, xytext=(0,0),
        arrowprops=dict(arrowstyle='->', color=GREEN, lw=2.5))
    ax.text(kn[0]+0.07, kn[1]-0.08, 'R(n)*k  (pos n)', fontfamily=MONO, fontsize=10, color=GREEN)

    arc_theta = np.linspace(theta_n, theta_m, 80)
    ax.plot(0.35*np.cos(arc_theta), 0.35*np.sin(arc_theta), color=RED, lw=2.0)
    mid = (theta_m + theta_n) / 2
    ax.text(0.42*np.cos(mid)-0.05, 0.42*np.sin(mid)+0.02,
            'n-m', fontfamily=MONO, fontsize=11, color=RED)

    ax.set_title('RoPE: dot product depends on relative position (n - m)',
                 fontfamily=MONO, fontsize=11, color=DARK)
    plt.tight_layout()
    plt.savefig('images/ch13_rope.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    rope_plot()
    print('ch13_rope done')
