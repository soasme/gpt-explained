import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DARK='#222222'; BLUE='#1177BB'; GREEN='#228811'; RED='#CC2222'; MONO='monospace'

A_vals = [[1,2],[3,4]]
B_vals = [[5,6],[7,8]]
C_vals = [[19,22],[43,50]]

def draw_matrix(ax, data, ox, oy, cell=0.7, color=DARK, highlight=None, hlcolor=RED):
    rows = len(data); cols = len(data[0])
    for i in range(rows):
        for j in range(cols):
            fc = hlcolor if (highlight and (i,j) in highlight) else 'white'
            rect = mpatches.FancyBboxPatch((ox+j*cell, oy-i*cell-cell), cell, cell,
                boxstyle='round,pad=0.02', linewidth=1.2, edgecolor=color, facecolor=fc)
            ax.add_patch(rect)
            ax.text(ox+j*cell+cell/2, oy-i*cell-cell/2, str(data[i][j]),
                    ha='center', va='center', fontfamily=MONO, fontsize=13, color=DARK)

def mat_multiply():
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
    ax.set_facecolor('white'); ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(-1, 4)

    draw_matrix(ax, A_vals, 0.5, 3.5, highlight={(0,0),(0,1)}, hlcolor='#DDEEFF', color=BLUE)
    ax.text(2.3, 1.5, 'A', ha='center', fontfamily=MONO, fontsize=14, color=BLUE)

    ax.text(2.0, 1.7, 'x', ha='center', fontfamily=MONO, fontsize=16, color=DARK)

    draw_matrix(ax, B_vals, 2.5, 3.5, highlight={(0,0),(1,0)}, hlcolor='#DDFFD4', color=GREEN)
    ax.text(3.7, 1.5, 'B', ha='center', fontfamily=MONO, fontsize=14, color=GREEN)

    ax.text(4.4, 1.7, '=', ha='center', fontfamily=MONO, fontsize=18, color=DARK)

    draw_matrix(ax, C_vals, 5.0, 3.5, highlight={(0,0)}, hlcolor='#FFE0D0', color=RED)
    ax.text(6.2, 1.5, 'C', ha='center', fontfamily=MONO, fontsize=14, color=RED)

    ax.text(5.0, -0.5,
            'C[0][0] = row 0 of A  dot  col 0 of B = 1x5 + 2x7 = 19',
            fontfamily=MONO, fontsize=10, color=DARK)

    ax.set_title('Matrix multiplication: C[i][j] = dot(row i of A, col j of B)',
                 fontfamily=MONO, fontsize=11, color=DARK)
    plt.tight_layout()
    plt.savefig('images/ch02_matrix_multiply.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    mat_multiply()
    print('ch02_matrix_multiply done')
