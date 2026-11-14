"""Pipeline diagram for the report."""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(7.2, 1.9))
ax.set_xlim(0, 100); ax.set_ylim(0, 26); ax.axis('off')

def box(x, y, w, h, text, colour):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1.2', fc=colour, ec='#444', lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=6.2, linespacing=1.15)

def arrow(x0, y0, x1, y1):
    ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='-|>', lw=0.8, color='#333'))

box(0.5, 9, 12, 8, 'GONG H-alpha\nimage\n2048 x 2048', '#eef1f5')
box(18, 15.5, 22, 8.5, 'ResNet34 U-Net, 512 px crops\n12 models, flip TTA', '#dbe8f7')
box(18, 2, 22, 8.5, 'ConvNeXt-T U-Net, whole disk\n1536 px, 3 models, flip TTA', '#dbe8f7')
box(45, 9, 12, 8, 'average of the\ntwo groups\n(probability map)', '#e8f3e0')
box(62, 9, 15, 8, 'candidates cut at\n0.3 to 0.8 (6 levels)\nforest of masks', '#fdf0d5')
box(81.5, 9, 17, 8, 'LightGBM IoU score,\nexact best antichain\n(kept = disjoint masks)', '#fdf0d5')
arrow(13.2, 14.5, 17.6, 19.5); arrow(13.2, 11.5, 17.6, 6.5)
arrow(40.6, 19.5, 44.6, 14.5); arrow(40.6, 6.5, 44.6, 11.5)
arrow(57.6, 13, 61.6, 13); arrow(77.6, 13, 81.1, 13)
fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
fig.savefig(sys.argv[1], dpi=200)
