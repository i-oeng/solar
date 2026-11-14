"""Report figures: predicted instances of the submitted CSV on three test images
(inference only, no labels) and the three-panel pipeline diagram."""
import csv
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as coco_mask

from solar_filament.data import find_data_root, read_grayscale

csv_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
maps_dir = Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)
data = find_data_root('data/kagglehub/competitions/filament-segmentation-2026')
rows = {}
for r in csv.DictReader(csv_path.open()):
    stem = r['filament_id'].rsplit('_', 1)[0]
    rows.setdefault(stem, []).append(r['segmentation_rle'])
stems = sorted(rows)
picks = [stems[len(stems) // 6], stems[len(stems) // 2], stems[5 * len(stems) // 6]]
rng = np.random.default_rng(0)
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9))
for col, stem in enumerate(picks):
    image = read_grayscale(data.test_dir / f'{stem}.jpeg')
    prob = cv2.imread(str(maps_dir / f'{stem}.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535
    overlay = np.stack([image] * 3, -1).astype(np.float32) / 255
    for rle in rows[stem]:
        m = coco_mask.decode({'size': list(image.shape), 'counts': rle.encode()}).astype(bool)
        colour = rng.uniform(0.35, 1.0, 3)
        overlay[m] = 0.35 * overlay[m] + 0.65 * colour
    axes[0, col].imshow(prob, cmap='magma', vmin=0, vmax=1)
    axes[1, col].imshow(overlay)
    axes[0, col].set_title(f'{stem[:8]} {stem[-2:]}', fontsize=7)
    for row in (0, 1):
        axes[row, col].axis('off')
axes[0, 0].text(-0.02, 0.5, 'blended probability', transform=axes[0, 0].transAxes, rotation=90,
                ha='right', va='center', fontsize=7)
axes[1, 0].text(-0.02, 0.5, 'selected instances', transform=axes[1, 0].transAxes, rotation=90,
                ha='right', va='center', fontsize=7)
fig.subplots_adjust(left=0.04, right=0.995, top=0.95, bottom=0.01, wspace=0.02, hspace=0.04)
fig.savefig(out_dir / 'examples.pdf', dpi=170)
fig.savefig(out_dir / 'examples.png', dpi=110)
print(picks)
