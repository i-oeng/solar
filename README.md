# Solar Filament Segmentation Challenge 2026: PQ 0.38 public

Entry of Islam Kossymbayev (Nazarbayev University) to the Solar Filament Segmentation
Challenge 2026 (Kaggle `filament-segmentation-2026`, IEEE BigData Cup).

The pipeline averages two groups of U-Nets and then chooses the instances exactly:

1. **Crop level ResNet34 U-Nets**: 12 models (3 date-grouped folds x 2 seeds x 2 view-sampling
   rules), 512 px crops, flip TTA.
2. **Whole-disk ConvNeXt-Tiny U-Nets**: 3 models (one per fold), disk resized to 1024 px then
   1536 px, flip TTA, extra instance-boundary output.
3. The two group means are averaged into one probability map.
4. **Candidate forest and exact selection**: the map is cut at six grow thresholds
   (0.3 to 0.8). Candidates nest, so they form a forest. A LightGBM regressor predicts each
   candidate's matched IoU, and dynamic programming picks the best set of disjoint
   candidates for Panoptic Quality (`src/solar_filament/candidate_tree.py`).

Out of fold development PQ over the 707 training images is 0.443; the public score of the
submitted file is 0.38. Details, tables and uncertainty are in the report
(`report/`, template of the organizers).

## Reproduce the submission without training

```bash
pip install -r requirements.txt
# weights are release assets (v1.0): 12 ResNet34, 3 ConvNeXt and the scorer, about 1.5 GB
python reproduce.py --data PATH/TO/competition --weights weights --work work
```

`pipeline.ipynb` does the same, including the weight download. A GPU is needed for the
model steps (about 45 minutes on an RTX 5060). `submission_public_0.38.csv` is the file that
was uploaded to Kaggle. Probabilities from different GPU kernels can differ in the last
digit, so a rerun may differ by a mask or two; use `--limit 3` for a quick check.

The weights are not in git because of their size. The notebook downloads them from the
GitHub release assets (https://github.com/i-oeng/solar/releases/tag/v1.0) listed in `weights/README.md`.

## Layout

| Path | Content |
|---|---|
| `reproduce.py`, `pipeline.ipynb` | inference only pipeline |
| `src/solar_filament/` | data, models, tiling, candidates, scorer features, candidate forest, metrics |
| `scripts/` | training (`train_global_unet.py`, `run_fold_members.py`), evaluation, submission tools |
| `results/` | date-grouped fold files and the postprocessing settings |
| `report/` | report text for the organizers' template and figures |

## Training (not needed to reproduce)

* ResNet34 members: `python -m solar_filament.train` through `scripts/run_fold_members.py`
  and `scripts/run_annotation_round.py` (30 epochs, about 12 minutes on an RTX 5060 each).
* ConvNeXt members: `scripts/run_global_member.sh NAME FOLD ENCODER SEED AMP DATA`.
* Scorer and level set: `scripts/evaluate_tree_selection.py --export`.

## Rules

Only the competition training annotations were used. No test image was labelled or used
for training, and no other MAGFiLO annotation was used. Encoders start from public
ImageNet weights (torchvision, timm). The whole-disk model follows ideas from the public
NSO entry (github.com/pragyaangaur/NSO-Solar-Filament-Challenge); the code was written from
scratch.
