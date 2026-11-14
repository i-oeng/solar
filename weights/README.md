Release assets (tag v1.0) are named with `__` in place of `/`:

* `rescorer_tree_a.txt`
* `convnext__f0_1536.pt`, `convnext__f1_1536.pt`, `convnext__f2_1536.pt`
* `resnet34__annotation_round_f{0,1,2}_s{2026,2029}_{most_complete,uniform}.pt` (12 files)

`pipeline.ipynb` downloads them into this folder.
