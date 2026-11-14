#!/bin/bash
# One whole-disk member for one fold on any CUDA machine (xixi, Kaggle, Colab): 1024 px,
# then 1536 px initialized from it, out-of-fold maps after each stage, then test maps.
#
#   run_global_member.sh NAME FOLD ENCODER SEED AMP DATA [WORKERS]
#
# Run from the project root with PYTHONPATH containing src (and timm). PYTHON overrides
# the interpreter. Outputs: checkpoints/global/NAME_fF_{1024,1536}.pt,
# outputs/global_NAME_fF_{1024,1536} and outputs/global_test_NAME_fF_1536.
set -e
NAME=$1 F=$2 ENC=$3 SEED=$4 AMP=$5 DATA=$6 W=${7:-4}
P="${PYTHON:-python} -W ignore"
COMMON="--data $DATA --fold $F --seed $SEED --encoder $ENC --amp $AMP --workers $W"
mkdir -p checkpoints/global
$P scripts/train_global_unet.py $COMMON --size 1024 --epochs 30 --lr 3e-4 --warmup 1 --batch 2 --accumulate 2 \
  --output checkpoints/global/${NAME}_f${F}_1024.pt --maps-dir outputs/global_${NAME}_f${F}_1024
$P scripts/train_global_unet.py $COMMON --size 1536 --epochs 24 --lr 1e-4 --warmup 0.5 --batch 1 --accumulate 4 \
  --init checkpoints/global/${NAME}_f${F}_1024.pt --output checkpoints/global/${NAME}_f${F}_1536.pt \
  --maps-dir outputs/global_${NAME}_f${F}_1536
$P scripts/train_global_unet.py --data $DATA --amp $AMP --predict --test --size 1536 \
  --output checkpoints/global/${NAME}_f${F}_1536.pt --maps-dir outputs/global_test_${NAME}_f${F}_1536
echo MEMBER DONE
