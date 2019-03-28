#! /bin/bash

python spatialrun.py \
    --data-dir data/ShapeNetCore.v2.PC15k \
    --batch-size 32 \
    --epochs 1000 \
    --save-interval 10 \
    --val-interval 10 \
    --viz-interval 1 \
    --output-dir runs \
    --log-name normal-denorm
    # --log-name normal-clip
    # --log-name normal

echo "Done"
exit
