#!/bin/bash

export TORCH_ALLOW_CLASSES="argparse.Namespace"

# Note: Using master_port 29505 to avoid collision if training is still running
torchrun --nproc_per_node=1 --master_port=29505 main.py \
    --output_dir logs/proposed_r50_v8_triplefilter_LONG_eval \
    -c config/OV_COCO/OVDQUO_RN50.py \
    --options dataset_file=ovcoco text_len=77 \
    --eval \
    --resume logs/proposed_r50_v8_triplefilter/checkpoint0004.pth