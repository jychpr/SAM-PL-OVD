#!/bin/bash

output_dir=$1

export TORCH_ALLOW_CLASSES="argparse.Namespace"

torchrun --nproc_per_node=1 --master_port=29500 main.py \
    --output_dir $output_dir \
    -c config/OV_COCO/OVDQUO_RN50.py \
    --options dataset_file=ovcoco text_len=77 \
    --amp \
    --eval_start_epoch 5 \
    --eval_every_epoch 5