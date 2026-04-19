# #!/bin/bash

# export TORCH_ALLOW_CLASSES="argparse.Namespace"

# # Notice the --eval flag and the --resume pointing to your finished V6 weights
# torchrun --nproc_per_node=1 --master_port=29500 main.py \
#     --output_dir logs/proposed_r50_v6_LONG_eval \
#     -c config/OV_COCO/OVDQUO_RN50.py \
#     --options dataset_file=ovcoco text_len=77 \
#     --eval \
#     --resume logs/proposed_r50_v6_dualstream/checkpoint.pth


#!/bin/bash

export TORCH_ALLOW_CLASSES="argparse.Namespace"

torchrun --nproc_per_node=1 --master_port=29500 main.py \
    --output_dir logs/proposed_r50_v7_fusion_LONG_eval \
    -c config/OV_COCO/OVDQUO_RN50.py \
    --options dataset_file=ovcoco text_len=77 \
    --eval \
    --resume logs/proposed_r50_v7_fusion/checkpoint.pth