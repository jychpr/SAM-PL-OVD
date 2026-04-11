#!/bin/bash
echo "Precomputing FastSAM Priors for COCO Validation (5k images)..."
python tools/sam_priors/precompute_sam_coco_val.py

echo "Precomputing FastSAM Priors for COCO Training (118k images)..."
python tools/sam_priors/precompute_sam_coco.py

echo "Pre-computation Pipeline Finished."