import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from ultralytics import FastSAM

# Unified Configuration for both Train and Val
SPLITS = [
    {
        "name": "Validation",
        "json": "data/Annotations/instances_val2017_basetarget.json",
        "img_dir": "data/Images/val2017/",
        "out_dir": "data/sam_priors/val2017/"
    },
    {
        "name": "Train",
        "json": "data/Annotations/instances_train2017_base.json",
        "img_dir": "data/Images/train2017/",
        "out_dir": "data/sam_priors/train2017/"
    }
]

MASK_SIZE = 28  # Highly optimized grid size to prevent storage explosion

print("Warming up FastSAM Engine for Pre-computation...")
fastsam_model = FastSAM("FastSAM-x.pt")

def extract_and_save_boxes(img_filename, img_id, img_dir, out_dir):
    img_path = os.path.join(img_dir, img_filename)
    if not os.path.exists(img_path): return
    
    out_file = os.path.join(out_dir, f"{img_id}.pt")
    if os.path.exists(out_file): return # Skip if already computed
        
    results = fastsam_model(img_path, device='cuda', retina_masks=True, imgsz=1024, conf=0.1, iou=0.6, verbose=False)
    
    # Fallback if FastSAM finds absolutely nothing
    if len(results[0].boxes.xyxy) == 0 or results[0].masks is None:
        boxes_cxcywh = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
        masks_tensor = torch.zeros((1, MASK_SIZE, MASK_SIZE), dtype=torch.bool)
        torch.save({'boxes': boxes_cxcywh, 'masks': masks_tensor}, out_file)
        return

    img_h, img_w = results[0].orig_shape
    boxes_xyxy = results[0].boxes.xyxy
    masks_full = results[0].masks.data # [N, H, W]
    
    scores = results[0].boxes.conf
    _, indices = scores.sort(descending=True)
    
    # Cap at top 300 to prevent memory blowouts
    boxes_xyxy = boxes_xyxy[indices][:300]
    masks_full = masks_full[indices][:300]
    
    # 1. Format the Bounding Boxes (cx, cy, w, h)
    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0 / img_w
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0 / img_h
    w = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / img_w
    h = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / img_h
    boxes_cxcywh = torch.stack([cx, cy, w, h], dim=-1).cpu()
    
    # 2. Extract, Crop, and Resize Masks (Memory Safe)
    cropped_masks = []
    for i in range(len(boxes_xyxy)):
        # Get integer coordinates for cropping
        x1, y1, x2, y2 = boxes_xyxy[i].int().tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        
        # Failsafe for invalid boxes (w or h <= 0)
        if x2 <= x1 or y2 <= y1:
            cropped_masks.append(torch.zeros((MASK_SIZE, MASK_SIZE), dtype=torch.bool))
            continue
            
        # Crop the mask to the exact bounding box
        mask_crop = masks_full[i, y1:y2, x1:x2].unsqueeze(0).unsqueeze(0).float()
        
        # Resize to standard grid using nearest interpolation to keep it binary
        mask_resized = F.interpolate(mask_crop, size=(MASK_SIZE, MASK_SIZE), mode='nearest').squeeze(0).squeeze(0)
        cropped_masks.append(mask_resized.bool().cpu())
        
    masks_tensor = torch.stack(cropped_masks)
    
    # Save as a dictionary
    payload = {
        'boxes': boxes_cxcywh,
        'masks': masks_tensor
    }
    torch.save(payload, out_file)

if __name__ == "__main__":
    for split in SPLITS:
        os.makedirs(split["out_dir"], exist_ok=True)
        print(f"\nLoading {split['name']} Annotations from {split['json']}")
        
        with open(split['json'], 'r') as f:
            coco_data = json.load(f)
            
        images = coco_data['images']
        print(f"Found {len(images)} images in {split['name']} split. Starting Extraction...")
        
        for img in tqdm(images):
            extract_and_save_boxes(img['file_name'], img['id'], split['img_dir'], split['out_dir'])
            
    print("\nTotal Pre-computation Complete. Priors and Masks safely saved to disk.")