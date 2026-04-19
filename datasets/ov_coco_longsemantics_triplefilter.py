# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.
"""
from pathlib import Path

import torch
import json
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
import datasets.transforms as T

class OVCocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self, img_folder, ann_file, transforms, return_masks, pseudo_box
    ):
        super(OVCocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.all_categories = {
            k["id"]: k["name"] for k in self.coco.dataset["categories"]
        }
        
        raw_categories = [self.all_categories[k] for k in sorted(self.all_categories.keys())]
        
        # --- THESIS V6: RESTORING LONG SEMANTICS (DESCRIPTIVE PROMPTS) ---
        COCO_SEMANTICS = {
            'person': 'a human being with a head, torso, arms, and legs, often wearing clothing',
            'bicycle': 'a two-wheeled human-powered vehicle with a metal frame, handlebars, pedals, and a seat',
            'car': 'a four-wheeled motorized vehicle with windows, doors, and a metallic body designed for roads',
            'motorcycle': 'a heavy two-wheeled motorized vehicle with a seat, an engine, and handlebars',
            'airplane': 'a large flying vehicle with fixed wings, a tubular fuselage, and engines',
            'bus': 'a large long motorized vehicle with many windows and seats for transporting multiple passengers',
            'train': 'a series of connected railway cars moving along a metal track',
            'truck': 'a large motorized cargo vehicle with a front cab and a flatbed or enclosed trailer',
            'boat': 'a watercraft with a curved hull designed to float and move across water',
            'traffic light': 'a traffic light signal',
            'fire hydrant': 'a metal fire hydrant on a sidewalk',
            'stop sign': 'a red octagonal stop sign',
            'parking meter': 'a coin-operated parking meter',
            'bench': 'a long outdoor seat made of wood or metal designed for multiple people to sit',
            'bird': 'a feathered flying animal with wings, a beak, and two legs',
            'cat': 'a small domesticated feline animal with fur, pointed ears, whiskers, and a long tail',
            'dog': 'a domesticated canine animal with fur, four legs, a snout, and a tail',
            'horse': 'a large four-legged animal with a mane, hooves, and a long tail, often ridden',
            'sheep': 'a four-legged farm animal covered in a thick coat of white or grey wool',
            'cow': 'a large four-legged bovine farm animal with hooves and often horns, known for producing milk',
            'elephant': 'a massive grey animal with a long flexible trunk, large floppy ears, and tusks',
            'bear': 'a large heavy mammal with thick fur, a short tail, and sharp claws',
            'zebra': 'a wild horse-like animal with distinct alternating black and white stripes',
            'giraffe': 'a tall African animal with a very long neck, long legs, and a patterned spotted coat',
            'backpack': 'a fabric bag carried on the back with two shoulder straps and zippers',
            'umbrella': 'a portable circular canopy of fabric on a folding metal frame attached to a central rod',
            'handbag': 'a medium-to-large bag typically carried by women to hold personal items',
            'tie': 'a long narrow piece of fabric worn around the neck under a shirt collar',
            'suitcase': 'a large rectangular piece of luggage with a handle used for carrying clothes',
            'frisbee': 'a flat circular plastic disc designed to be thrown and caught for sport',
            'skis': 'a pair of long narrow flat runners worn on the feet for gliding over snow',
            'snowboard': 'a single wide flat board strapped to the feet for sliding down snow-covered slopes',
            'sports ball': 'a round sports ball for playing games',
            'kite': 'a lightweight frame covered with fabric or paper flown in the wind at the end of a long string',
            'baseball bat': 'a wooden or metal baseball bat',
            'baseball glove': 'a leather baseball glove',
            'skateboard': 'a short narrow wooden board with four small wheels mounted underneath',
            'surfboard': 'a long narrow fiberglass board used for riding ocean waves',
            'tennis racket': 'a stringed tennis racket',
            'bottle': 'a narrow-necked glass or plastic container used to hold liquids',
            'wine glass': 'a glass goblet for wine',
            'cup': 'a small bowl-shaped container with a handle used for drinking beverages',
            'fork': 'a metal or plastic utensil with two or more prongs used for eating food',
            'knife': 'a utensil with a handle and a flat metal blade with a sharp edge',
            'spoon': 'a utensil consisting of a small shallow oval bowl on a long handle',
            'bowl': 'a round deep dish or basin used for holding food or liquid',
            'banana': 'a long curved yellow fruit with a thick peel',
            'apple': 'a round fruit with red or green skin and a solid whitish interior',
            'sandwich': 'two pieces of bread with meat, cheese, or other filling placed between them',
            'orange': 'a round citrus fruit with a tough bright reddish-yellow dimpled rind',
            'broccoli': 'a green vegetable with a thick stalk and a tree-like flowery head',
            'carrot': 'a long pointed orange root vegetable',
            'hot dog': 'a hot dog sausage in a bun',
            'pizza': 'a round flat piece of dough baked with tomato sauce, cheese, and toppings',
            'donut': 'a small ring-shaped fried cake often covered in sweet icing or sugar',
            'cake': 'a sweet baked dessert made from dough or batter, often decorated with frosting',
            'chair': 'a piece of furniture for one person to sit on, with a back and four legs',
            'couch': 'a long upholstered piece of furniture for multiple people to sit or lie on',
            'potted plant': 'a green plant in a pot',
            'bed': 'a large piece of furniture for sleep or rest, typically with a mattress and frame',
            'dining table': 'a table used for eating meals',
            'toilet': 'a ceramic bowl with a hinged seat and a water tank used for disposing of human waste',
            'tv': 'a rectangular electronic device with a flat screen used for viewing broadcasting',
            'laptop': 'a portable folding personal computer with a flat screen and a keyboard',
            'mouse': 'a small hand-held pointing device used to control a computer screen cursor',
            'remote': 'a small handheld device with buttons used to operate electronic equipment from a distance',
            'keyboard': 'a flat rectangular panel of physical keys used for typing into a computer',
            'cell phone': 'a mobile cellular phone',
            'microwave': 'a box-like electronic oven that cooks or heats food quickly using radiation',
            'oven': 'an enclosed compartment for cooking and heating food at high temperatures',
            'toaster': 'an electrical appliance designed to toast sliced bread using heating elements',
            'sink': 'a fixed basin with a water faucet and a drain used for washing',
            'refrigerator': 'a large tall appliance with doors that keeps food and drinks cold',
            'book': 'a written or printed work consisting of pages bound together with a cover',
            'clock': 'a circular or digital device for measuring and displaying the time',
            'vase': 'a decorative vertical container often made of glass or ceramic used to hold flowers',
            'scissors': 'a hand-operated cutting instrument consisting of two pivoted metal blades',
            'teddy bear': 'a soft stuffed teddy bear toy',
            'hair drier': 'a handheld hair drier appliance',
            'toothbrush': 'a small brush with a long handle used for cleaning teeth'
        }
        
        self.category_list = [COCO_SEMANTICS.get(name, f"a visual of a {name} object") for name in raw_categories]
        # ---------------------------------------------------------
        
        self.category_ids = {v: k for k, v in self.all_categories.items()}
        self.label2catid = {
            k: self.category_ids[raw_categories[k]] for k, v in enumerate(self.category_list)
        }
        self.catid2label = {v: k for k, v in self.label2catid.items()}
        self.use_pseudo_box = pseudo_box != ""
        if self.use_pseudo_box:
            with open(pseudo_box, "r") as f:
                pseudo_annotations = json.load(f)
            self.pseudo_annotations = dict()
            for annotation in pseudo_annotations:
                if annotation["image_id"] not in self.pseudo_annotations:
                    self.pseudo_annotations[annotation["image_id"]] = []
                self.pseudo_annotations[annotation["image_id"]].append(annotation)
        self.prepare = ConvertCocoPolysToMask(return_masks, map=self.catid2label)

    def __getitem__(self, idx):
        img, target = super(OVCocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        if self.use_pseudo_box:
            pseudo_annotations = (
                self.pseudo_annotations[image_id]
                if image_id in self.pseudo_annotations
                else [])
            target.extend(pseudo_annotations)
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
            
        # --- THESIS: Inject FastSAM Priors into DataLoader (Train + Val) ---
        import os
        prior_val = f"data/sam_priors/val2017/{image_id}.pt"
        prior_train = f"data/sam_priors/train2017/{image_id}.pt"
        
        if os.path.exists(prior_val):
            payload = torch.load(prior_val)
            target['sam_proposals'] = payload['boxes']
            target['sam_masks'] = payload['masks']
        elif os.path.exists(prior_train):
            payload = torch.load(prior_train)
            target['sam_proposals'] = payload['boxes']
            target['sam_masks'] = payload['masks']
        else:
            target['sam_proposals'] = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
            target['sam_masks'] = torch.zeros((1, 28, 28), dtype=torch.bool)
        # -----------------------------------------------------
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, map=None):
        self.return_masks = return_masks
        self.map = map

    def __call__(self, image, target):
        w, h = image.size
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        anno = target["annotations"]
        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        classes = [
            obj["category_id"] if "category_id" in obj else -i - 1
            for i, obj in enumerate(anno)
        ]
        pseudo_label_map = {}
        pseudo_mask=[]
        weight=[]
        for i, obj in enumerate(anno):
            if "class_label" in obj:
                pseudo_label_map[-i - 1] = obj["class_label"]
            if "pseudo" in obj:
                pseudo_mask.append(obj["pseudo"])
            else:
                pseudo_mask.append(0)
            if "weight" in obj:
                weight.append(obj["weight"]) 
            else:
                weight.append(1.0)
        classes = torch.tensor(classes, dtype=torch.int64)
        pseudo_mask= torch.tensor(pseudo_mask, dtype=torch.int64)
        weight= torch.tensor(weight, dtype=torch.float32)
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        pseudo_mask=pseudo_mask[keep]
        weight=weight[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["pseudo_label_map"] = pseudo_label_map
        target["pseudo_mask"] = pseudo_mask
        target["weight"] = weight
        if self.map is not None: 
            for idx, label in enumerate(target["labels"]):
                target["labels"][idx] = (
                    self.map[label.item()] if label.item() >= 0 else label.item()
                )
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return image, target


def make_coco_transforms(image_set, args):
    MEAN = [0.48145466, 0.4578275, 0.40821073]
    STD = [0.26862954, 0.26130258, 0.27577711]
    normalize = T.Compose([T.ToRGB(), T.ToTensor(), T.Normalize(MEAN, STD)])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if image_set == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize(scales, max_size=1333),
                        ]
                    ),
                ),
                normalize,
            ]
        )
    if image_set == "val":
        return T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                normalize,
            ]
        )
    raise ValueError(f"unknown {image_set}")


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f"provided COCO path {root} does not exist"
    mode = "instances"
    if args.label_version=="standard":
        # --- FIXED TO MATCH YOUR CURRENT REPO STRUCTURE ---
        PATHS = {
            "train": (
                root / "Images/train2017",
                root / "Annotations" / f"{mode}_train2017_base.json",
            ),
            "val": (
                root / "Images/val2017",
                root / "Annotations" / f"{mode}_val2017_basetarget.json",
            ),
        }
    elif args.label_version == "custom":
            PATHS["train"]=(
            root / "Images/train2017",
            root / "instances_train2017_base_RN50relabel_pseudo.json",
        )
    img_folder, ann_file = PATHS[image_set]
    dataset = OVCocoDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set, args),
        return_masks=args.masks,
        pseudo_box=args.pseudo_box,
    )
    return dataset