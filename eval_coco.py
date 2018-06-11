"""
Mobile Mask R-CNN Train & Eval Script
for Training on the COCO Dataset

written by github.com/GustavZ
"""
# Import Packages
import os
import sys
import random
import numpy as np

# Import Mobile Mask R-CNN
from mmrcnn import model as modellib, utils, visualize
from mmrcnn.model import log
import coco

# Paths
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_DIR = os.path.join(ROOT_DIR, 'data/coco')
DEFAULT_WEIGHTS = os.path.join(ROOT_DIR, "mobile_mask_rcnn_coco.h5")
NUM_EVALS = 10

# Load Model
config = coco.CocoConfig()
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
#model_path = DEFAULT_WEIGHTS
model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("> Loading weights from {}".format(model_path))
model.load_weights(model_path, by_name=True)

# Dataset
class_names = ['person']  # all classes: None
dataset_val = coco.CocoDataset()
COCO = dataset_val.load_coco(COCO_DIR, "val", class_names=class_names, return_coco=True)
dataset_val.prepare()
print("> Running COCO evaluation on {} images.".format(NUM_EVALS))
coco.evaluate_coco(model, dataset_val, COCO, "bbox", limit=NUM_EVALS)
model.keras_model.save(MODEL_DIR+"/mobile_mask_rcnn_{}.h5".format(config.NAME))


# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, config,
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                            dataset_train.class_names, figsize=(8, 8))

results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_val.class_names, r['scores'], ax=get_ax())

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []

for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]

    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

print("mAP: ", np.mean(APs))

image_ids = np.random.choice(dataset_val.image_ids, 10)

image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, config,
                               image_ids[0], use_mini_mask=False)

print (r['masks'])
