import os
import random
import sys

import numpy as np

# Root directory of the project
from samples.shapes.shapes import ShapesConfig, ShapesDataset

ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

################### Init config #################

config = ShapesConfig()
config.display()

################### Defining Datasets #################

# Training dataset
dataset_train = ShapesDataset()
dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(3, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

################### Constructing the model #################

# Create model in training mode
model_train = modellib.MaskRCNN(mode="training", config=config,
                                model_dir=MODEL_DIR)


################### Preparing mAP Callback #################

class _InfConfig(ShapesConfig):
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    DETECTION_MIN_CONFIDENCE = 0.0


model_inference = modellib.MaskRCNN(mode="inference", config=_InfConfig(), model_dir=MODEL_DIR)
mean_average_precision_callback = modellib.MeanAveragePrecisionCallback(model_train, model_inference, dataset_val, 1,
                                                                        verbose=1)

################### Training #################

# Which weights to start with?
model_train.load_weights(COCO_MODEL_PATH, by_name=True,
                         exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                  "mrcnn_bbox", "mrcnn_mask"])

model_train.train(dataset_train, dataset_val,
                  learning_rate=config.LEARNING_RATE / 10,
                  epochs=10,
                  layers="all",
                  custom_callbacks=[mean_average_precision_callback])


################### Save model weights #################

# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)

################### Inference #################

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

# Recreate the model in inference mode
model_inference = modellib.MaskRCNN(mode="inference",
                                    config=inference_config,
                                    model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model_inference.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model_inference.load_weights(model_path, by_name=True)

################### Test on random image #################

image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
    modellib.load_image_gt(dataset_val, inference_config,
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

results = model_inference.detect([original_image], verbose=1)

r = results[0]
print("Results: {0}".format(r))

################### Evaluation #################

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model_inference.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps = \
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

print("mAP: ", np.mean(APs))
