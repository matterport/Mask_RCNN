"""
Mobile Mask R-CNN Train & Eval Script
for Training on the COCO Dataset

written by github.com/GustavZ
adopted from github.com/matterport
"""

# Import Packages
import os
import sys
import imgaug

# Import Mobile Mask R-CNN
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from samples.coco import coco

# Paths
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_DIR = os.path.join(ROOT_DIR, 'data/coco')

# Model
config = coco.CocoConfig()
model = modellib.MaskRCNN(mode="training", model_dir = MODEL_DIR, config=config)
model_path = model.get_imagenet_weights()
print("> Loading weights ", model_path)
model.load_weights(model_path, by_name=True)
#model.keras_model.summary()

# Dataset
class_names = ['person']
dataset_train = coco.CocoDataset()
dataset_train.load_coco(COCO_DIR, "train", class_ids=class_names)
dataset_train.prepare()
dataset_val = coco.CocoDataset()
dataset_val.load_coco(COCO_DIR, "val", class_ids=class_names)
dataset_val.prepare()

# Training - Config
augmentation = imgaug.augmenters.Fliplr(0.5)

# Training - Stage 1
print("> Training network heads")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=160,
            layers='heads',
            augmentation=augmentation)

# Training - Stage 2
# Finetune layers  stage 4 and up
print("> Fine tune {} stage 4 and up".format(config.ARCH))
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=120,
            layers="11M+",
            augmentation=augmentation)

# Training - Stage 3
# Fine tune all layers
print("> Fine tune all layers")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=40,
            layers='all',
            augmentation=augmentation)

# Evaluation
NUM_EVALS = 500
print("Running COCO evaluation on {} images.".format(NUM_EVALS))
coco.evaluate_coco(model, dataset_val, coco, "bbox", limit=NUM_EVALS)
