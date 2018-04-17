"""
Mobile Mask R-CNN Train & Eval Script
for Training on the COCO Dataset

written by github.com/GustavZ
adopted from github.com/matterport
"""

# Import Packages
import os
import sys
import time
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import zipfile
#import urllib.request
import shutil

# Import Mobile Mask R-CNN
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from samples.coco import coco

# Weights and Logs
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2017"

# COCO Dataset
config = coco.MobileCocoConfig()
COCO_DIR = os.path.join(ROOT_DIR,"data/coco")
MODEL_DIR = os.path.join(ROOT_dir, "logs")

# Model
model = modellib.MaskRCNN(mode="training", model_dir = MODEL_DIR, config=config)
model_path = model.get_imagenet_weights()
print("Loading weights ", model_path)
model.load_weights(model_path, by_name=True)
model.keras_model.summary()

# Training - Stage 1
print("Training network heads")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=160,
            layers='heads',
            augmentation=augmentation)

# Training - Stage 2
# Finetune layers from ResNet stage 4 and up
print("Fine tune {} stage 4 and up".format(config.ARCH))
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=120,
            layers="11M+",
            augmentation=augmentation)

# Training - Stage 3
# Fine tune all layers
print("Fine tune all layers")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=40,
            layers='all',
            augmentation=augmentation)
