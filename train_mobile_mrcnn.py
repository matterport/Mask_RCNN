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
import urllib.request
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
model.get_imagenet_weights()
model.keras_model.summary()

# Training - Stage 1
print("Train heads")
model.train(train_dataset_keypoints, val_dataset_keypoints,
           learning_rate=config.LEARNING_RATE,
           epochs=15,
           layers='heads')
# Training - Stage 2
# Finetune layers from stage 4 and up
print("Training Resnet layer 4+")
model.train(train_dataset_keypoints, val_dataset_keypoints,
           learning_rate=config.LEARNING_RATE / 10,
           epochs=20,
           layers='4+')
# Training - Stage 3
# Finetune layers from stage 3 and up
print("Training Resnet layer 3+")
model.train(train_dataset_keypoints, val_dataset_keypoints,
           learning_rate=config.LEARNING_RATE / 100,
           epochs=100,
           layers='all')
