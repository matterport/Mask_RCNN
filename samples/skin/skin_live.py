#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import time
import random
import numpy as np
import imgaug 
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import zipfile
import urllib.request
import json
import shutil
from PIL import Image, ImageDraw
import cv2


# In[4]:


# Root directory of the project
ROOT_DIR = os.path.abspath("../../../Mask_RCNN/")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Import Mask RCNN
# Import mrcnn libraries
sys.path.append(ROOT_DIR) 
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn.model import log


############################################################
#  Configurations
############################################################

class SkinConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "skin"
    
    GPU_COUNT = 4
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + skin

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 200
    
    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 70
    
    # Matterport originally used resnet101, but I downsized to fit it on my graphics card
    BACKBONE = 'resnet50'

    # Skip detections with < 90% confidence
#     DETECTION_MIN_CONFIDENCE = 0.9
     # To be honest, I haven't taken the time to figure out what these do
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50 
    POST_NMS_ROIS_INFERENCE = 500 
    POST_NMS_ROIS_TRAINING = 1000 
    
    
config = SkinConfig()
config.display()


# In[8]:



class InferenceConfig(SkinConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    DETECTION_MIN_CONFIDENCE = 0.85
    

inference_config = InferenceConfig()


# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)


model_path = '/home/anirudh/Umbilicus_Skin_Detection/Mask_RCNN/logs/mask_rcnn_skin_0016.h5'
model.load_weights(model_path, by_name=True)


cv2.namedWindow("RGB")
cv2.namedWindow("Mask")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    rval, frame = vc.read()
    ima1 = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = model.detect([ima1], verbose=0)
    r = results[0]
    if (r['rois'].shape[0]==0):
        print("skipping")
        continue

    maskpre=  r['masks'][:,:,0]
    maskpre = maskpre.astype(np.uint8)
    print("Detecting")
    cv2.imshow("Mask", maskpre*255)
    cv2.imshow("RGB",frame)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("RGB")
cv2.destroyWindow("Mask")