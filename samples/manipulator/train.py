#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN - Train on Manipulator Dataset
# 
# 
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
# 
# The code of the *Manipulator* dataset is included below.

# In[1]:


import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import skimage.io

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

data_dir = '/media/ianormy/Storage/Data/datasets/manipulator'

train_data_dir = os.path.join(os.path.join(data_dir, 'train'), 'images')
train_quad_labels = [e for e in os.scandir(train_data_dir)]

# ## Configurations


class ManipulatorConfig(Config):
    """Configuration for training on the manipulator dataset.
    Derives from the base Config class and overrides values specific
    to the manipulator dataset.
    """
    # Give the configuration a recognizable name
    NAME = "manipulator"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 4 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # training steps
    STEPS_PER_EPOCH = 100

    # validation steps since the epoch is small
    VALIDATION_STEPS = 25

    DETECTION_MIN_CONFIDENCE = 0

    MAX_GT_INSTANCES = 200

    DETECTION_MAX_INSTANCES = 1
    
config = ManipulatorConfig()
config.display()


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


class ManipulatorDataset(utils.Dataset):
    """Gets the manipulator dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_manipulator(self, quad_labels, li, ui):
        """load a subset of the manipulator dataset.
        :arg dataset_dir: root directory of the dataset.
        :arg li: lower index
        :arg ui: upper index
        """
        # Add classes
        # naming the dataset manipulator and the class tip
        self.add_class("manipulator", 1, "tip")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for idx, entry in enumerate(quad_labels[li:ui]):
            filename=entry.path
            image_id = Path(filename).stem
            self.add_image(
                "manipulator",
                image_id=image_id,
                path=filename)
            
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), 'masks')
        mask_name = os.path.splitext(os.path.join(mask_dir, os.path.basename(info['path'])))[0]+'_mask.tif'
        # Read mask files from .tif image
        mask = []
        im = cv2.imread(os.path.join(mask_dir, mask_name))[:,:,0]
        pad_rows = np.zeros([256, 1024])
        m = np.vstack([im, pad_rows])
        m = np.uint8(m/255)
        mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        gray = cv2.imread(info['path'])
        gray_pad = pad_image(gray)
        #color_image = cv2.merge([gray_pad, gray_pad, gray_pad])
        return gray_pad

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "manipulator":
            return info["id"]
        else:
            super(self.__class__).image_reference(self, image_id)


def pad_image(im, target_size=(1024, 1024)):
    pad_h_top = 0
    pad_w_left = 0
    pad_h_bottom = max(target_size[0] - im.shape[0], 0)
    pad_w_right = max(target_size[1] - im.shape[1], 0)
    pad_im = cv2.copyMakeBorder(im, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, cv2.BORDER_CONSTANT)
    return pad_im


# Training dataset
dataset_train = ManipulatorDataset()
dataset_train.load_manipulator(train_quad_labels, 0, 100)
dataset_train.prepare()

# Validation dataset
dataset_val = ManipulatorDataset()
dataset_val.load_manipulator(train_quad_labels, 101, 145)
dataset_val.prepare()


# ## Create Model

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)


# ## Training

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, 
            dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=50,
            layers='heads')

# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
model_path = os.path.join(MODEL_DIR, "mask_rcnn_manipulator.h5")
model.keras_model.save_weights(model_path)
