import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

import utils
import visualize
from visualize import display_images
import model as modellib
from model import log

ROOT_DIR = os.getcwd()

import coco
import gdxray
config = gdxray.TrainConfig()
DATA_DIR = os.path.expanduser("~/data/gdxray")




# Load dataset
if config.NAME == 'shapes':
    dataset = shapes.ShapesDataset()
    dataset.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
elif config.NAME == "coco":
    dataset = coco.CocoDataset()
    dataset.load_coco(DATA_DIR, "train")
elif config.NAME == "gdxray":
    dataset = gdxray.XrayDataset()
    dataset.load_gdxray(DATA_DIR, "train", "Castings")

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))


plt.ion()
ax = None

for image_id in dataset.image_ids:
	image = dataset.load_image(image_id)
	mask, class_ids = dataset.load_mask(image_id)
	original_shape = image.shape
	# Resize
	image, window, scale, padding = utils.resize_image(
	    image,
	    min_dim=config.IMAGE_MIN_DIM,
	    max_dim=config.IMAGE_MAX_DIM,
	    padding=config.IMAGE_PADDING)
	mask = utils.resize_mask(mask, scale, padding)
	# Compute Bounding box
	bbox = utils.extract_bboxes(mask)

	for box in bbox:
		print(box,"box")

	# Display image and additional stats
	print("image_id: ", image_id, dataset.image_reference(image_id))
	print("Original shape: ", original_shape)
	log("image", image)
	log("mask", mask)
	log("class_ids", class_ids)
	log("bbox", bbox)
	# Display image and instances
	if ax:
		ax.clear()
	ax = visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names, ax=ax)
	input("next?")


