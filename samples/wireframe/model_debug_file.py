
"""
Imports
"""
import os
import sys
from mrcnn import utils
import mrcnn.model as modellib
from samples.wireframe.WireframeGenerator import generate_data
from samples.wireframe import Wireframe
import tensorflow as tf


"""
Configs
"""
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library
config = Wireframe.WireframeConfig()
WIREFRAME_DIR = os.path.join(ROOT_DIR, "datasets/wireframe")


"""
Data Preperation
"""

NUM_TRAINING_IMAGES = 100
MAX_ICONS_PER_IMAGE = 3
# generate_data(NUM_TRAINING_IMAGES, MAX_ICONS_PER_IMAGE)

# Training dataset
dataset_train = Wireframe.WireframeDataset()
dataset_train.load_wireframe(WIREFRAME_DIR, "train")
dataset_train.prepare()

# Validation dataset
dataset_val = Wireframe.WireframeDataset()
dataset_val.load_wireframe(WIREFRAME_DIR, "val")
dataset_val.prepare()

dataset = Wireframe.WireframeDataset()
dataset.load_wireframe(WIREFRAME_DIR, "train")

dataset.prepare()



"""
Model Generation
"""
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)




"""
Train
"""
# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=2,
            layers='heads')
