# -*- coding: utf-8 -*-
"""

Train Mask R-CNN

@author: Mattia Brusamento
"""

from datetime import datetime
import os
import sys
import time
import numpy as np
from imgaug import augmenters as iaa 

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


from mrcnn import model as modellib, utils
from materials_trash_detector.trash_config import TrashConfig
from materials_trash_detector.trash_dataset import TrashDataset


# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join("logs")


def get_augmentation_pipeline(custom=False, n=10):
    if custom:
        return iaa.SomeOf(n, [
                    iaa.AdditiveLaplaceNoise(scale=0.2*255, per_channel=True), #add some laplace noise, which contains also heavy tails deviations.
                    iaa.GaussianBlur(sigma=(0.0, 3.0), name="Blur"), # blur the image
                    iaa.MotionBlur(k=25, angle=np.random.randn()*20), # simulate motion movements
                    iaa.Dropout([0.0, 0.05], name='Dropout'), # drop 0-5% of all pixels
                    iaa.Fliplr(0.5),
                    iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
                    iaa.GammaContrast((0.5, 2.0)),
                    iaa.ChangeColorTemperature((1100, 10000)),
                    iaa.Add((-20, 20),name="Add"),
                    iaa.Multiply((0.8, 1.2), name="Multiply"),
                    iaa.Fliplr(0.5),
                    iaa.Flipud(0.5),
                    iaa.Affine(scale=(0.8, 2.0)),
                    iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                    iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees
                ], random_order=True)
    else:
        return iaa.Sequential([
                    iaa.AdditiveGaussianNoise(scale=0.01 * 255, name="AWGN"),
                    iaa.GaussianBlur(sigma=(0.0, 3.0), name="Blur"),
                    iaa.Fliplr(0.5),
                    iaa.Add((-20, 20),name="Add"),
                    iaa.Multiply((0.8, 1.2), name="Multiply"),
                    iaa.Affine(scale=(0.8, 2.0)),
                    iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                    iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees
                ], random_order=True) 



def train(config, data_dir, train_anno, val_anno, model, 
            train_epochs=[20, 60, 100], custom_aug=False):
    # Training dataset. 
    dataset_train = TrashDataset()
    dataset_train.load_trash(data_dir, train_anno)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = TrashDataset()
    dataset_val.load_trash(data_dir, val_anno)
    dataset_val.prepare()

    # Image Augmentation
    # Right/Left flip 50% of the time
    # Image Augmentation Pipeline
    
    augmentation = get_augmentation_pipeline(custom=custom_aug)
    
    # *** This training schedule is an example. Update to your needs ***

    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=train_epochs[0],
            layers='heads',
            augmentation=augmentation)

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=train_epochs[1],
            layers='4+',
            augmentation=augmentation)

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=train_epochs[2],
            layers='all',
            augmentation=augmentation)



if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument('--data_dir', required=True,
                        metavar="/path/to/data/",
                        help='Directory of the dataset')
    parser.add_argument('--train_anno', required=True,
                        metavar="annotations.json",
                        help='Filename of annotations for training inside data_dir')
    parser.add_argument('--valid_anno', required=True,
                        metavar="annotations.json",
                        help='Filename of annotations for validation inside data_dir')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'last' or 'imagenet")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory')

    args = parser.parse_args()
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    config = TrashConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

    train(config, args.data_dir, args.train_anno, args.val_anno, model)

