"""
Mask R-CNN
Train on the bdd100k dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained bdd100k weights
    python3 bdd100k.py train --dataset=/path/to/bdd100k/dataset --weights=bdd100k

    # Resume training a model that you had trained earlier
    python3 bdd100k.py train --dataset=/path/to/bdd100k/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 bdd100k.py train --dataset=/path/to/bdd100k/dataset --weights=imagenet

    # Apply color splash to an image
    python3 bdd100k.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 bdd100k.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
BDD100K_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_drivableArea.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class Bdd100kConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "drivableArea"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    #BACKBONE = "resnet50"
    BACKBONE = "resnet101"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + driving area

    # Input image resizing
    #IMAGE_MIN_DIM = 100
    #IMAGE_MAX_DIM = 128

    # Input image resizing
    IMAGE_MIN_DIM = 200
    IMAGE_MAX_DIM = 256

    # Input image resizing
    #IMAGE_MIN_DIM = 400
    #IMAGE_MAX_DIM = 512

    # Input image resizing as Original size
    #IMAGE_MIN_DIM = 720
    #IMAGE_MAX_DIM = 1280

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    #DETECTION_MIN_CONFIDENCE = 0.7

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Skip detections with < 90% confidence
    #DETECTION_MIN_CONFIDENCE = 0.95

    # Skip detections with < 90% confidence
    #DETECTION_MIN_CONFIDENCE = 0.99


############################################################
#  Dataset
############################################################

class Bdd100kDataset(utils.Dataset):

    def load_bdd100k(self, dataset_dir, subset):
        """Load a subset of the bdd100k dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("bdd100k", 1, "drivableArea")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # Label annotations for BDD100K are saved in the following format:
        # { "name": "0000f77c-6257be58.jpg",
        #   "attributes": { },
        #   "timestamp": ,
        #   "labels": [
        #     { "category": " ",
        #       "attributes": { },
        #       "manualShape": ,
        #       "manualAttributes": ,
        #       "box2d": { },
        #       "id": ,
        #     },
        #     { "category": "drivable area",
        #       "attributes": { },
        #       "manualShape": ,
        #       "manualAttributes": ,
        #       "poly2d": [
        #          { "vertices": [
        #               [ ],
        #               [ ],
        #               [ ],
        #               ...,
        #              ],
        #            "types": " ",
        #            "closed": ,
        #          }
        #        ],
        #       "id": ,
        #     }
        #   ]
        # }
        # We mostly care about the x and y coordinates from poly2d
        annotations_dir = "{}/../../../labels/bdd100k_labels_images_{}.json".format(dataset_dir, subset)
        annotations = json.load(open(annotations_dir))
        # annotations is a list of dictionary corresponding to each image
        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the bbox that make up
            # the outline of each object instance. These are stores in poly2d
            polygons = [d['poly2d'] for d in a['labels']
                         if d['category'][:4] == 'area' or d['category'] == 'drivable area']
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['name'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "bdd100k",
                image_id=a['name'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a bdd100k dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "bdd100k":
            return super(self.__class__, self).load_mask(image_id)

        # Convert bbox to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            x = []
            y = []
            for d in p:
                for l in d['vertices']:
                    x.append(l[0]-2)
                    y.append(l[1]-2)
            rr, cc = skimage.draw.polygon(y, x)
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "bdd100k":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = Bdd100kDataset()
    dataset_train.load_bdd100k(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = Bdd100kDataset()
    dataset_val.load_bdd100k(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    with tf.device("/gpu:0"):
        
	# Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=200,
                    layers='all')
        '''
        # Training - Stage 1
        print("Training network heads and lr=0.001")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=200,
                    layers='heads')
        
        # Training - Stage 2
        print("Training network heads and lr=0.0001")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=200,
                    layers='heads')
        
        # Training - Stage 3
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up and lr=0.001")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=200,
                    layers='4+')
        
        # Training - Stage 4
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up and lr=0.0001")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=200,
                    layers='4+')
        
        # Training - Stage 5
        # Fine tune all layers
        print("Fine tune all layers and lr=0.001")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=200,
                    layers='all')
        
        # Training - Stage 6
        # Fine tune all layers
        print("Fine tune all layers and lr=0.0001")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=200,
                    layers='all')
        '''


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        input_image_path, input_image_name = os.path.split(image_path)
        # file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        file_name = "splash_{}".format(input_image_name)
        output_path = os.path.join(input_image_path, "../splash", file_name)
        skimage.io.imsave(output_path, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", output_path)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect bdd100k.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/bdd100k/dataset/",
                        help='Directory of the bdd100k dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'bdd100k'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = Bdd100kConfig()
    else:
        class InferenceConfig(Bdd100kConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        with tf.device("/gpu:0"):
            model = modellib.MaskRCNN(mode="training", config=config,
                                      model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "bdd100k":
        weights_path = BDD100K_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "bdd100k":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or splash
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
