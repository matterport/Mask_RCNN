"""
Mask R-CNN
Train a driveable area classifer based on mask-rcnn.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 driving_area.py train --dataset=/path/to/bdd/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 driving_area.py train --dataset=/path/to/bdd/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 driving_area.py train --dataset=/path/to/bdd/dataset --weights=imagenet

    # Apply driveable area filter to an image
    python3 driving_area.py detect --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply driveable area filter filter  to video using the last weights you trained
    python3 driving_area.py detect --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import glob
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DATA_DIR = os.path.abspath("../../../data")
import random

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from imgaug import augmenters as iaa
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Path to trained weights file
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class DriveableConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "driveable"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 2 + 1  # Current_lane + potential_lane + un-driveable

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1750
    #VALIDATION_STEPS = 200
    LEARNING_RATE = 0.005
    #BACKBONE = "resnet50"
    USE_MINI_MASK = False
    GPU_COUNT = 2

    # Skip detections with < 80% confidence
    DETECTION_MIN_CONFIDENCE = 0.80

    ############################################################
#  Dataset
############################################################

class DriveableDataset(utils.Dataset):

    def load_driveable(self, dataset_dir, subset):
        """Load a subset of the bdd dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("driveable", 1, "current")
        self.add_class("driveable", 2, "alternative")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        files = glob.glob(dataset_dir+"/*.*")
        if(subset=="train"):
            for f in files:
                height, width = 720,1280
                self.add_image(
                    source = "driveable",
                    image_id = os.path.splitext(os.path.basename(f))[0],
                    path = f,
                    width = width, height = height
                )
        else:
            files_sub = random.sample(files,1200)
            for f in files_sub:
                height, width = 720,1280
                self.add_image(
                    source = "driveable",
                    image_id = os.path.splitext(os.path.basename(f))[0],
                    path = f,
                    width = width, height = height
                )

    
    def load_mask(self, image_id, subset='train/'):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a bdd dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "driveable":
            return super(self.__class__, self).load_mask(image_id)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        mask_dir = os.path.join(DATA_DIR,"drivable_maps/color_labels/")
        #assert subset in ["train", "val"]
        mask_dir = os.path.join(mask_dir, "train/")
        #info = self.image_info[image_id]
        info = str(image_info["id"]) + '_drivable_color.png'
        img = os.path.join(mask_dir,info)
        #print("img: ",img)
        mask = cv2.imread(img)
        mask = np.delete(mask,1,2)
        mask = mask[:,:,::-1]
        if (mask[:,:,0].sum()==0 and mask[:,:,1].sum()>0):
            mask = np.delete(mask,0,2)
            return mask.astype(bool),np.asarray([2], dtype=np.int32)
        if (mask[:,:,1].sum()==0 and mask[:,:,0].sum()>0):
            mask = np.delete(mask,1,2)
            return mask.astype(bool),np.asarray([1], dtype=np.int32)
        if (mask[:,:,1].sum()>0 and mask[:,:,0].sum()>0):
            return mask.astype(np.bool), np.asarray([1,2], dtype=np.int32)
        else:
            return mask.astype(np.bool), np.asarray([1,2], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "driveable":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = DriveableDataset()
    dataset_train.load_driveable(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DriveableDataset()
    dataset_val.load_driveable(args.dataset, "val")
    dataset_val.prepare()
    
    augmentation = iaa.Fliplr(0.25)

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=500,
                layers="all",
                augmentation=augmentation)
    
def color_filter(image, mask):
    """Apply color filter effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    #gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    gray = np.zeros(image.shape)
    gray[:,:] = (0,255,0)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        driveable_mask = np.where(mask, (0.35*gray) + (0.65*image), image).astype(np.uint8)
    else:
        driveable_mask = gray.astype(np.uint8)
    return driveable_mask


def detect_and_mask_driveable(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color filter effect
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color filter
        driveable_mask = color_filter(image, r['masks'])
        # Save output
        file_name = "detect_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, driveable_mask)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "driveable_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            if(count>3000):
                break
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color filter
                driveable_region = color_filter(image, r['masks'])
                # RGB -> BGR to save image to video
                driveable_region = driveable_region[..., ::-1]
                # Add image to video writer
                vwriter.write(driveable_region)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect Driveable area.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/bdd/dataset/",
                        help='Directory of the BDD dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color filter effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color filter effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.image or args.video,\
               "Provide --image or --video to apply color filter"

    #print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = DriveableConfig()
    else:
        class InferenceConfig(DriveableConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    #Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
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
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)
    

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "detect":
        detect_and_mask_driveable(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))

