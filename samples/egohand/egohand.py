import os
import numpy as np
import scipy.io as sio
import gc
import six.moves.urllib as urllib
import cv2
import time
import xml.etree.cElementTree as ET
import random
import shutil as sh
from shutil import copyfile
import skimage.draw
import imgaug
import csv
from utils import Dataset
from config import Config
from model import MaskRCNN
from visualize import display_instances
import utils
from utils import extract_bboxes, download_trained_weights
from sklearn import metrics
import matplotlib.pyplot as plt

# download ehohand dataset from http://vision.soic.indiana.edu/egohands_files/egohands_data.zip
# website
# Each activity contain four folder of images, manually split train and test dataset
# Folder ends with 'B_T', 'H_S' and 'S_H'  are taken as training dataset
# Folder ends with 'T_B' are taken as test dataset
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################
# define the prediction configuration
class EgohandConfig(Config):
    # define the name of the configuration
    NAME = "Egohand"

    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 4

    # simplify GPU config
    GPU_COUNT = 1

    # We use a GPU, which can fit one images.
    # Adjust if you use a bigger GPU
    IMAGES_PER_GPU = 1


############################################################
#  Dataset
############################################################
class EgohandDataset(utils.Dataset):

    def load_dataset(self, dataset_dir, subset):
        """Load a subset of the egohand dataset.
        """

        # define one class
        self.add_class("Egohand", 1, "myleft")
        self.add_class("Egohand", 2, "myright")
        self.add_class("Egohand", 3, "yourleft")
        self.add_class("Egohand", 4, "yourright")

        # Train or validation dataset?
        assert subset in ["train", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # list all directories
        directories = os.listdir(dataset_dir)

        # grab all images from each directory
        for i in range(len(directories)):
            image_path_array = []
            print(os.path.join(dataset_dir, directories[i]))
            for roots, dirs, files in os.walk(os.path.join(dataset_dir, directories[i])):
                for file in files:
                    if file.endswith(".jpg"):
                        file = os.path.join(roots, file)
                        image_path_array.append(file)


                # sort image_path_array to ensure its in the low to high order expected in polygon.mat
                image_path_array.sort()
                print(os.path.join(dataset_dir, directories[i]) + "\polygons.mat")
                boxes = sio.loadmat(os.path.join(dataset_dir, directories[i]) + "\polygons.mat")
                # there are 100 of these per folder in the egohands dataset
                polygons = boxes["polygons"][0]
                hands = ['myleft', 'myright', 'yourleft', 'yourright']
                # find all images
                for img in range(len(image_path_array)):
                    annotation = []
                    class_id = []
                    folder, filename = os.path.split(image_path_array[img])
                    image = cv2.imread(image_path_array[img])
                    height, width = image.shape[:2]
                    cindex = 0
                    for pointlist in polygons[img]:
                        if pointlist.size == 0:
                            print("....")
                        else:
                            annotation.append(pointlist)
                            class_id.append(hands[cindex])
                        cindex += 1
                    # add to dataset
                    self.add_image('Egohand', image_id=filename[:-4], path=image_path_array[img], width=width, height=height, annotation=annotation, class_ids=class_id)
        print("done")
        return

    # load the masks for an image
    def load_mask(self, image_id):
        """Generate instance masks for an image.
               Returns:
                masks: A bool array of shape [height, width, instance count] with
                    one mask per instance.
                class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "Egohand":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["annotation"])], dtype=np.uint8)
        class_ids = list()
        print(len(info["annotation"]))
        for i, (pt, classes) in enumerate(zip(info["annotation"], info["class_ids"])):
            # Get indexes of pixels inside the polygon and set them to 1
            x = []
            y = []
            for point in pt:
                if (len(point) == 2):
                    x.append(int(point[0]))
                    y.append(int(point[1]))
            rr, cc = skimage.draw.polygon(y, x)
            mask[rr, cc, i] = 1
            class_ids.append(self.class_names.index(classes))
        print(np.asarray(class_ids, dtype='int32'))
        return mask.astype(np.bool), np.asarray(class_ids, dtype='int32')
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "Egohand":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = EgohandDataset()
    dataset_train.load_dataset(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = EgohandDataset()
    dataset_val.load_dataset(args.dataset, "test")
    dataset_val.prepare()

    # Image Augmentation
    # rotate
    # augmentation = imgaug.augmenters.Affine(rotate=(-15, 15), cval=192, mode='constant')

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')   # , augmentation=augmentation

    new_history = model.keras_model.history.history
    for k in new_history:
        history[k] = history[k] + new_history[k]
        epochs = range(1, len(next(iter(history.values()))) + 1)
        plt.figure(figsize=(17, 5))

        plt.subplot(131)
        plt.plot(epochs, history["loss"], label="Train loss")
        plt.plot(epochs, history["val_loss"], label="Valid loss")
        plt.legend()
        plt.subplot(132)
        plt.plot(epochs, history["mrcnn_class_loss"], label="Train class ce")
        plt.plot(epochs, history["val_mrcnn_class_loss"], label="Valid class ce")
        plt.legend()
        plt.subplot(133)
        plt.plot(epochs, history["mrcnn_bbox_loss"], label="Train box loss")
        plt.plot(epochs, history["val_mrcnn_bbox_loss"], label="Valid box loss")
        plt.legend()

        plt.show()

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect Hand.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/Egohand/dataset/",
                        help='Directory of the Egohand dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("command:", args.command)

    # Configurations
    if args.command == "train":
        config = EgohandConfig()
    else:
        class InferenceConfig(EgohandConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            download_trained_weights(weights_path)
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
        model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "detect":
        predicted = detect(model, args.dataset, args.subset)
        print(metrics.accuracy_score(y_test, predicted))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
