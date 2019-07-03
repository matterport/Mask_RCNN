"""
Mask R-CNN
Train on the Clevr dataset.

------------------------------------------------------------

    # Train a new model starting from pre-trained COCO weights
    python3 clevr.py train --dataset=/path/to/clevr/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 clevr.py train --dataset=/path/to/clevr/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 clevr.py train --dataset=/path/to/clevr/dataset --weights=imagenet

    # Since the clevr dataset does not have mask by default, we have to generate masks by using the code at https://github.com/facebookresearch/clevr-dataset-gen
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from skimage.filters import threshold_mean
from skimage import color

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class ClevrConfig(Config):
    """Configuration for training on the clevr dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "clevr"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    # Background + (cube + sphere + cylinder) * (rubber + metal) *
    # (gray + blue + brown + yellow + red + green + purple + cyan) * (small + large)
    # NUM_CLASSES = 1 + (3*2*8*2)
    NUM_CLASSES = 1 + (3 * 2 * 8 * 2)

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    BACKBONE = "resnet50"


############################################################
#  Dataset
############################################################

class ClevrDataset(utils.Dataset):

    def load_clevr(self, dataset_dir, json_path):
        """Load a subset of the Clevr dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have 3 classes to add.
        shape_categories = {'cube': 1,
                            'sphere': 2,
                            'cylinder': 3}
        material_categories = {'rubber': 1,
                               'metal': 2}
        color_categories = {'gray': 1,
                            'blue': 2,
                            'brown': 3,
                            'yellow': 4,
                            'red': 5,
                            'green': 6,
                            'purple': 7,
                            'cyan': 8}
        size_categories = {'small': 1,
                           'large': 2}

        categories = {}
        temp_class_id = 1

        for shape in shape_categories:
            for mat in material_categories:
                for col in color_categories:
                    for size in size_categories:
                        class_name = shape + " " + mat + " " + col + " " + size
                        categories[class_name] = temp_class_id
                        temp_class_id = temp_class_id + 1

        print("Categories", categories)

        # Add classes
        for i in categories:
            self.add_class("clevr", categories[i], i)

        # Add Images
        annotations = json.load(open(json_path))
        scenes = annotations['scenes']

        for scene in scenes:
            image_path = os.path.join(dataset_dir, scene['image_filename'])
            class_ids = [categories[s['shape']
                                    + " " + s['material']
                                    + " " + s['color']
                                    + " " + s['size']] for s in scene['objects']]
            print("class_ids == ",class_ids)

            self.add_image(
                "clevr",
                image_id=scene['image_filename'],  # use file name as a unique image id
                path=image_path,
                class_ids=class_ids)

    def load_mask(self, image_id):

        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # Fetch Image Info from load_clevr
        image_info = self.image_info[image_id]

        mask = []
        mask_dir = image_info['path'][:-4] + '/mask/'

        for f in sorted(next(os.walk(mask_dir))[2]):
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir, f))
                m = color.rgb2gray(m)
                thresh = threshold_mean(m)
                m = m > thresh
                mask.append(m)
        mask = np.stack(mask, axis=-1)

        class_ids = image_info['class_ids']
        class_ids = np.array(class_ids, dtype=np.int32)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "clevr":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = ClevrDataset()
    dataset_train.load_clevr(args.dataset, args.json)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ClevrDataset()
    dataset_val.load_clevr(args.dataset, args.json)
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/clevr/dataset/",
                        help='Directory of the clevr dataset')
    parser.add_argument('--json', required=True,
                        metavar="/path/to/scenes/json",
                        help='Path to the scenes.json file')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("JSON: ", args.json)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = ClevrConfig()
    else:
        class InferenceConfig(ClevrConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=args.logs)

    # Select weights file to load
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

    # Train
    train(model)
