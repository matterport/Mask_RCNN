import os
import sys
import json
import numpy as np
import skimage.draw
import collections

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils


############################################################
#  Configurations
############################################################

class DetConfig(Config):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "wolf"

    # A GPU with 12GB memory can fit two images.
    IMAGES_PER_GPU = 2

    #CLASS_NAMES = ['sign', 'yield_sign', 'stop_sign', 'oneway_sign', 'donotenter_sign', 'wrongway_sign']
    CLASS_NAMES = ['wolf']

    ALL_CLASS_NAMES = ['BG'] + CLASS_NAMES

    # Number of classes (including background)
    NUM_CLASSES = len(ALL_CLASS_NAMES)

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100    # TODO was 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9  # TODO was 90

    TRAINING_VERBOSE = 1

    TRAIN_BN = False
    #  'relu' or 'leakyrelu'
    ACTIVATION = 'relu'


############################################################
#  Dataset
############################################################

# Example usage:
# Load images from a training directory.
#    dataset_train = DetDataset()
#    dataset_train.load_dataset_images(dataset_path, "train", config.CLASS_NAMES)

# Alternatively use the convenience function for taking from one directory and spliting into train and test
#   dataset_train, dataset_val = create_datasets(dataset_path+'/train', config.CLASS_NAMES)


class DetDataset(utils.Dataset):

    def load_by_annotations(self, dataset_dir, annotations_list, class_ids):
        """Load a specific set of annotations and from them images.
        dataset_dir: Root directory of the dataset.
        annotations_list: The annotations (and images) to be loaded.
        class_ids: List of classes to use.
        """
        # Add classes.
        for i, name in enumerate(class_ids):
            # Skip over background if it appears in the class name list
            index = i + 1
            if name != 'BG':
                print('Adding class {:3}:{}'.format(index, name))
                self.add_class('wolf', index, name)

        # Add images
        for a in annotations_list:
            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "wolf",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_dataset_images(self, dataset_dir, subset, class_ids):
        """Load a subset of the dataset. This function expects train and val directories.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        class_names: List of classes to use.
        """

        # Train or validation dataset?
        assert subset in ["train", "val"]

        print(dataset_dir)
        print(subset)

        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        annotations = json.load(open(os.path.join(dataset_dir, "annotations.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Find the unique classes and track their count
        uniq_class_names = collections.Counter()
        for a in annotations:
            for id, region in a['regions'].items():
                object_name = region['region_attributes']['object_name']
                uniq_class_names[object_name] += 1

        print(uniq_class_names)

        # Add classes.
        for i, name in enumerate(class_ids):
            # Skip over background if it occurs in the
            index = i + 1
            if name != 'BG':
                print('Adding class {:3}:{}'.format(index, name))
                self.add_class('wolf', index, name)

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "wolf",
                image_id=a['filename'],  # use file name as a unique image id
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

        # If not an object in our dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] not in self.class_names:
            print("warning: source {} not part of our classes, delegating to parent.".format(image_info["source"]))
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            try:
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                mask[rr, cc, i] = 1
            except:
                print(image_info)
                raise

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "wolf":
            return info["path"]
        else:
            print("warning: DetDataSet: using parent image_reference")
            super(self.__class__, self).image_reference(image_id)


def split_annotations(dataset_dir, train_pct=.8, annotation_filename="annotations.json"):
    """ divide up an annotation file for training and validation
    dataset_dir: location of images and annotation file.
    train_pct: the split between train and val default is .8
    annotation_filename: name of annotation file.
    """

    # Load annotations
    annotations = json.load(open(os.path.join(dataset_dir, annotation_filename)))
    annotations = list(annotations.values())  # don't need the dict keys

    # The VIA tool saves images in the JSON even if they don't have any
    # annotations. Skip unannotated images.
    annotations = [a for a in annotations if a['regions']]

    # Find the unique classes and track their count
    uniq_class_names = collections.Counter()
    for a in annotations:
        for id, region in a['regions'].items():
            object_name = region['region_attributes']['object_name']
            uniq_class_names[object_name] += 1

    n_annotations = len(annotations)

    # Randomize the annotations then divide
    np.random.shuffle(annotations)

    # Divide between training and validation
    n_for_train = int(n_annotations*train_pct)
    train_ann = annotations[:n_for_train]
    val_ann = annotations[n_for_train:]

    def validate_unique(ann, img_files={}):
        for a in ann:
            filename = a['filename']
            if filename in img_files:
                raise RuntimeError(filename+' already exists')
            else:
                img_files[filename] = 1
        return img_files

    img_files = validate_unique(train_ann)
    img_files = validate_unique(val_ann, img_files)
    assert len(train_ann)+len(val_ann)  == len(img_files)

    return train_ann, val_ann, uniq_class_names


def create_datasets(dataset_dir, class_ids, train_pct=.8):
    """ set up the training and validation trainng set.
    dataset_dir: location of images and annotation file.
    class_ids: list of classes that being trained for.
    train_pct: the split between train and val default is .8
    """

    train_ann, val_ann, object_counts = split_annotations(dataset_dir, train_pct=train_pct)

    train_ds = DetDataset()
    train_ds.load_by_annotations(dataset_dir, train_ann, class_ids)

    val_ds = DetDataset()
    val_ds.load_by_annotations(dataset_dir, val_ann, class_ids)

    assert len(train_ds.image_info) == len(train_ann) and len(val_ds.image_info) == len(val_ann)

    return train_ds, val_ds


