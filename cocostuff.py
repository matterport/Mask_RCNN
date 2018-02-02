"""
Adapted for the COCO Stuff dataset from the base COCO code.
https://github.com/nightrome/cocostuff

Tested with COCO Stuff 2017 from
http://cocodataset.org

Note: This class should most likely eventually disappear, once all differences
between the coco and stuff datasets are bridged.

Note: This script relies on the nightrome fork of cocoapi:
https://github.com/nightrome/cocoapi
"""

import os
import time

from pycocotools import coco as pycoco
from pycocotools.cocostuffeval import COCOStuffeval

# import zipfile
# import urllib.request
# import shutil

import coco
import model as modellib


# Root directory of the project
ROOT_DIR = os.getcwd()
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


class CocoStuffConfig(coco.CocoConfig):
    """ Subclass CocoConfig to set the number of classes
    to also include stuff classes
    """
    NAME = "coco_stuff"
    NUM_CLASSES = 183  # 1 + 91 + 92


class CocoStuffDataset(coco.CocoDataset):
    """ There are only small differences in the stuff dataset,
    so adapt for them here.
    """
    def load_cocostuff(self, dataset_dir, subset, year, class_ids=None):
        """ Load a subset of the COCO Stuff dataset
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        """
        # TODO: the stuff_ prefix should just be configurable in CocoDataset
        coco = pycoco.COCO(os.path.join(
            dataset_dir,
            "annotations",
            "stuff_{}{}.json".format(subset, year)
        ))

        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = os.path.join(
            dataset_dir,
            "{}{}".format(subset, year))

        # Load all classes or a subset?
        if not class_ids:
            # This has a side effect on self.class_info.
            # Note we do not extend class_ids here.
            # Below load_cats will try to find the classes in
            # the coco api (only loaded with "stuff"), which doesn't have them.
            # It would be much better if all categories are present
            # in the stuff set, or kept in a separate file.
            self.load_coco_categories(
                dataset_dir, subset, year)

            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            # The source has to remain coco,
            # otherwise load_mask fails
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))

        return coco

    def load_coco_categories(self, dataset_dir, subset, year, class_ids=None):
        """ Load coco categories and merge them with the stuff ones
        This is useful for inference on a model trained also on coco,
        otherwise you might get unknown index errors.
        It would be much more efficient to keep all categories in a separate file,
        or all categories in the stuff dataset...
        """
        coco = pycoco.COCO(os.path.join(
            dataset_dir,
            "annotations",
            "instances_{}{}.json".format(subset, year)
        ))

        if not class_ids:
            class_ids = sorted(coco.getCatIds())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        return class_ids

    def download(self, dataDir, imgDir):
        # http://images.cocodataset.org/annotations/annotations_trainval2017.zip
        # http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
        # http://images.cocodataset.org/zips/train2017.zip
        # TODO
        pass


def evaluate_cocostuff(model, dataset, coco_api, eval_type="bbox", limit=0, image_ids=None):
    image_ids = image_ids or dataset.image_ids

    if limit:
        image_ids = image_ids[:limit]

    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=1)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        image_results = coco.build_coco_results(dataset, coco_image_ids[i:i + 1],
                                                r["rois"], r["class_ids"],
                                                r["scores"], r["masks"])
        print(image_results)
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco_api.loadRes(results)

    # Evaluate
    cocoEval = COCOStuffeval(coco_api, coco_results)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on COCO Stuff.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on COCO Stuff")
    parser.add_argument("--dataset", required=True,
                        metavar="/path/to/cocostuff/",
                        help="Directory of the COCO Stuff dataset")
    parser.add_argument("--year", required=False,
                        default="2014",
                        metavar="<year>",
                        help="Year of the COCO Stuff dataset")
    parser.add_argument("--model", required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument("--logs", required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help="Logs and checkpoints directory (default=logs/)")
    parser.add_argument("--limit", required=False,
                        default=500,
                        metavar="<image count>",
                        help="Number of images to use for evaluation (default=500)")
    parser.add_argument("--download", required=False,
                        default=False,
                        metavar="<True|False>",
                        help="Automatically download and unzip COCO Stuff files (default=False)",
                        type=bool)

    args = parser.parse_args()

    if args.command == "train":
        config = CocoStuffConfig()
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        config = CocoStuffConfig()
        # evaluate images one by one
        config.BATCH_SIZE = 1
        config.IMAGES_PER_GPU = 1
        config.DETECTION_MIN_CONFIDENCE = 0
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    config.display()

    if args.model.lower() == "last":
        model_path = model.find_last()[1]
    else:
        model_path = args.model

    if model_path and model_path != "":
        print("Loading weights from {}".format(model_path))
        model.load_weights(model_path, by_name=True)
    else:
        print("Not loading initial weights!")

    if args.command == "train":
        dataset_train = CocoStuffDataset()
        dataset_train.load_cocostuff(
            args.dataset,
            "train",
            args.year)
        dataset_train.prepare()

        dataset_val = CocoStuffDataset()
        dataset_val.load_cocostuff(
            args.dataset,
            "val",
            args.year)
        dataset_val.prepare()

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1 - heads
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
                    epochs=160,
                    layers='all')

    elif args.command == "evaluate":
        dataset_val = CocoStuffDataset()
        coco_api = dataset_val.load_cocostuff(args.dataset, "val", args.year)
        dataset_val.prepare()

        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_cocostuff(model, dataset_val, coco_api, "segm", limit=int(args.limit))
    else:
        print("Unknown command {}".format(args.command))
