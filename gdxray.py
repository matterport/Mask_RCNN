"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import math
import time
import scipy
import numpy as np

import zipfile
import urllib.request
import shutil

from PIL import Image
from skimage import draw
from config import Config
from preprocessing import prepare_welding
import utils
import model as modellib
import numpy as np

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Classes
BACKGROUND_CLASS = 0
CASTING_DEFECT = 1
WELDING_DEFECT = 2

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"

DATASETS = {
    "Castings": "http://dmery.sitios.ing.uc.cl/images/GDXray/Castings.zip",
    "Welding": "http://dmery.sitios.ing.uc.cl/images/GDXray/Welds.zip"
}


# These layers change weights depending on the number of classes
EXCLUDE_LAYER_WEIGHTS = [
    'mrcnn_bbox_fc',      # [1024,324]  --> [1024,8]
    'mrcnn_class_logits', # [1024,2]    --> [1024,81]
    'mrcnn_mask',         # [1,1,256,2] --> [1,1,256,81]
]



############################################################
#  Configurations
############################################################


class TrainConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "gdxray"

    # We use a GPU with 11 GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # We classify weld defect and casting defect

    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 768

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 50

    # Non-max suppression threshold to filter RPN proposals.
    # You can reduce this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.5

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)


class InferenceConfig(TrainConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0


############################################################
#  Dataset
############################################################

class XrayDataset(utils.Dataset):
    """
    Dataset of Xray Images

    Images are referred to using their image_id (relative path to image).
    An example image_id is: "Castings/C0001/C0001_0004.png"
    """

    def load_gdxray(self, dataset_dir, subset, series, auto_download=False):
        """Load a subset of the GDXray dataset.
        dataset_dir: The root directory of the GDXray dataset.
        subset: What to load (train, test)
        series: If provided, only loads images that have the given classes ("Casting","Welding")
        auto_download: Automatically download and unzip GDXray images and annotations
        """
        if auto_download is True:
            self.auto_download(dataset_dir, series)

        castings_metadata = "metadata/gdxray/castings_{0}.txt".format(subset)
        welds_metadata = "metadata/gdxray/welds_{0}.txt".format(subset)

        if series=="Castings":
            metadata = [castings_metadata]

        if series=="Welds":
            metadata = [welds_metadata]

        if series=="All":
            metadata = [castings_metadata, welds_metadata]

        image_ids = []
        for metadata_path in metadata:
            with open(metadata_path,"r") as metadata_file:
                image_ids += metadata_file.readlines()
        # Strip all the newlines
        image_ids = [p.rstrip() for p in image_ids]

        boxes = self.load_boxes(dataset_dir, series)

        # Add classes
        self.add_class(source="gdxray", class_id=CASTING_DEFECT, class_name="Casting Defect")
        self.add_class(source="gdxray", class_id=WELDING_DEFECT, class_name="Welding Defect")

        # Add images
        for image_id in image_ids:
            path = os.path.join(dataset_dir,image_id)
            im = Image.open(path)
            width, height = im.size

            if not os.path.exists(self.get_mask_path(dataset_dir, image_id, 0)):
                print("Skipping ",image_id," Reason: No mask")
                continue

            print("Adding image:", image_id)

            self.add_image(
                "gdxray",
                image_id=image_id,
                path=path,
                width=width,
                height=height,
                dataset_dir=dataset_dir,
                annotations=boxes.get(image_id,[])
            )
            # self.create_mask(dataset_dir,image_id)


    def load_boxes(self, dataset_dir, series):
        """
        Create a map of bounding boxes for the series
        series: Castings or Welding

        returns:
            map["Castings/C0001/C0064_0001.png"] -> [[x1,x2,y1,y2],[x1,x2,y1,y2],...]
        """
        id_format = "{series}/{folder}/{folder}_{id:04d}.png"
        series_dir = os.path.join(dataset_dir, series)
        box_map = {}

        for root, dirs, files in os.walk(series_dir):
            for folder in dirs:
                metadata_file = os.path.join(root,folder,"ground_truth.txt")
                if os.path.exists(metadata_file):
                    for row in np.loadtxt(metadata_file):
                        row_id = int(row[0])
                        image_id = id_format.format(series=series,folder=folder,id=row_id)
                        box = [row[3],row[1],row[4],row[2]] # (y1, x1, y2, x2)
                        box_map.setdefault(image_id,[])
                        box_map[image_id].append(box)
                    # Mask R-CNN expects a numpy array of boxes
                    box_map[image_id] = np.array(box_map[image_id])
        return box_map


    def create_mask(self, dataset_dir, image_id):
        """Estimate instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        """
        info = self.get_image_info(image_id)
        print("Creating mask for ",image_id,info)

        for i,box in enumerate(info["annotations"]):
            # Convert to center and radius
            # Box dimensions: (y1, x1, y2, x2)
            center_x = (box[1]+box[3])/2
            center_y = (box[0]+box[2])/2
            r_x = math.ceil(abs(box[3]-box[1])/2)
            r_y = math.ceil(abs(box[2]-box[0])/2)
            # Make a bitmap mask
            mask = np.zeros((info["height"], info["width"]), dtype=np.uint8)
            rr, cc = draw.ellipse(center_y, center_x, r_y, r_x, shape=mask.shape)
            mask[rr, cc] = 1
            # Debugging
            #import matplotlib.pyplot as plt
            #image = scipy.ndimage.imread(info["path"])
            #image[rr, cc] = mask[rr, cc]*0.4
            #plt.imshow(image)
            #plt.imshow(mask)
            #plt.show()
            # Save image
            path = self.get_mask_path(dataset_dir, image_id, i)
            im = Image.fromarray(mask)
            im.save(path)


    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        masks = []
        info = self.image_info[image_id]
        image_id = info["id"]
        dataset_dir = info["dataset_dir"]

        for i in range(100):
            path = self.get_mask_path(dataset_dir, image_id, i)
            if os.path.exists(path):
                mask = scipy.ndimage.imread(path)
                mask = mask.astype(np.bool)
                masks.append(mask)
                # Debugging
                #import matplotlib.pyplot as plt
                #image = scipy.ndimage.imread(info["path"])
                #plt.figure()
                #plt.imshow(image)
                #plt.figure()
                #plt.imshow(mask)
                #plt.show()
            else:
                break
        mask = np.stack(masks,axis=-1)

        # Get defect type
        CLASS = CASTING_DEFECT
        if "weld" in image_id.lower():
            CLASS = WELDING_DEFECT
        class_ids = np.array([CLASS for _ in range(len(masks))], dtype=np.int32)

        return mask, class_ids


    def get_mask_path(self, dataset_dir, image_id, index):
        """Return the path to a mask"""
        image_file = os.path.join(dataset_dir, image_id)
        series_dir = os.path.dirname(image_file)
        mask_dir = os.path.join(series_dir,"masks")
        mask_name = os.path.basename(image_file).replace(".png","_%i.png"%index)
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        return mask_path


    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super().image_reference(image_id)


    def auto_download(self, dataset_dir, series):
        """Download and extract the zip file from GDXray

        dataset_dir: The directory to place the dataset
        series: The series to download "Castings, Welds, Both"
        """
        if series=="All":
            all_series = ["Castings","Welding"]
        else:
            all_series = [series]

        for series in all_series:
            url = DATASETS[series]

            zip_file = "{0}/{1}.zip".format(dataset_dir, series)
            series_dir = os.path.join(dataset_dir, series)

            # Make the dataset dir if it doesn't exist
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)

            # Download images if not available locally
            if not os.path.exists(series_dir):
                print("Downloading images to " + zip_file + " ...")
                with urllib.request.urlopen(url) as response, open(zip_file, 'wb') as out:
                    shutil.copyfileobj(response, out)
                print("... done downloading.")
                print("Unzipping " + zip_file + "...")
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(dataset_dir)
                print("... done unzipping")

                # Clean up
                print("Removing ",zip_file)
                os.remove(zip_file)

                mac_dir = os.path.join(dataset_dir,"__MACOSX")
                if os.path.exists(mac_dir):
                    print("Removing ",mac_dir)
                    shutil.rmtree(mac_dir)

        # Prepare the welding set
        weld_original = os.path.join(dataset_dir, "Welding")
        weld_processed = os.path.join(dataset_dir, "Welds")
        if os.path.exists(weld_original) and not os.path.exists(weld_processed):
            print("Preparing the welding data")
            prepare_welding()

        print("Finished downloading datasets")




############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"], r["masks"])
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--series', required=True,
                        metavar="<Casting | Welding | Both>",
                        help='GDXray series to extract and evaluate on')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=bool)
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Series: ", args.series)
    print("Logs: ", args.logs)
    print("Auto Download: ", args.download)

    # Expand user
    args.dataset = os.path.expanduser(args.dataset)

    # Configurations
    if args.command == "train":
        config = TrainConfig()
    else:
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True, exclude=EXCLUDE_LAYER_WEIGHTS)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = XrayDataset()
        dataset_train.load_gdxray(args.dataset, "train", series=args.series, auto_download=args.download)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = XrayDataset()
        dataset_val.load_gdxray(args.dataset, "test", series=args.series, auto_download=args.download)
        dataset_val.prepare()

        # *** This training schedule is an example. Update to your needs ***

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
                    epochs=160,
                    layers='all')

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = XrayDataset()
        coco = dataset_val.load_coco(args.dataset, "minival", year=args.year, return_coco=True, auto_download=args.download)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
