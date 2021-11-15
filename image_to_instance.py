from mrcnn import visualize

import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from mrcnn import visualize

import zipfile
import urllib.request
import shutil

import matplotlib.pyplot as plt

### Root directory of the project
ROOT_DIR = os.path.abspath("../")
CURRENT_DIR = os.path.abspath("./")

### Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

### Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

### Directory to save logs and model checkpoints, if not provided
### through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(CURRENT_DIR, "output_maeng/logs")
DEFAULT_DATASET_YEAR = ""

########################################################################################################################
#  Configurations
########################################################################################################################


class Deetas_Config(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    ### Give the configuration a recognizable name
    NAME = "deetas"

    ### We use a GPU with 12GB memory, which can fit two images.
    ### Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    ### Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    ### Number of classes (including background)
    NUM_CLASSES = 1 + 25  # Deetas has 25 classes


########################################################################################################################
#  Dataset
########################################################################################################################

class Deetas_Dataset(utils.Dataset):
    def load_deetas(self, dataset_dir, subset, class_ids=None,
                  class_map=None, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        ### coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        # json_deetas = COCO("{}/annotations/seg_{}.json".format(dataset_dir, subset)) # annotation path
        json_deetas = COCO("{}/sample_21_10_21/N-B-C-008.json".format(dataset_dir)) # sample
        
        image_dir = "{}/data_21_10_21/image".format(dataset_dir)
        print(image_dir)

        ### Load all classes or a subset?
        if not class_ids:
            ### All classes
            class_ids = sorted(json_deetas.getCatIds())

        ### All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(json_deetas.getImgIds(catIds=[id])))
            ### Remove duplicates
            image_ids = list(set(image_ids))
        else:
            ### All images
            image_ids = list(json_deetas.imgs.keys())

        ### Add classes
        for i in class_ids:
            self.add_class("coco", i, json_deetas.loadCats(i)[0]["name"])

        ### Add images
        for i in image_ids:
            self.add_image(
                "json_deetas", image_id=i,
                path=os.path.join(image_dir, json_deetas.imgs[i]['file_name']),
                width=json_deetas.imgs[i]["width"],
                height=json_deetas.imgs[i]["height"],
                annotations=json_deetas.loadAnns(json_deetas.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return json_deetas

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
        ### If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(Deetas_Dataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        ### Build mask of shape [height, width, instance_count] and list
        ### of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                ### Some objects are so small that they're less than 1 pixel area
                ### and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                ### Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    ### For crowd masks, annToMask() sometimes returns a mask
                    ### smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        ### Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            ### Call super class to return an empty mask
            return super(Deetas_Dataset, self).load_mask(image_id)

    ### The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            ### polygon -- a single object might consist of multiple parts
            ### we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            ### uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            ### rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


########################################################################################################################
#  COCO Evaluation
########################################################################################################################

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


def generate_mask(model, dataset_class, deetas_data, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    idx_image_ndarray = image_ids or dataset_class.image_ids


    # Limit to a subset
    if limit:
        idx_image_ndarray = idx_image_ndarray[:limit]

    # Get corresponding COCO image IDs.
    idx_image_list = [dataset_class.image_info[id]["id"] for id in idx_image_ndarray]

    t_prediction = 0
    t_start = time.time()

    results_list = []
    print("idx_image_ndarray", len(idx_image_ndarray))
    for idx_for, image_id in enumerate(idx_image_ndarray):
        print("idx_for :", idx_for)
        if idx_for >= 100:
            break
        ### Load image
        image = dataset_class.load_image(image_id)


        ### Run detection
        t = time.time()
        detections = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)
        
        # exit()

        boxes = detections["rois"]
        masks = detections["masks"]
        class_ids = detections["class_ids"]
        

        names = {}
        with open('/home/dblab/maeng_space/dataset/deetas/class_list/deetas_with_background.names', 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        deetas_names_dict = names
        
        name_list = []
        for key in names:
            name_list.append(names[key])

        class_names = name_list
        # print(class_ids, class_names)
        

        save_dir = '/home/dblab/maeng_space/git_repository/object_detector/Mask_RCNN/output_maeng/detection/ex_02'
        ### apply_mask
        # color = visualize.random_colors(1)
        # image_with_mask = visualize.apply_mask(image, mask_ndarray, color)
        masked_image = visualize.display_instances(image, boxes, masks, class_ids, class_names)

        # plt.imsave(save_dir+str(idx_for)+'.jpeg', masked_image)
            
        plt.savefig("{}/{}.png".format(save_dir, dataset_class.image_info[image_id]["id"]))

        # print(type(masked_image), masked_image.shape)

        
        ### Convert results to COCO format
        ### Cast masks to uint8 because COCO tools errors out on bool
        # image_results = build_coco_results(dataset_class, idx_image_list[idx_for:idx_for + 1],
        #                                    detections["rois"], detections["class_ids"],
        #                                    detections["scores"],
        #                                    detections["masks"].astype(np.uint8))
        # results_list.extend(image_results)
        

    # print(len(results_list))

    ### Load results. This modifies results with additional attributes.
    # deetas_results = deetas_data.loadRes(results_list)
    # print(deetas_results)

    ### Evaluate
    # cocoEval = COCOeval(deetas_data, coco_results, eval_type)
    # cocoEval.params.imgIds = idx_image_list
    # cocoEval.evaluate()
    # cocoEval.accumulate()
    # cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(idx_image_ndarray)))
    print("Total time: ", time.time() - t_start)


########################################################################################################################
#  Training
########################################################################################################################


if __name__ == '__main__':
    import argparse

    ### Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        default='generate_mask',
                        metavar="<command>",
                        help="only generate_mask")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the any dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=100,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    ### Configurations
    class InferenceConfig(Deetas_Config):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0
    config = InferenceConfig()
    config.display()

    ### Create model
    model = modellib.MaskRCNN(mode="inference", config=config,
                                model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    ### Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    ### Train or evaluate

    if args.command == "generate_mask":
        ### Validation dataset
        input_dataset = Deetas_Dataset()
        deetas_data = input_dataset.load_deetas(args.dataset, "test", return_coco=True)
        input_dataset.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        generate_mask(model, input_dataset, deetas_data, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
