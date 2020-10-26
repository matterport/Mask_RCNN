import os
import sys 
import numpy as np
from pycocotools import mask as maskUtils
import imgaug 
import skimage
from matplotlib import pyplot as plt
import cv2
import time
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR) 
print(ROOT_DIR)

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
from angiodataset import AngioDataset
import json

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

"""Arrange resutls to match COCO specs in http://cocodataset.org/#format
"""
def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
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
                "category_id": dataset.get_source_class_id(class_id, "angio2020"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results

"""Runs official COCO evaluation.
dataset: A Dataset object with valiadtion data
eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
limit: if not 0, it's the number of images to use for evaluation
"""
def evaluate_coco(model, dataset, data, eval_type="bbox", limit=0, image_ids=None):
    # Pick images from the dataset
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

        class_names = ['BG', 'lad', 'diagonal', 'lcx1', 'lcx2', 'distal']
        # visualize.display_instances(, r['rois'], r['masks'], r['class_ids'], 
        #                     class_names, r['scores'])

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    results = data.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(data, results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


class AngioConfig(Config):

    NAME = 'angio2020'

    IMAGES_PER_GPU = 1

    IMAGE_CHANNEL_COUNT = 1

    MEAN_PIXEL = np.array([129.8])

    NUM_CLASSES = 1 + 5  # Background + rest

    STEPS_PER_EPOCH = 16432
    VALIDATION_STEPS = 317

    DETECTION_MIN_CONFIDENCE = 0.7

    BACKBONE = 'resnet101' 

    IMAGE_MAX_DIM = 512
    IMAGE_MIN_DIM = 512

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    MINI_MASK_SHAPE = (56, 56)

    LEARNING_RATE = 0.001

#Constants

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_INFERENCE_IMAGE_DIR = 'A:/val_images'

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Angio Dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' 'eval' or 'inference on angio dataset")
    parser.add_argument('--dataset', required=False,
                        default='A:/',
                        metavar="/path/to/angio/",
                        help="Directory of the dataset (default=A:/)")
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'imagenet'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--inference_datapath', required=False,
                        default=DEFAULT_INFERENCE_IMAGE_DIR,
                        metavar="/path/to/inference_images/",
                        help="Path to folder containing images to run prediction (default=A:/val_images)"
                        )
    parser.add_argument('--eval_data', required=False,
                        default='val',
                        metavar="<eval_data>",
                        help="which set of data to evaluate model on 'val' or 'train' or 'test' (default=val)"
                        )

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    IMAGE_DIR = args.inference_datapath
    mode = args.command
    datasetdir = args.dataset
    eval_data = args.eval_data

    if mode == 'train':
        config = AngioConfig()
        
    elif mode == 'inference' or mode == 'eval':
        class InferenceConfig(AngioConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.05
            BACKBONE = 'resnet101'

            # BACKBONE = modellib.mobilenet
            # COMPUTE_BACKBONE_SHAPE = modellib.ompute_mobilenet_shapes

        config = InferenceConfig()

    config.display()

    exclude = []
    # create model and load weights
    if mode =='train':
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
        if not args.model:
            weights_path = model.find_last()
        elif args.model == 'imagenet':
            exclude = ['conv1']
            weights_path =  model.get_imagenet_weights()
        elif args.model == 'coco':
            exclude = ['mrcnn_bbox_fc', 'mrcnn_class_logits', 'mrcnn_mask']
            weights_path = COCO_MODEL_PATH 
        else:
            weights_path = args.model
    elif mode == 'inference' or mode == 'eval':
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
        if not args.model:
            weights_path = model.find_last()
        else:
            weights_path = args.model


    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True, exclude=exclude)

    if mode == 'inference':
        class_names = ['BG', 'artery']

        # Load a random image from the images folder
        file_names = next(os.walk(IMAGE_DIR))[2]

        for name in file_names:
            if name.split('.')[-1] == 'jpeg':
                
                # load image
                pred_image = skimage.io.imread(os.path.join(IMAGE_DIR, name))
                image = skimage.color.gray2rgb(pred_image)

                # converts image to rgb if it is grayscale
                if pred_image.ndim != 3:
                    pred_image = np.expand_dims(pred_image, -1)
                else:
                    pred_image = skimage.color.rgb2gray(pred_image)
                    pred_image = np.expand_dims(pred_image, -1)
                
                # Run detection
                results = model.detect([pred_image], verbose=1)
                # Visualize results
                r = results[0]
                visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                            class_names, r['scores'], title=name.split('.')[0])
        
        # print(mean / cnt)
                                            
    elif mode == 'eval':
        dataset_val = AngioDataset()
        dataset_val.load_angio(datasetdir, eval_data)
        dataset_val.prepare()

        #load angio data in coco format as coco object
        data = COCO(datasetdir +  f'data_{eval_data}.json')

        evaluate_coco(model, dataset_val, data, "segm")

    elif mode == 'train':
        # train dataset
        dataset_train = AngioDataset()
        dataset_train.load_angio(datasetdir, 'train')
        dataset_train.prepare()

        # val dataset
        dataset_val = AngioDataset()
        dataset_val.load_angio(datasetdir, 'val')
        dataset_val.prepare()

        # augmentation
        augmentation = imgaug.augmenters.Sometimes(0.7, [
            imgaug.augmenters.Fliplr(0.5),
            imgaug.augmenters.Flipud(0.5),
            imgaug.augmenters.Rotate((-180, 180)),
            imgaug.augmenters.ShearX((-20, 20)),
            imgaug.augmenters.ShearY((-20, 20)),
            imgaug.augmenters.ElasticTransformation(alpha=(0, 70.0), sigma=(4.0, 6.0)),
            imgaug.augmenters.GammaContrast((0.5, 2.0)),
            imgaug.augmenters.PiecewiseAffine(scale=(0.01, 0.05)),
            imgaug.augmenters.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))
        ])


        # train heads
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2 
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune 4+")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=80,
                    layers='4+',
                    augmentation=augmentation)

        print("Fine tune 3+")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE * 3 / 10,
                    epochs=120,
                    layers='3+',
                    augmentation=augmentation)

        # Training - Stage 4
        # Finetune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all',
                    augmentation=augmentation)
