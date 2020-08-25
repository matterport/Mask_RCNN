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

from mrcnn import utils
from mrcnn import visualize
import json

<<<<<<< HEAD
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

        class_names = ['BG', 'artery']
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



=======
>>>>>>> 55e3eb8e2f70958126214ffc51d2e51e82fd2ce2
class AngioDataset(utils.Dataset):

    def fetchAnnotations(self, image_id, annotations):
        matched = []
        for annotation in annotations:
            if annotation['image_id'] == image_id:
                matched.append(annotation)
        return matched

    def load_angio(self, datasetdir, subset='train'):

        # load annotation file
        #TODO separate train and val data json with data_train & data_val
        with open('{}/{}'.format(datasetdir, 'data_{}.json'.format(subset))) as f:
            angio_annotation = json.load(f)

        # load all class_ids
        class_ids = []
        categories = angio_annotation['categories']
        for category in categories:
            class_ids.append(category['id'])
            self.add_class('angio2020', category['id'], category['name'])

        # add image to dataset
        images = angio_annotation['images']
        for image in images:
            self.add_image(
                'angio2020', 
                image_id=image['id'], 
                path='{}/{}/{}'.format(datasetdir, subset, image['filename']),
                width=image['width'],
                height=image['height'],
                annotations=self.fetchAnnotations(image['id'], angio_annotation['annotations'])
                )

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info['source'] == 'angio2020':
            return info['path']
        else:
            super(AngioDataset, self).image_reference(image_id)

    def annToMask(self, rle):
        """
        Converts RLE annotation to binary mask.
        """
        mask = maskUtils.decode(rle)
        return mask
    
    """Generate instance masks for an image.
    Returns:
    masks: A bool array of shape [height, width, instance count] with
        one mask per instance.
    class_ids: a 1D array of class IDs of the instance masks.
    """
    def load_mask(self, image_id):
        # If not an angio dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info['source'] != 'angio2020':
            return super(AngioDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        # add instance mask and class ids for each rle
        for annotation in annotations:
            # converts rle to mask
            mask = self.annToMask(annotation['segmentation'])
            class_id = annotation['category_id']

            instance_masks.append(mask)
            class_ids.append(class_id)

        # pack everything into an array and return
        mask = np.stack(instance_masks, axis=2).astype(np.bool)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids

    """
        Overides original load_image function to load the specified image and return a [H,W,1] Numpy array.
    """
    def load_image(self, image_id):
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        
        #Convert all to grayscale and expand depth dimension for consistency.
        image = np.expand_dims(skimage.color.rgb2gray(image), -1)

        return image
<<<<<<< HEAD

class AngioConfig(Config):

    NAME = 'angio2020'

    IMAGES_PER_GPU = 1

    IMAGE_CHANNEL_COUNT = 1

    MEAN_PIXEL = np.array([100.0])

    NUM_CLASSES = 1 + 1  # Background + artery

    STEPS_PER_EPOCH = 1500
    VALIDATION_STEPS = 300

    DETECTION_MIN_CONFIDENCE = 0.7

    BACKBONE = 'resnet101' 

    IMAGE_MAX_DIM = 512

    MAX_GT_INSTANCES = 100

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2, 3]
    MINI_MASK_SHAPE = (56, 56)

mode = 'inference'
last = True
datasetdir = 'A:/'

if mode == 'train':
    config = AngioConfig()
    
elif mode == 'inference' or mode == 'eval':
    class InferenceConfig(AngioConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.3
        BACKBONE = 'resnet101'

        # BACKBONE = modellib.mobilenet
        # COMPUTE_BACKBONE_SHAPE = modellib.compute_mobilenet_shapes

    config = InferenceConfig()

config.display()

# create model and load weights
if mode =='train':
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=os.path.join(ROOT_DIR, "logs"))
    if last:
        weights_path = model.find_last()
    else:
        weights_path = model.get_imagenet_weights()
elif mode == 'inference' or mode == 'eval':
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=os.path.join(ROOT_DIR, "logs"))
    weights_path = model.find_last()


print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True, exclude=['conv1'])

if mode == 'inference':
    class_names = ['BG', 'artery']

    # Load a random image from the images folder
    IMAGE_DIR = 'A:/train_images'
    file_names = next(os.walk(IMAGE_DIR))[2]

    for name in file_names:
        if name.split('.')[-1] == 'jpeg':

            # Load image
            image = skimage.io.imread(os.path.join(IMAGE_DIR, name))
            
            #Convert all to grayscale for consistency. This image is used for prediction
            pred_image = np.expand_dims(skimage.color.rgb2gray(image), -1)
            print(pred_image)

            # converts image to rgb for visualisation
            vis_image = skimage.color.gray2rgb(image)
            
            # Run detection
            results = model.detect([pred_image], verbose=1)
            # Visualize results
            r = results[0]
            # print(r['masks'].max())
            visualize.display_instances(vis_image, r['rois'], r['masks'], r['class_ids'], 
                                        class_names, r['scores'])
elif mode == 'eval':
    dataset_val = AngioDataset()
    dataset_val.load_angio(datasetdir, 'train')
    dataset_val.prepare()

    #load angio data in coco format as coco object
    data = COCO(datasetdir +  'data_train.json')

    evaluate_coco(model, dataset_val, data, "bbox")

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
                epochs=20,
                layers='heads',
                augmentation=augmentation)

    # # Training - Stage 2
    # # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=60,
                layers='4+',
                augmentation=augmentation)

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=100,
                layers='all',
                augmentation=augmentation)
=======
>>>>>>> 55e3eb8e2f70958126214ffc51d2e51e82fd2ce2
