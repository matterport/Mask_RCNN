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