# -*- coding: utf-8 -*-
"""

Dataset for Mask R-CNN
Configurations and data loading code for COCO format.

@author: Mattia Brusamento
"""

import os
import sys
import time
import numpy as np

# Download and install the Python coco tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

from mrcnn import model as modellib, utils


############################################################
#  Dataset
############################################################

class TrashDataset(utils.Dataset):

    def load_trash(self, data_dir, anno_file):
        print("Loading Trash Data:" + str(data_dir) + "" + str(anno_file))
        trash = COCO(os.path.join(data_dir, anno_file))

        # Add classes
        class_ids = sorted(trash.getCatIds())
        for i in class_ids:
            self.add_class("trash", i, trash.loadCats(i)[0]["name"])

        # Add images
        image_ids = list(trash.imgs.keys())

        for i in image_ids:
            self.add_image(
                "trash", image_id=i,
                path=os.path.join(data_dir, trash.imgs[i]['file_name']),
                width=trash.imgs[i]["width"],
                height=trash.imgs[i]["height"],
                annotations=[a for a in trash.loadAnns(trash.getAnnIds()) if a['image_id'] == str(i)])

        return trash

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
        image_info = self.image_info[image_id]

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "trash.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(TrashDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the trash Website."""
        info = self.image_info[image_id]
        if info["source"] == "trash":
            info['file_name']
        else:
            super(TrashDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
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
