import cv2
import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from pycocotools import mask as maskUtils
from pathlib import Path
import json
import imageio
from PIL import Image

def toBinMask(img=None, path=None, threshold=10):
    if path:
        # read mask in grayscale format
        img = cv2.imread(path, 0)

    # create binary map by thresholding
    ret, binMap = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    #convert bin map to rle
    return binMap

def makeSegmentations(data_path, subset, save_path):
    p = Path(data_path)
    # navigates the items found in data folder
    for item in p.iterdir():
        # if item is a folder containing masks
        if item.is_dir():
            # image_id is the folder name
            image_id = item.name
            # iterate through all the masks
            for f in item.iterdir():
                # get binary mask
                # binMask = np.expand_dims(toBinMask(path=str(f)), -1)
                binMask = toBinMask(path=str(f))

                originalImage = cv2.imread(data_path + '/' + image_id + '.jpeg', 0)

                # isolate masked region
                segmented = cv2.bitwise_and(originalImage, binMask)
                segmented_v = segmented

                # find sum of pixel value
                sumPx = np.sum(segmented)

                # count occurences of pixel value above 0 this also doubles as masked area
                occ = (segmented > 0).sum()

                # mean pixel value
                meanPx = sumPx / occ

                # threshold segmented image by mean pixel value
                ret, segmented_thresh = cv2.threshold(segmented , meanPx + 10, 0, cv2.THRESH_TOZERO_INV)

                # binary map
                ret, segmented_binary = cv2.threshold(segmented_thresh, 1, 255, cv2.THRESH_BINARY )

                # find area percentage
                arteryArea = (segmented_binary > 0).sum()
                percentage = arteryArea / occ

                # save
                imageio.imwrite(save_path + f.name.split('.')[0] + 'segmented.png', segmented_v)
                imageio.imwrite(save_path + f.name.split('.')[0] + 'segmented_threshold.png', segmented_thresh)
                imageio.imwrite(save_path + f.name.split('.')[0] + 'segmented_threshold_binary.png', segmented_binary)
                print(f.name.split('.')[0] + ' area percentage: ' + str(percentage))

makeSegmentations('A:/val', 'val', 'A:/segmented/')