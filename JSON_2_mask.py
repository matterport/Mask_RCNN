#!/usr/bin/env python

# Adapted from https://medium.com/@dataturks/converting-polygon-bounded-boxes-in-the-dataturks-json-format-to-mask-images-f747b7ba921c

from skimage import draw
from skimage import io
import numpy as np
import urllib.request
import json
import logging
import os
import sys
import re
from shutil import copyfile

#all warnings are about low contrast images
import warnings
warnings.simplefilter("ignore")

# EDIT THESE
path_to_dataturks_annotation_json_folder = "Nuclei_discovery/ds_002/ann/"
path_to_masks_folder = "Hand_drawn_masks/"
jsons = os.listdir(path_to_dataturks_annotation_json_folder)
path_to_real_images = "Nuclei_discovery/ds_002/img/"


#enable info logging.
logging.getLogger().setLevel(logging.INFO)

def poly2mask(blobs, c, path_to_masks_folder, h, w, label):
    mask = np.zeros((h, w))
    for l in blobs:
        fill_row_coords, fill_col_coords = draw.polygon(l[1], l[0], l[2])
        mask[fill_row_coords, fill_col_coords] = 1.0
    io.imsave(path_to_masks_folder + "/" + label + "_" + str(c) + ".png", mask)


def convert_dataturks_to_masks(path_to_dataturks_annotation_json, path_to_masks_folder, file_name):
    # make sure everything is setup.
    if (not os.path.isdir(path_to_masks_folder)):
        logging.exception(
            "Please specify a valid directory path to write mask files, " + path_to_masks_folder + " doesn't exist")
    if (not os.path.exists(path_to_dataturks_annotation_json)):
        logging.exception(
            "Please specify a valid path to dataturks JSON output file, " + path_to_dataturks_annotation_json + " doesn't exist")

    # create folder for each image
    if (not os.path.exists(path_to_masks_folder + "/" + "masks")):
        os.mkdir(path_to_masks_folder + "/" + "masks")
    
    path_to_masks_folder = path_to_masks_folder + "/" + "masks"
    
    # load in json information
    f = open(path_to_dataturks_annotation_json)
    train_data = f.readlines()
    train = []
    for line in train_data:
        data = json.loads(line)
        train.append(data)
        
    # get the image size 
    for image in train:
        h = image.get("size").get("height")
        w = image.get("size").get("width")
        c = 1
        # get the points for each mask in the image
        for objects in image.get("objects"):
            blobs = []
            label = objects.get("classTitle")
            points = objects.get("points").get("exterior")
            x_coord = []
            y_coord = []
            l = []
            for p in points:
                x_coord.append(p[0])
                y_coord.append(p[1])
            shape = (h, w)
            l.append(x_coord)
            l.append(y_coord)
            l.append(shape)
            blobs.append(l)
            # create mask for each object 
            poly2mask(blobs, c, path_to_masks_folder, h, w, label)
            c += 1
    f.close()

# create all masks for multple images
for file in jsons:
    name = re.search(r'(.*)\..*\.json', file)
    name = name.group(1)
    path_to_current_masks_folder = path_to_masks_folder + name 
    # create directories
    if (not os.path.exists(path_to_current_masks_folder)):
        os.mkdir(path_to_current_masks_folder)
    if (not os.path.exists(path_to_current_masks_folder + "/image")):
        os.mkdir(path_to_current_masks_folder + "/image")
    # copy real image to correct folder
    copyfile(path_to_real_images + name + ".png", path_to_current_masks_folder + "/image/" + name + ".png")
    
    # create masks
    convert_dataturks_to_masks(path_to_dataturks_annotation_json_folder + file, path_to_current_masks_folder, name)
