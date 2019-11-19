#!/usr/bin/env python3

import skimage as si
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
import configparser
from tqdm import tqdm

#all warnings are about low contrast images
import warnings
warnings.simplefilter("ignore")
	
config = configparser.ConfigParser()
config.read("config.ini")

#set user input variables
path_to_dataturks_annotation_json_folder = config["USER"]["jsons"]
path_to_masks_folder = config["USER"]["output"]
jsons = os.listdir(path_to_dataturks_annotation_json_folder)
path_to_real_images = config["USER"]["images"]
size = int(config["USER"]["size"])

#set colors based on mask type
colors = config["COLOR"]

#enable info logging.
logging.getLogger().setLevel(logging.INFO)

#*******************************************************************************
#Function 	: poly2mask
#Description: using given coordinates, a mask is drawn, colored based on label, 
#				and saved
#Parameters : blobs - list of x,y coordinates for the mask
#			  c - current cell number 
#			  path_to_masks_folder - path to output folder
#			  h - height of image
#			  w - width of image
#			  label - label of mask being drawn
#Returned 	: None
#Output		: Image of mask saved to output folder
#*******************************************************************************

def poly2mask(blobs, num, path_to_masks_folder, h, w, label):
	mask = np.zeros((h, w, 3))
	if label in colors.keys():
			color = list(eval(colors[label]))
	else:
		color = list(eval(colors["Default"]))
	for l in blobs:
		fill_row_coords, fill_col_coords = draw.polygon(l[1], l[0], l[2])
		mask[fill_row_coords, fill_col_coords] = color
	io.imsave(path_to_masks_folder + "/" + label + "_" + str(num) + ".png", si.img_as_ubyte(mask))

#*******************************************************************************
#Function 	: multi_mask
#Description: make a mask for every individual in a json in SEPARATE files
#Parameters : train - dictionary of images with information of each individual
#			  path_to_masks_folder - path to output folder
#Returned 	: None
#Output		: Image of one mask saved to output folder from poly2mask
#*******************************************************************************

def multi_masks(train, path_to_masks_folder):
	for image in train:
		h = image.get("size").get("height")
		w = image.get("size").get("width")
		c = 1
		# get the points for each mask in the image
		for objects in tqdm(image.get("objects")):
			blobs = []
			label = objects.get("classTitle")
			if (config["USER"]["skip_other"] == "True" and label == "Other"):
				continue
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

#*******************************************************************************
#Function 	: add_border_class_rect
#Description: add a border class around each individual in an image
#Parameters : l - list of x,y coordinates for the mask
#			  mask - array of pixels with object mask filled in
#			  size - size of border, total border width = 2*size + 1
#Returned 	: Mask with border class pixels added
#Output		: None
#*******************************************************************************

def add_border_class_rect(l, mask, size):
	border = np.zeros((mask.shape[0], mask.shape[1]))
	rr, cc = draw.polygon_perimeter(l[1], l[0], l[2])
	border[rr,cc] = 1
	for i in range(border.shape[0]):
		for j in range(border.shape[1]):
			if border[i,j] == 1:
				rr,cc = draw.rectangle((i-size,j-size), (i+size,j+size), shape = (mask.shape[0],mask.shape[1]))
				mask[rr,cc] = list(eval(colors["Border"]))            
			
#*******************************************************************************
#Function 	: poly2mask_single
#Description: using given coordinates, a mask is drawn for every individual 
#				in the file, colored based on label, and saved in one file
#Parameters : blobs - list of x,y coordinates for the mask
#			  c - current cell number 
#			  path_to_masks_folder - path to output folder
#			  h - height of image
#			  w - width of image
#Returned 	: None
#Output		: Image of all masks in one file saved to output folder
#*******************************************************************************

def poly2mask_single(blobs, c, path_to_masks_folder, h, w):
	mask = np.zeros((h, w,3))
	for label in blobs.keys():
		print(label)
		if label in colors.keys():
			color = list(eval(colors[label]))
		else:
			color = list(eval(colors["Default"]))
		for l in tqdm(blobs[label]):
			fill_row_coords, fill_col_coords = draw.polygon(l[1], l[0], l[2])
			mask[fill_row_coords, fill_col_coords] = color
			add_border_class_rect(l, mask, size)
	io.imsave(path_to_masks_folder + "/" + str(c) + ".png", si.img_as_ubyte(mask))

#*******************************************************************************
#Function 	: single_mask
#Description: make a mask for every individual in a json in ONE file
#Parameters : train - dictionary of images with information of each individual
#			  path_to_masks_folder - path to output folder
#			  file_name - name of image file
#Returned 	: None
#Output		: Image of all masks saved to output folder from poly2mask_single
#*******************************************************************************

def single_mask(train, path_to_masks_folder, file_name):
	for image in train:
		h = image.get("size").get("height")
		w = image.get("size").get("width")
		blobs = {}
		# get the points for each mask in the image
		for objects in image.get("objects"):
			label = objects.get("classTitle")
			if (config["USER"]["skip_other"] == "True" and label == "Other"):
				continue
			if label not in blobs.keys():
				blobs[label] = []
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
			blobs[label].append(l)
		# create mask for each object 
		poly2mask_single(blobs, file_name, path_to_masks_folder, h, w)

#*******************************************************************************
#Function 	: convert_dataturks_to_masks
#Description: create folder for masks, read in jsons, and proceed based on user 
#				input for masks, single or individual
#Parameters : path_to_dataturks_annotation_json - path to folder of jsons
#			  path_to_masks_folder - path to output folder
#			  file_name - name of image file
#Returned 	: None
#Output		: Image of masks saved to output folder
#*******************************************************************************

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
	
	#load in json information
	f = open(path_to_dataturks_annotation_json)
	train_data = f.readlines()
	train = []
	for line in train_data:
		data = json.loads(line)
		train.append(data)
		
	if config["USER"]["mask"] == "single":
		single_mask(train, path_to_masks_folder, file_name)
	else:
		multi_masks(train, path_to_masks_folder)
	
	f.close()
	
#*******************************************************************************
#Description: create folder for raw image and copy current image there, 
#				make masks out of jsons
#Output		: Image of masks saved to output folder
#*******************************************************************************
	
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
	copyfile(path_to_real_images + name + ".jpg", path_to_current_masks_folder + "/image/" + name + ".png")
	print(name)
	# create masks
	convert_dataturks_to_masks(path_to_dataturks_annotation_json_folder + file, path_to_current_masks_folder, name)	
