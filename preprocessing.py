# Standard imports
import cv2
import numpy as np;
import os
import sys
import itertools
import math
import logging
import json
import re
import random
import glob
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

import utils
import model as modellib
from model import log

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.expanduser("~/data/gdxray")


from PIL import Image


def crop(infile,width):
	"""Horizontally crop an image"""
	im = Image.open(infile)
	imgwidth, imgheight = im.size
	for i in range(imgwidth//width):
		box = (i*width, 0, (i+1)*width, imgheight)
		yield im.crop(box)


def crop_images_in_directory(image_dir):
	"""Horizontally crop all the images in a directory"""
	pattern = image_dir+"/*.png"
	print("Searching for files with pattern",pattern)
	for infile in glob.glob(pattern):
		for i,cropped in enumerate(crop(infile,768)):
			newname = infile.replace(".png","_%i.png"%i)
			cropped.save(newname)
			print("Crop:", infile,"->",newname)
		print("Removing original file: ", infile)
		os.remove(infile)


def rename_images_in_directory(image_dir, template):
	"""Rename all the images in a directory according to template"""
	pattern = image_dir+"/*.png"
	print("Searching for files with pattern",pattern)

	for i,infile in enumerate(sorted(glob.glob(pattern))):
		directory = os.path.dirname(infile)
		basename = os.path.basename(infile)
		new_image_name = os.path.join(directory, template%i)
		# rename
		print("Rename:", infile, "->", new_image_name)
		os.rename(infile, new_image_name)


def build_mask_file(mask_path, mask_dir_out):
	"""Generate masks using the images in mask_dir_in"""
	if not os.path.exists(mask_path):
		raise Exception("No such path: "+mask_path)

	im = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
	ret, thresh = cv2.threshold(im, 127, 255, 0)
	im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	for i,contour in enumerate(contours):
		mask = np.zeros_like(im, np.uint8)
		cv2.drawContours(mask,contours,i,(255,255,255),-1) # -1 fills contour
		# Name the mask
		new_mask_name = os.path.basename(mask_path).replace(".png","_%i.png"%i)
		new_mask_path = os.path.join(mask_dir_out, new_mask_name)
		print("Saving mask:", new_mask_path)
		cv2.imwrite(new_mask_path, mask)


def generate_metadata(dataset_dir, image_dir):
	"""Generate a metadata file to describe this dataset"""
	with open("metadata.txt","w") as metadata:
		for image in sorted(glob.glob(image_dir+"/*.png")):
			relpath = os.path.relpath(image, dataset_dir)
			print("Adding metadata",relpath)
			metadata.write(relpath+os.linesep)


def weld_segmentation(path):
	# cv2 is does not throw an error if the path is wrong
	if not os.path.exists(path):
		raise Exception("No such path: "+path)

	# Read image
	im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

	ret, thresh = cv2.threshold(im, 127, 255, 0)
	im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#See, there are three arguments in cv2.findContours() function, first one is source image, second is contour retrieval mode, third is contour approximation method. And it outputs a modified image, the contours and hierarchy. contours is a Python list of all the contours in the image. Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.

	#We will discuss second and third arguments and about hierarchy in details later. Until then, the values given to them in code sample will work fine for all images.
	#How to draw the contours?
	#To draw the contours, cv2.drawContours function is used. It can also be used to draw any shape provided you have its boundary points. Its first argument is source image, second argument is the contours which should be passed as a Python list, third argument is index of contours (useful when drawing individual contour. To draw all contours, pass -1) and remaining arguments are color, thickness etc.

	color_im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)


	#To draw all the contours in an image:
	cv2.drawContours(color_im, contours, -1, (128,255,0), 3)
	#To draw an individual contour, say 4th contour:
	cv2.drawContours(color_im, contours, 3, (0,255,0), 3)
	#But most of the time, below method will be useful:
	cnt = contours[4]
	cv2.drawContours(color_im, [cnt], 0, (0,255,0), 3)

	cv2.imshow("Keypoints", color_im)
	cv2.waitKey(0)


def prepare_welding():
	datadir = os.path.join(DATA_DIR,'Welds')
	welding = os.path.join(DATA_DIR, 'Welding')

	# Delete welding directory if it exists
	if os.path.exists(welding):
		shutil.rmtree(welding)

	# Create a new welding directory
	welding = shutil.copytree(datadir, welding)

	image_dir = os.path.join(welding,"W0001")
	masks_dir = os.path.join(welding,"W0002")

	# Crop the images and masks to 768 pixels wide
	crop_images_in_directory(image_dir)
	crop_images_in_directory(masks_dir)

	# Rename all the cropped images
	template = "W0001_%04.d.png"
	rename_images_in_directory(image_dir, template)
	rename_images_in_directory(masks_dir, template)

	# Copy masks over to the correct directory
	new_mask_dir = os.path.join(image_dir,"masks")
	os.mkdir(new_mask_dir)
	for mask_file in glob.glob(masks_dir+"/*.png"):
		build_mask_file(mask_file, new_mask_dir)

	# Generate a metadata file
	generate_metadata(DATA_DIR, image_dir)


if __name__=="__main__":
	prepare_welding()



