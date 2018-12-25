

import os
from os.path import dirname, abspath
import sys
from random import randint , choice, uniform
import numpy as np
import cv2
from mrcnn import utils
from PIL import Image
from cucu_utils import *

#asher todo: change this variable name - to bum_of_object
minimum_number_of_cucumbers = 50
maximum_number_of_cucumbers = 70
#number_of_cucumbers = 4
min_scale = 0.5
max_scale = 0.8




class genDataset(utils.Dataset):
    def __init__(self, folder_objects_cucumber,folder_objects_leaf,folder_objects_flower, folder_bgs,config):
        """
        self variables:
            folder_object - folder containing object asher todo: what is exactly an object? image +annotations?
            folder_bgs - todo: TBD
            cucumberObj - container for all images in dataSet containing objects
            bg - container for all images in dataSet containing backGrounds
        """
        utils.Dataset.__init__(self)
        # asher todo: get rid of this variable later, config is not really needed here
        self.config = config
        self.folder_objects_cucumber = folder_objects_cucumber
        self.folder_objects_leaf = folder_objects_leaf
        self.folder_objects_flower = folder_objects_flower
        self.folder_bgs = folder_bgs
        self.cucumberObj = []
        self.leafObj = []
        self.flowerObj = []
        self.bg = []
 
        # asher todo: beautify calls -> pass a super folder which contains sub folders per object
        for root, _, files in os.walk(self.folder_objects_cucumber):
            for filename in files:
                self.cucumberObj.append(Image.open(os.path.join(root, filename)).convert('RGBA'))
        _, _, files_objects = next(os.walk(self.folder_objects_cucumber))
        self.number_of_cucumbers = len(files_objects)

        for root, _, files in os.walk(self.folder_objects_flower):
            for filename in files:
                self.flowerObj.append(Image.open(os.path.join(root, filename)).convert('RGBA'))
        _, _, files_objects = next(os.walk(self.folder_objects_flower))
        self.number_of_flowers = len(files_objects)
        
        for root, _, files in os.walk(self.folder_objects_leaf):
            for filename in files:
                self.leafObj.append(Image.open(os.path.join(root, filename)).convert('RGBA'))
        _, _, files_objects = next(os.walk(self.folder_objects_leaf))
        self.number_of_leaves = len(files_objects)
        
        
                
        for root, _, files in os.walk(self.folder_bgs):
            for filename in files:
                #self.bg.append(cv2.cvtColor(cv2.imread(os.path.join(root, filename)), cv2.COLOR_BGR2RGB))
                self.bg.append(Image.open(os.path.join(root, filename)).convert('RGBA'))
        _, _, files_bgs = next(os.walk(self.folder_bgs))
        self.number_of_bgs = len(files_bgs)
        print("folder: " + folder_objects_cucumber + " inited")
        # asher todo add prints
        print("folder: " + folder_bgs + " inited")

    
    def load_shapes(self, count, height, width):
        """
        Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """

        # Add classes
        self.add_class("shapes", 1, "cucumber")
        self.add_class("shapes", 2, "leaf")
        self.add_class("shapes", 3, "flower")


        # Add images
        for i in range(count):
            print('Image', i, end='\r')
            bg_color, shapes = self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None, width=width, height=height, bg_color=bg_color, shapes=shapes)
    
    def load_image(self, image_id):
        """
        for now we only have one shape- cucumber.
        function creates 'collage' bg+one image.

        function is called by load_image_gt - it is crucial for generating on-the-fly training set 
        for NN.
        image_id - associates with certain attributes (image_info) of this image generated 
                   on constructing train_dataset and val_dataset
        """
        info = self.image_info[image_id]
        
        index = randint(0, self.number_of_bgs-1) 
        
        # pull some random background from loaded bg set. which are typcally big
        y_topRight, x_topRight,channels = np.asarray(self.bg[index]).shape
        y_max, x_max ,_ = np.asarray(self.bg[index]).shape

        # pick random up-right corner
        x_topRight = randint(x_max- self.config.IMAGE_MAX_DIM//4 , x_max)
        y_topRight = randint(y_max- self.config.IMAGE_MAX_DIM//4 , y_max)
        # print("y_topRight:" , y_topRight , "index:", index) #asher todo: delete this
        # pick bottom-left corner for cropping the bg to fir image size which is (self.config.IMAGE_MAX_DIM)^2
        # x_bottomLeft = randint(0, x_topRight- self.config.IMAGE_MAX_DIM)
        # y_bottomLeft = randint(0, y_topRight- self.config.IMAGE_MAX_DIM)
        x_bottomLeft =x_topRight- self.config.IMAGE_MAX_DIM
        y_bottomLeft = y_topRight- self.config.IMAGE_MAX_DIM
        # build random area of configure IMAGE_SHAPE for net, which is IMAGE_MAX_DIM*IMAGE_MAX_DIM
        area = (x_bottomLeft, y_bottomLeft,   x_bottomLeft+self.config.IMAGE_MAX_DIM, y_bottomLeft+self.config.IMAGE_MAX_DIM)
        image = self.bg[index].crop(area)
        
        for shape, location, scale, angle, index in info['shapes']:
            image = self.draw_shape(image, shape, location, scale, angle, index)
        # asher todo: erase it later
        npImage = np.array(image)
        # remove transparency channel to fit to network data
        ImageWithoutTransparency = npImage[:,:,:3]
        return ImageWithoutTransparency
    
    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    
    def draw_shape_without_transparency(self, image, shape, location, scale, angle, index):
        """
        Draws a shape from the given specs.
        image - is just initiated to zeroes matrix 
        """
        if 'cucumber' == shape:
            x_location, y_location = location
            x_scale, y_scale = scale
            image = add_imageWithoutTransparency(image, np.array(self.cucumberObj[index]), x_location, y_location, x_scale, y_scale, angle)
        if 'leaf' == shape:
            x_location, y_location = location
            x_scale, y_scale = scale
            image = add_imageWithoutTransparency(image, np.array(self.leafObj[index]), x_location, y_location, x_scale, y_scale, angle)
        if 'flower' == shape:
            x_location, y_location = location
            x_scale, y_scale = scale
            image = add_imageWithoutTransparency(image, np.array(self.flowerObj[index]), x_location, y_location, x_scale, y_scale, angle)
        return image


    def draw_shape(self, Collage, shape, location, scale, angle, index):
        """
        Draws another cucumber on a selected background
        Get the center x, y and the size s
        x, y, s = dims
        """
        

        if shape == 'cucumber':
            x_location, y_location = location
            x_scale, y_scale = scale
            Collage = add_image(Collage, self.cucumberObj[index], x_location, y_location, x_scale, y_scale, angle)
        if shape == 'leaf':
            x_location, y_location = location
            x_scale, y_scale = scale
            Collage = add_image(Collage, self.leafObj[index], x_location, y_location, x_scale, y_scale, angle)
        if shape == 'flower':
            x_location, y_location = location
            x_scale, y_scale = scale
            Collage = add_image(Collage, self.flowerObj[index], x_location, y_location, x_scale, y_scale, angle)
        return Collage
    
    
    def random_shape(self, height, width):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        """
        # Shape
        shape = choice(["cucumber","cucumber","cucumber", "leaf","leaf","leaf","leaf", "flower", "flower"])
        # Color
        # TopLeft x, y
        x_location = randint(0, height)
        y_location = randint(0, width)
        # Scale x, y
        x_scale = uniform(min_scale, max_scale)
        y_scale = uniform(min_scale, max_scale)
        # Angle
        angle = randint(-10,10)
        # Image index
        if "cucumber" == shape:
            index = randint(0, self.number_of_cucumbers-1)
        elif "leaf" == shape:
            index = randint(0, self.number_of_leaves-1)
        elif "flower" == shape:
            index = randint(0, self.number_of_flowers-1)

        return shape, (x_location, y_location), (x_scale, y_scale), angle, index
    
    # asher note: we don't use this func. for now ->doesn't support multi-shapes for now!!!!!!!!!!!!!!!!!!!!
    def random_image_opencv(self, height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        """
        # Pick random background color
        bg_color = np.array([randint(0, 255) for _ in range(3)])
        # Generate a few random shapes and record their
        # bounding boxes
        shapes = []
        boxes = []
        indexes  = []
        N = randint(minimum_number_of_cucumbers, maximum_number_of_cucumbers)
        
        image = np.ones([height, width, 3], dtype=np.uint8)
        
        for _ in range(N):
            shape, location, scale, angle, index = self.random_shape(height, width)
            
            image = add_image(image, self.cucumberObj[index], location[0], location[1], scale[0], scale[1], angle)
            y, x, _ = self.cucumberObj[index].shape
            
            #shapes.append((shape, color, dims))
            shapes.append((shape, location, scale, angle, index))
            #TODO boxes
            #x, y, s = dims
            #boxes.append([y-s, x-s, y+s, x+s])
            boxes.append([location[1], location[0], location[1] + y, location[0] + x])
            
        # Apply non-max suppression wit 0.3 threshold to avoid
        # shapes covering each other
        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.4)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes
    
    
    def random_image(self, height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        """
        # Pick random background color
        bg_color = np.array([randint(0, 255) for _ in range(3)])
        # Generate a few random shapes and record their
        # bounding boxes
        shapes = []
        boxes = []
        indexes  = []
        N = randint(minimum_number_of_cucumbers, maximum_number_of_cucumbers)
            
        for _ in range(N):
            shape, location, scale, angle, index = self.random_shape(height, width)
            if "cucumber" == shape:
                y, x,channels = np.asarray(self.cucumberObj[index]).shape
            elif "leaf" == shape:
                y, x,channels = np.asarray(self.leafObj[index]).shape
            elif "flower" == shape:
                y, x,channels = np.asarray(self.flowerObj[index]).shape
            shapes.append((shape, location, scale, angle, index))
            boxes.append([location[1], location[0], location[1] + y, location[0] + x])
            
        # Apply non-max suppression wit 0.3 threshold to avoid
        # shapes covering each other
        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.7)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes
    
    def load_mask(self, image_id):
        """
        Generate instance masks for shapes of the given image ID.
        image_id = a key to get atttributes of Collage.
        (a generated image with bg + different objects of different shapes)->(image_info)
        """
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)        

        #asher note: for now itterates only once on cucumber shape
        for i, (shape, location, scale, angle, index) in enumerate(info['shapes']):
            image = np.zeros([info['height'], info['width'], 3], dtype=np.uint8)
            # save in temp for easier inspection if needed
            temp = image_to_mask(self.draw_shape_without_transparency(image, shape, location, scale, angle, index))
            # construct array of masks related to all shapes of objescts in current Collage
            mask[:, :, i] = temp[:,:]
            
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        
        #print(occlusion)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
       
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask.astype(np.bool), class_ids.astype(np.int32)        

