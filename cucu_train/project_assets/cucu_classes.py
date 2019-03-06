
import os
from os.path import dirname, abspath
import sys
from random import randint , choice, uniform
import numpy as np
import cv2
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from project_assets.cucu_utils import *

from cucu_config import *
from cucu_config import cucuConf


import math

# here you can add more object keys 
objKeysList= ['BG', 'cucumbers', 'flowers', 'leaves', 'stems']

from collections import defaultdict
class genDataset(utils.Dataset):
    def __init__(self,objByCategoryPaths,config):
        def collectAndCountObjImagesByCategory():
            self.objByCategoryPaths = objByCategoryPaths
            self.containerOfObjForGeneratingImages={}
            self.quantityOfObjByCategory={}
            
            for key in objKeysList:
                for root, _, files in os.walk(self.objByCategoryPaths[key]):
                    for filename in files:
                        self.containerOfObjForGeneratingImages[key].append(Image.open(os.path.join(root, filename)).convert('RGBA'))
                _, _, files_objects = next(os.walk(self.objByCategoryPaths[key]))
                
                self.quantityOfObjByCategory[key] = len(files_objects)
        utils.Dataset.__init__(self)

        collectAndCountObjImagesByCategory()
        
        # asher todo: what todo with that?
        self.config = config

    
    def load_shapes(self, numOfImagesToGenerate, height, width):
        """
        Generate the requested number of synthetic images.
        height, width: the size of the generated images.
        """

        #asher todo: move this section into init
        #Add categories
        SKIP_BACKGROUND_KEY = 1
        for index, key in enumerate(objKeysList, SKIP_BACKGROUND_KEY):
            self.add_class("shapes", index, key)

        
        # Add images
        for i in range(numOfImagesToGenerate):  
            # decide wich background:
            bgIndex = randint(0, self.quantityOfObjByCategory['BG']-1) 
            #asher todo: remove bg_color when generating
            bg_color, shapes = self.GenerateRandomSpecsForImage(height, width)
            self.add_image("shapes", image_id=i, path=None, width=width, height=height, bg_color=bg_color, shapes=shapes, bgIndex=bgIndex)
    
    def load_image(self, specification_id):
        """
        creates an image using spcifications choosed earlier for creating it.
        the specifications are tracked usiing specification_id

        function is called by load_image_gt - it is crucial for generating on-the-fly training set 
        for NN.
        specification_id - associates with certain attributes (image_info) of this image generated 
                   on constructing train_dataset and val_dataset
        """
        info = self.image_info[specification_id]
                
        # pull some random background from loaded bg set. which are typcally big
        y_topRight, x_topRight,channels = np.asarray(self.containerOfObjForGeneratingImages['BG'][info['bgIndex']]).shape
        y_max, x_max ,_ = np.asarray(self.containerOfObjForGeneratingImages['BG'][info['bgIndex']]).shape

        # pick random up-right corner
        x_topRight = randint(x_max - self.config.IMAGE_MAX_DIM//2 , x_max)
        y_topRight = randint(y_max - self.config.IMAGE_MAX_DIM//2 , y_max)
        x_bottomLeft = x_topRight - self.config.IMAGE_MAX_DIM
        y_bottomLeft = y_topRight - self.config.IMAGE_MAX_DIM
        # build random area of configure IMAGE_SHAPE for net, which is IMAGE_MAX_DIM*IMAGE_MAX_DIM

        # temporary values (left, upper, right, lower)-tuple
        if self.config.IMAGE_MAX_DIM == 1024:
            area = (0, 0, 1024, 1024)
        else:
            area = (x_bottomLeft,y_bottomLeft,x_topRight,y_topRight)
        image = self.containerOfObjForGeneratingImages['BG'][info['bgIndex']].crop(area)

        for shape, location, scale, angle, index in info['shapes']:
            image = self.draw_shape(image, shape, location, scale, angle, index)
        
        # remove transparency channel to fit to network data
        npImage = np.array(image)
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
        topLeftX_location, topLeftY_location = location
        x_scale, y_scale = scale
        image = add_imageWithoutTransparency(image, np.array(self.containerOfObjForGeneratingImages[shape][index]), topLeftX_location, topLeftY_location, x_scale, y_scale, angle)
        
        return image


    def draw_shape(self, imageToDrawShapeOn, shape, location, scale, angle, index, erode_coeff=5, gaussian_coeff=3):
        """
        Draws another cucumber on a selected background
        Get the center x, y and the size s
        x, y, s = dims
        :param erode_coeff: size of kernel for erosion to apply on mask when blending to image
        :param gaussian_coeff: size of kernel for gaussian blur around mask when blending to image
        """
        topLeftX_location, topLeftY_location = location
        x_scale, y_scale = scale
        Collage = add_image(imageToDrawShapeOn, self.containerOfObjForGeneratingImages[shape][index], topLeftX_location,\
                             topLeftY_location, x_scale, y_scale, angle, erode_coeff, gaussian_coeff)
        
        return Collage
    
    
    def GenerateRandomSpecsForObjInImage(self, height, width):
        """Generates specifications of a random shape that interesects at least partially with given height ,width 
        of generated image.
        """
        def drawObjCategoryFromDistribution():
            return choice(["cucumber","cucumber","cucumber", "leaf","leaf","leaf","leaf", "flower", "flower"])
        # Shape
        shape = drawObjCategoryFromDistribution()
        
        # Location in Image
        verifyObjIntersectsWithImageFrame = cucuConf.BOUNDING_DELTA
        topLeftX_location = randint(0, height - int(verifyObjIntersectsWithImageFrame*height))
        topLeftY_location = randint(0, width - int(verifyObjIntersectsWithImageFrame*width))
        
        # Scalinf of object
        x_scale = uniform(cucuConf.MIN_SCALE_OBJ, cucuConf.MAX_SCALE_OBJ)
        y_scale = x_scale
        
        # Angle
        angle = randint(-10,10)

        # object index in containerOfObjForGeneratingImages
        # asher todo: perhaps create class of containerOfObjForGeneratingImages with attributes
        objIndexInContainer = randint(0, self.quantityOfObjByCategory[shape]-1)

        return shape, (topLeftX_location, topLeftY_location), (x_scale, y_scale), angle, objIndexInContainer
    
    def GenerateRandomSpecsForImage(self, height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        """
        #asher todo: observe and remove bg_Color
        # Pick random background color
        bg_color = np.array([randint(0, 255) for _ in range(3)])
        # Generate a few random shapes and record their
        # bounding boxes
        shapes = []
        boxes = []

        # pick objects number in generated image
        randomObjAmmountInGeneratedImage = randint(math.floor(cucuConf.MIN_GENERATED_OBJECTS * cucuConf.SCALE_OBJECT_NUM_NEXT_EPOCH_ROUND),\
                                            math.floor(cucuConf.MAX_GENERATED_OBJECTS * cucuConf.SCALE_OBJECT_NUM_NEXT_EPOCH_ROUND))
            
        for _ in range(randomObjAmmountInGeneratedImage):
            shape, location, scale, angle, index = self.GenerateRandomSpecsForObjInImage(height, width)
            y, x,channels = np.asarray(self.containerOfObjForGeneratingImages[shape][index]).shape
        
            shapes.append((shape, location, scale, angle, index))
            boxes.append([location[1], location[0], location[1] + y, location[0] + x])
            
        # Apply non-max suppression wit 0.3 threshold to avoid
        # shapes covering each other
        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(randomObjAmmountInGeneratedImage), 0.7)
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
            mask[:, :, i] = temp[:, :]
            
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        
        #print(occlusion)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
       
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask.astype(np.bool), class_ids.astype(np.int32)        




import os
import numpy as np
import json
from mrcnn import utils
from project_assets.cocoapi.PythonAPI.pycocotools.coco import COCO
from project_assets.cocoapi.PythonAPI.pycocotools import mask as maskUtils







class realDataset(utils.Dataset):
    def load_dataset(self,annotations_path, dataset_dir):
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

        coco = COCO(annotations_path)
        image_dir = "{}".format(dataset_dir)

        # All classes
        # asher todo: instead of coco lets add our categories
        class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))

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
        # If not a COCO image, delegate to parent class.
        # if image_info["source"] != "coco":
        #     return super(realDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
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
            return super(realDataset, self).load_mask(image_id)

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


class project_paths(object):
    def __init__(self, projectRootDir, TensorboardDir, trainedModelsDir,visualizeEvaluationsDir,trainOutputLog, cocoModelPath,trainDatasetDir, valDatasetDir, testDatasetDir,trainResultContainer,testAnnotationsDir=None):
        self.projectRootDir=projectRootDir
        self.TensorboardDir=TensorboardDir
        self.trainedModelsDir=trainedModelsDir
        self.cocoModelPath=cocoModelPath
        self.trainDatasetDir=trainDatasetDir
        self.valDatasetDir=valDatasetDir
        self.testDatasetDir=testDatasetDir
        self.testAnnotationsDir=testAnnotationsDir
        self.trainResultContainer=trainResultContainer
        self.visualizeEvaluationsDir=visualizeEvaluationsDir
        self.trainOutputLog=trainOutputLog


