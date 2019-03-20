#system
import os
from os.path import dirname, abspath
import sys

#math
import math
from random import randint , choice, uniform
import numpy as np

#COCO image and json processing
import json
from project_assets.cocoapi.PythonAPI.pycocotools.coco import COCO 
from project_assets.cocoapi.PythonAPI.pycocotools import mask as maskUtils

#generated image processing
import cv2
from mrcnn import utils
from PIL import Image, ImageFile
from project_assets.cucu_utils import add_image, image_to_mask, add_imageWithoutTransparency

#cucu dependencies
from cucu_config import cucuConfForTrainingSession, globalObjectShapesList

#design
from collections import defaultdict



ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library
ImageFile.LOAD_TRUNCATED_IMAGES = True


class genDataset(utils.Dataset):
    def __init__(self,objByCategoryPaths,config):
        def collectAndCountObjImagesByCategory():
            self.objByCategoryPaths = objByCategoryPaths
            self.containerOfObjForGeneratingImages= defaultdict(list)
            self.quantityOfObjByCategory={}
            
            for key in globalObjectShapesList:
                for root, _, files in os.walk(self.objByCategoryPaths[key]):
                    for filename in files:
                        self.containerOfObjForGeneratingImages[key].append(Image.open(os.path.join(root, filename)).convert('RGBA'))
                    self.quantityOfObjByCategory[key] = len(files)
                    print("files num:" + str(self.quantityOfObjByCategory[key]) + key)
        
        def initiateConfiguration():
            self.config = config

        def initiateClassificationNames():
            for index, key in enumerate(globalObjectShapesList):
                addClassificationToNN("shapes", index, key)

        def addClassificationToNN(classificationType, index, classificationName):
            # NN's class_info attribute holds class's names and initialized with BG classification, therefore, avoid duplicate
            def classNameIsNotBackground(classificationName):
                return classificationName != 'BG'

            if classNameIsNotBackground(classificationName):
                self.add_class(classificationType, index, classificationName)
        
        utils.Dataset.__init__(self)
        collectAndCountObjImagesByCategory()
        initiateConfiguration()
        initiateClassificationNames()


    
    def load_shapes(self, numOfImagesToGenerate, height, width):
        """
        Generate the requested number of synthetic images.
        height, width: the size of the generated images.
        """

        # Add images
        for i in range(numOfImagesToGenerate):  
            # decide which background:
            bgIndex = randint(0, self.quantityOfObjByCategory['BG']-1) 
            #asher todo: remove bg_color when generating
            bg_color, shapes = self.GenerateRandomSpecsForImage(height, width)
            self.add_image("shapes", image_id=i, path=None, width=width, height=height, shapes=shapes, bgIndex=bgIndex)
    
    def load_image(self, image_id):
        """
        creates an image using spcifications choosed earlier for creating it.
        the specifications are tracked usiing image_id

        function is called by load_image_gt - it is crucial for generating on-the-fly training set 
        for NN.
        image_id - associates with certain attributes (image_info) of this image generated 
                   on constructing train_dataset and val_dataset
        """
        info = self.image_info[image_id]
                
        # load shape of pre-specified background
        y_max, x_max ,_ = np.asarray(self.containerOfObjForGeneratingImages['BG'][info['bgIndex']]).shape
        # todo: change y_max to imageHeight and x_max to imageWidth
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
            return choice(["cucumber","cucumber","cucumber", "leaf","leaf","leaf","leaf", "flower", "flower","stem","stem","stem","stem"])
        # Shape
        shape = drawObjCategoryFromDistribution()
        
        # Location in Image
        verifyObjIntersectsWithImageFrame = cucuConfForTrainingSession.BOUNDING_DELTA
        topLeftX_location = randint(0, height - int(verifyObjIntersectsWithImageFrame*height))
        topLeftY_location = randint(0, width - int(verifyObjIntersectsWithImageFrame*width))
        
        # Scalinf of object
        x_scale = uniform(cucuConfForTrainingSession.MIN_SCALE_OBJ, cucuConfForTrainingSession.MAX_SCALE_OBJ)
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
        randomObjAmmountInGeneratedImage = randint(math.floor(cucuConfForTrainingSession.MIN_GENERATED_OBJECTS * cucuConfForTrainingSession.SCALE_OBJECT_NUM_NEXT_EPOCH_ROUND),\
                                            math.floor(cucuConfForTrainingSession.MAX_GENERATED_OBJECTS * cucuConfForTrainingSession.SCALE_OBJECT_NUM_NEXT_EPOCH_ROUND))
            
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

    def annToMask(self, ann, height, width): #asher note: no need to copy
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

class HybridDataset(utils.Dataset):
    def __init__(self,pathsToHandleGeneratingImages, pathToRealImagesAnnotations, pathToRealImagesDataset, config):
        utils.Dataset.__init__(self)
        self.config = config
        self.pathToRealImagesAnnotations = pathToRealImagesAnnotations
        self.pathToRealImagesDataset = pathToRealImagesDataset
        self.generatedDataset = genDataset(pathsToHandleGeneratingImages, config)
        self.realDataset = realDataset()
    
    def imageIdBelongsToGeneratedDataset(self, image_id):
        # since image_id starts from zero, if image_id = num_images, it's actually the first image in realDataset
        return image_id < self.generatedDataset.num_images
    def mapToRealDatasetImageId(self, image_id):
        return image_id - self.generatedDataset.num_images

    def load_mask(self, image_id):
        
        
        if self.imageIdBelongsToGeneratedDataset(image_id):
            return self.generatedDataset.load_mask(image_id)
        else:
            realImageId = self.mapToRealDatasetImageId(image_id)
            return self.realDataset.load_mask(realImageId)


    def load_dataset(self,annotations_path, dataset_dir):
        self.generatedDataset.load_shapes(self.config.TRAIN_SET_SIZE, self.config.IMAGE_SHAPE[0], self.config.IMAGE_SHAPE[1])
        self.realDataset.load_dataset(self.pathToRealImagesAnnotations, self.pathToRealImagesDataset)

    def load_image(self, image_id):
        if self.imageIdBelongsToGeneratedDataset(image_id):
            return self.generatedDataset.load_image(image_id)
        else:
            realImageId = self.mapToRealDatasetImageId(image_id)
            return self.realDataset.load_image(realImageId)


    def prepare(self, class_map=None):
        """Prepares the Dataset class for use."""

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])
        def createHybridClassInfoDict():
            return dict(self.generatedDataset.class_info, self.realDataset.class_info)
        def createHybridImageInfoListStartsWithGenImages():
            return self.generatedDataset.image_info.extend(self.realDataset.image_info)
            
        self.generatedDataset.prepare()
        self.realDataset.prepare()
        # Build (or rebuild) everything else from the info dicts.
        self.class_info = createHybridClassInfoDict()
        self.image_info = createHybridImageInfoListStartsWithGenImages()
        
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                    for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                    for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)


class project_paths(object):
    def __init__(self, projectRootDir, TensorboardDir, trainedModelsDir,visualizeEvaluationsDir,trainOutputLog, currSessionInitialModelWeights,trainDatasetDir, valDatasetDir, testDatasetDir,trainResultContainer,testAnnotationsDir=None):
        self.projectRootDir=projectRootDir
        self.TensorboardDir=TensorboardDir
        self.trainedModelsDir=trainedModelsDir
        self.currSessionInitialModelWeights=currSessionInitialModelWeights
        self.trainDatasetDir=trainDatasetDir
        self.valDatasetDir=valDatasetDir
        self.testDatasetDir=testDatasetDir
        self.testAnnotationsDir=testAnnotationsDir
        self.trainResultContainer=trainResultContainer
        self.visualizeEvaluationsDir=visualizeEvaluationsDir
        self.trainOutputLog=trainOutputLog



class CucuLogger(object):
    def __init__(self, original_out, filepath):
        self.logger_file = open(filepath, "w+")
        self.orig_out = original_out

    def write(self, str):
        self.logger_file.write(str)
        self.orig_out.write(str)

    def flush(self):
        pass
    #todo: make sure that user input is printed to logger, decide which fields are importent
    def getFromUserCurrentSessionSpecs(self):
        print("####################################### PREFACE HEADER #######################################")
        print('Hello handsome, please provide spsifications for this session for easier analyzing later:')
        input()
        return 

    def __del__(self):
        self.logger_file.close()
