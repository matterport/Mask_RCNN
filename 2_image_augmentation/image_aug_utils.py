import numpy as np
from PIL import Image
import random
from scipy import ndarray, ndimage
import skimage.io as io
import skimage as sk
from skimage import transform
from skimage import util
import cv2
import shutil
import os
import json
from PIL import ImageDraw
from IPython.display import display
import glob
import time
import threading
import traceback
import matplotlib.pyplot as plt
import copy
from random import randint
from math import sin, cos, radians


def display(display_list):
    
    labels_list = ["Image", "Masks"]
    plt.figure(figsize=(15, 15))
    for i in range(len(display_list)):
        
        plt.subplot(1, len(display_list), i+1)
        plt.title(labels_list[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()


def brightness(img, brightness):
    '''
    
        Add brightness to the Image
    
    '''
    return sk.exposure.adjust_gamma(img, brightness).astype(np.uint8)




def rotatePolygon(polygon, degrees, height, width):
    """ 
    Description:
    
        Rotate polygon the given angle about its center. 
        
    Input:
        polygon (list of tuples)  : list of tuples with (x,y) cordinates 
                                    e.g [(1,2), (2,3), (4,5)]
        
        degrees int               : Rotation Degrees
    
    Output:
    
        polygon (list of tuples)  : Polygon rotated on angle(degrees)
                                e.g [(1,2), (2,3), (4,5)]
    
    """
    # Convert angle to radians
    theta = radians(degrees)
    
    # Getting sin and cos with respect to theta
    cosang, sinang = cos(theta), sin(theta) 

    # find center point of Polygon to use as pivot
    y, x = [i for i in zip(*polygon)]
    
    # find center point of Polygon to use as pivot
    
    cx1 = width[0] / 2
    cy1 = height[0] / 2
    cx2 = width[1] / 2
    cy2 = height[1] / 2
    
    # Rotating every point
    new_points = []
    for x, y in zip(x, y):
        tx, ty = x-cx1, y-cy1
        new_x = (tx*cosang - ty*sinang) + cx2
        new_y = (tx*sinang + ty*cosang) + cy2
        new_points.append((new_y, new_x))
    return new_points








def parse_via_json(json_file_path, images_dir_path):
    '''
    Description:
        Parse your json which you made from VIA-Image-Annotator. Converting your annotation to the format 
        which mrcnn support
        
    Input:
        json_file_path  (str) : Path of the json file
        images_dir_path (str) : Path of the Images Directory
        
    Output:
        images (dict) : Dictionary includes all of the images and their annotations respectively
    
    '''
    
    # Load json
    labels = json.load(open(json_file_path))['_via_img_metadata']
    
    # Seperating Keys
    keys = list(labels.keys())
    
    # Creating Images list seperately
    images = dict()
    
    
    
    # Taking key from the keys (labels)
    for key in keys:
        
        # Getting regions
        regions = labels[key]["regions"]
         
        # Getting image name
        image_name = labels[key]['filename']
        
        # Creating annotations
        annotations = dict()
        
        
        # Converting regions to mrcnn format
        for region in regions:
            try:
                
                # Getting x points
                x = region['shape_attributes']['all_points_x']
                
                # Getting y points
                y = region['shape_attributes']['all_points_y']
                
                # Getting the label names
                region_name = list(region['region_attributes']['class'].keys())[0]
                
                # Checking if label already exists (It means it another object belong to same class)
                if region_name not in annotations:
                    annotations[region_name] = [list(zip(x,y))]
                else:
                    annotations[region_name].append(list(zip(x,y)))
            
            except:
                traceback.print_exc()
                pass
         
        # Summing up all the annotations
        

        images[images_dir_path + image_name] = annotations
    
    # Return
    return images

def draw_polygons(img, polygons):
    # Ploting Rotated Image
    img1 = ImageDraw.Draw(img)  
    for i in polygons:
        for polygon in polygons[i]:
            img1.polygon(polygon, fill ="#FFF000", outline ="blue") 

    plt.imshow(img)
    plt.show()

class Augmentation:
    '''
        Doing all the work in multiprocessing
    
    
    '''
    
    def __init__(self, DIR_PATH, degree = 2):
        
        '''
            Initilizing all the variables 
        '''
        from multiprocessing.pool import ThreadPool
        
        # Creating thread pool of 32
        self.pool = ThreadPool(32)
        
        # Creating dict for augmenting images
        self.augmented_images = dict()
        
        # Checking for pending thread 
        self.pending = 0
        
        # Specefying Dir path
        self.DIR_PATH = DIR_PATH
        
        # Removing previous annotations
        shutil.rmtree(self.DIR_PATH, True)
        
        
        # Rotation degrees (Rotate every image at angle)
        self.degree = degree
        
        # creating and changing the permissions of folder created
        if not os.path.exists(self.DIR_PATH):
            os.makedirs(self.DIR_PATH, mode=0o777)

        
        
    def __del__(self):
        self.pool.close()
        self.pool.join()

        
    def callback(self, response):
        '''
            This callback fucntion initiate at the end of thread (getting the results processed by thread)
        
        '''
        
        # Getting results
        aug_image_name, response = response[0]
        
        # If thread end sucessfully than saving response
        if aug_image_name:
            self.augmented_images[aug_image_name] = response
        
        # Subtracting pending thread number
        self.pending -= 1
        
    def err(self, error):
        
        '''
        
            This fucntion initiate when any exception occurs with in a thread
        
        '''
        # Subtracting pending thread number
        self.pending -= 1
        print("Error", error)
        
        # printing exception trace-back
        traceback.print_exc()
        
        
        
    def run(self, img_path, labels):
        '''
        
            This function initiate threads
        
        '''

        try:
                # Reading Image
            img = np.asarray(Image.open(img_path))

            # Getting the width and height
            w,h = img.shape[1], img.shape[0]

            # Looping through all of the angles
                # 1. Augment images every 4 angles
            for angle in range(0, 360, self.degree):

                # Logging for thread created
                self.pending += 1

                # Getting random number for sharpness
                sharpness  = random.randint(100, 200)/100.0

                # Creating the image name
                aug_image_name = "{3}-{2}-{0}-{1}.jpg".format(angle, sharpness, random.randint(1111111,9999999), img_path.split("/")[-1])

                # Getting the save path
                save_path = self.DIR_PATH + aug_image_name  

                # Initiating thread
                self.r = self.pool.map_async(self.augment_image, [(img, labels, w, h, angle, sharpness, save_path)], callback=self.callback, error_callback=self.err)

#                 Only For Debugging (Sequential flow)
                
#                 aug_image_name,response = self.augment_image((img, labels, w, h, angle, sharpness, save_path ))
#                 self.augmented_images[aug_image_name] = response
        except StopIteration:
            print(f"Cannot find labels for {img_path}")
            pass
        
    def augment_image(self, args):
        '''
            This is the main fucntion which augment imagse
        
        '''
        
        # Parameters taken by the fucntion
        #         1. Pil Image : Image
        #         2. Cordinates : Cordinates of the regions
        #         3. width.     : Width of the image
        #         4. height.    : Height of the image
        #         5. angle      : angle to rotate image
        #         6. sharpness. : Scale of sharpness
        #         7. save_path. : path to save the image
        #         8. labels.    : labels according to the image
        pil_image, labels, width, height, angle, sharpness, save_path = args
        
        # Converting image to PIL object
        pil_image = Image.fromarray(pil_image)
        
        # Rotating the image
        augmented = pil_image.rotate(angle, expand = True)
        
        h1, w1 = pil_image.size
        h2, w2 = augmented.size

        
        rotated_labels = dict() 
        for i in labels:
            rotated_labels[i] = [rotatePolygon(i, (angle), [h1,h2], [w1,w2]) for i in labels[i]]

        augmented = Image.fromarray(brightness(np.asarray(augmented), random.randint(100, 200)/100.0))
        
        # Saving all of the things in response dict to save
        response = None
        if rotated_labels is not None:
            augmented.save(save_path)
            response  = rotated_labels
            aug_image_name = save_path.split("/")[-1]            
            return (aug_image_name, copy.deepcopy(response))
        else:
            return (False, False)



def save_mrcnn_labels(augmented_images,json_file_path ,classes):
    
    '''
    
        This function saves your converted annotations into a json file for training
    
    '''
    
    # Creating an empty dict to save annotations
    mrcnn_data = dict()
    
    # Creating temporary variable
    a1 = augmented_images
    
    # Getting all of the images and their augmentation
    for augmented_image in a1:
        
        # Getting the name and extension of the image
        parts = augmented_image.split(".")
        image_name, extension = ".".join(parts[:-1]), parts[-1]
        
        # Creating key for every augmented image
        key = image_name + ".json"
        mrcnn_data[key] = dict()
        mrcnn_data[key]['filename'] = image_name
        mrcnn_data[key]['file_attributes'] = dict()
        mrcnn_data[key]['regions'] = []
        
        for label in a1[augmented_image]:
            
            # Getting all x and y points
            for objects in a1[augmented_image][label]:
                all_x,all_y = [i for i in zip(*objects)]
                 
                # Creating shape attributes dict for every x and y
                region = dict()
                region["region_attributes"] = dict({"class": classes.index(label) + 1})
                region["shape_attributes"] = dict()
                region["shape_attributes"]["name"] = "polygon"
                region["shape_attributes"]["all_points_x"] = list(all_x)
                region["shape_attributes"]["all_points_y"] = list(all_y)
                mrcnn_data[key]['regions'].append(copy.deepcopy(region))

    
    # saving json
    with open(json_file_path, 'w') as f:
        json.dump(mrcnn_data, f)