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

def make_mask(xy,size):
    
    '''
        Description:
        polygon : [(x1,y1), (x2,y2)]
        size : size of the image (width,height)

        return : PIL image (mask)
    '''
    # Making a blank mask
    mask = Image.new('RGB',size)
    
    # Making Image Draw Object
    img1 = ImageDraw.Draw(mask)   
    
    # Drawing polygons with respect to the points
    for xy_poly in xy: img1.polygon(xy_poly,fill = '#FF0000')  
    
    # Return mask
    return mask

def get_polygons_from_mask(mask, size):
    '''
        Description:
            This funtion will return you polygons from a binary maskj
            
        Input:
            mask (PIL)   : PIL image of the mask 
            size (tuple) : Size of the original Image
            
        Output:
            all_polygons (list) : List of all of the polygons
    '''
    # Initilizing Variables
    np_points, all_polygons = [], []
    
    
    # Finding Borders
    border = cv2.copyMakeBorder(np.asarray(mask,np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0 )
    
    # Rbg to gray
    border = cv2.cvtColor(border, cv2.COLOR_RGB2GRAY)
    
    # Finding countours (different Polygons)
    polygons= cv2.findContours(border, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = polygons[0] if len(polygons) == 2 else polygons[1]
    
    # Flattening all polygons
    polygons = [polygon.flatten() for polygon in polygons]
    
    #converting point list to numpy array, reshaping, rounding and type conversion
    for point in polygons:
        np_points.append(np.array(point).reshape(-1, 2).round().astype(int))
    
    # changing list of lists to tuples
    for np_point in np_points:
        processed_polygons = []
        for i in np_point: processed_polygons.append(tuple((tuple(i))))

        all_polygons.append(processed_polygons)
    
    # Returning all polygons
    return all_polygons








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
            img = sk.io.imread(img_path)[:,:,:3]
            
            # Getting the width and height
            w,h = img.shape[1], img.shape[0]
            
            # Getting the coordinates
            coordinates = list(labels.values())[0]
            
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
                self.r = self.pool.map_async(self.augment_image, [(img, coordinates, w, h, angle, sharpness, save_path, labels )], callback=self.callback, error_callback=self.err)
                
                # Only For Debugging (Sequential Rub)
                # aug_image_name,response = self.augment_image((img, coordinates, w, h, angle, sharpness, save_path, labels ))
                # self.augmented_images[aug_image_name] = response
        except:
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
        pil_image, coordinates, width, height, angle, sharpness, save_path, labels = args
        
        # Getting the mid of the image
        cx, cy = width//2, height//2
        
        # Converting image to PIL object
        pil_image = Image.fromarray(pil_image)
        
        # Getting the size of the image
        size = pil_image.size
        
        # Making the mask according to cordinates
        mask = make_mask(coordinates,size)
        
        # Rotating the image
        augmented = pil_image.rotate(angle, expand = True)
        
        # Rotating the mask
        augmented_mask = mask.rotate(angle, expand = True)
        
        # Adding sharpness
        sharpness = random.randint(100, 200)/100.0
        
        # Augment image
        augmented = Image.fromarray(brightness(np.asarray(augmented), sharpness))
        
        # Getting the cordinates from above funtion
        coordinates = []
        for i in get_polygons_from_mask(augmented_mask, size):
            for j in zip(*i):
                coordinates.append(list(map(int,list(j))))
        
        # Saving all of the things in response dict to save
        response = None
        if coordinates is not None:
            augmented.save(save_path)
            response  = {list(labels.keys())[0] : coordinates}
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
        
        # Getting cordinates
        for label in a1[augmented_image]:
            
            # Getting all x and y points
            all_y = [a1[augmented_image][label][i] for i in range(len(a1[augmented_image][label])) if (i+1) % 2 == 0]
            all_x = [a1[augmented_image][label][i] for i in range(len(a1[augmented_image][label])) if (i+1) % 2 != 0]
            
            # Creating shape attributes dict for every x and y
            for i in range(len(all_x)):
                region = dict()
                region["region_attributes"] = dict({"class": classes.index(label) + 1})
                region["shape_attributes"] = dict()
                region["shape_attributes"]["name"] = "polygon"
                region["shape_attributes"]["all_points_x"] = all_x[i]
                region["shape_attributes"]["all_points_y"] = all_y[i]
                mrcnn_data[key]['regions'].append(copy.deepcopy(region))

    
    # saving json
    with open(json_file_path, 'w') as f:
        json.dump(mrcnn_data, f)