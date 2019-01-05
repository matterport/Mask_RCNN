
# coding: utf-8

# In[ ]:





# In[1]:



import tensorflow as tf
print(tf.__version__)
import os
# asher note: macOS workaround
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import glob
from os.path import dirname, abspath
import sys
import datetime
import random
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('QT5Agg')
from project_assets.cucu_classes import *
from cucu_config import cucumberConfig
from PIL import Image
# from cucu_realDatasetClass import *
ROOT_DIR = dirname(dirname(os.path.realpath(__file__)))

# create a container for training result per exexution of cucu_train.py
CONTAINER_ROOT_DIR = ROOT_DIR + "/cucu_train/trainResultContainer/"
now = datetime.datetime.now()
CURRENT_CONTAINER_DIR = CONTAINER_ROOT_DIR +"train_results_" + str(now)
os.chmod(ROOT_DIR, mode=0o777)
# create centralized class for used paths during training
cucuPaths = project_paths(
    projectRootDir=ROOT_DIR,
    TensorboardDir=        os.path.join(CURRENT_CONTAINER_DIR, "TensorBoardGraphs"),
    trainedModelsDir=      os.path.join(CURRENT_CONTAINER_DIR, "trained_models"),
    cocoModelPath=         os.path.join(ROOT_DIR, "mask_rcnn_coco.h5"),
    trainDatasetDir=       os.path.join(ROOT_DIR, "cucu_train/project_dataset/train_data"),
    valDatasetDir=         os.path.join(ROOT_DIR, "cucu_train/project_dataset/valid_data"),
    testDatasetDir=        os.path.join(ROOT_DIR, "cucu_train/project_dataset/test_data"),
    trainResultContainer=  CURRENT_CONTAINER_DIR
)

try:
    original_umask = os.umask(0)
    os.makedirs(cucuPaths.trainedModelsDir, mode=0o777)
finally:
    os.umask(original_umask)
import json
print(cucuPaths.projectRootDir)
# Import Mask RCNN
sys.path.append(cucuPaths.projectRootDir)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# In[11]:

import sys
print(sys.version)

#create configurations for model instentiating
config = cucumberConfig()
# config.display()





# In[3]:



# Training dataset
# asher todo: add a choice from which dataset to generate
dataset_train = genDataset( cucuPaths.trainDatasetDir + '/cucumbers_objects', 
                            cucuPaths.trainDatasetDir + '/leaves_objects',
                            cucuPaths.trainDatasetDir + '/flower_objects',
                            cucuPaths.trainDatasetDir + '/background_folder/1024', config)
dataset_train.load_shapes(100, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = genDataset(   cucuPaths.valDatasetDir + '/cucumbers_objects', 
                            cucuPaths.valDatasetDir + '/leaves_objects',
                            cucuPaths.valDatasetDir + '/flower_objects',
                            cucuPaths.valDatasetDir + '/background_folder/1024', config)
dataset_val.load_shapes(3, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

# In[ ]:



# # asher todo: change code to fit new load_image method of coco
# #show n random image&mask train examples
# n = 3
# image_ids = np.random.choice(dataset_train.image_ids, n)
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     print(image.shape)
#     # images = visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names, 3)

    # save images for presentations
    # cm = plt.get_cmap('gist_earth', lut=50)

    # img = Image.fromarray(images[0])
    # img.save(str(image_id) + "_pic" + ".png", "PNG")

    # apply color map to masks
    # img = (cm(images[1])[:, :, :3] * 255).astype(np.uint8)
    # img = Image.fromarray(img)
    # img.save(str(image_id) + "_mask_leaf" + ".png", "PNG")

    # img = (cm(images[2])[:, :, :3] * 255).astype(np.uint8)
    # img = Image.fromarray(img)
    # img.save(str(image_id) + "_mask_fruit" + ".png", "PNG")

    # img = (cm(images[3])[:, :, :3] * 255).astype(np.uint8)
    # img = Image.fromarray(img)
    # img.save(str(image_id) + "_mask_flower" + ".png", "PNG")

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=cucuPaths.TensorboardDir)


# In[ ]:
# add custom callbacks if needed
custom_callbacks=[]

# # seleect your weapon of choice
# list_of_trained_models = glob.glob(cucuPaths.trainedModelsDir +'/*')
# latest_trained_model = sorted(list_of_trained_models, key=os.path.getctime)[-1]
# model.load_weights(latest_trained_model, by_name=True)

model.load_weights(cucuPaths.cocoModelPath, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])

# In[ ]:


#asher todo: make for loop on generated and real data set
for _ in range(1):

    model.train(dataset_train, dataset_val, learning_rate= config.LEARNING_RATE, epochs=1,\
                            custom_callbacks=custom_callbacks, layers="heads",verbose=0)

    # Save weights
    now = datetime.datetime.now()
    model_path = os.path.join(cucuPaths.trainedModelsDir, "cucuWheights_" + str(now) + ".h5")
    model.keras_model.save_weights(model_path)
    #load just trained weights again
    list_of_trained_models = glob.glob(cucuPaths.trainedModelsDir +'/*')
    latest_trained_model = sorted(list_of_trained_models, key=os.path.getctime)[-1]
    model.load_weights(latest_trained_model, by_name=True)

    oldest_trained_model = min(list_of_trained_models, key=os.path.getctime)
    if len(list_of_trained_models) > config.MAX_SAVED_TRAINED_MODELS:
        os.remove(oldest_trained_model)
        



# In[ ]:










