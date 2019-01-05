
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
    visualizeEvaluationsDir = os.path.join(CURRENT_CONTAINER_DIR, "visualizeEvaluations"),
    cocoModelPath=         os.path.join(ROOT_DIR, "mask_rcnn_coco.h5"),
    trainDatasetDir=       os.path.join(ROOT_DIR, "cucu_train/project_dataset/train_data"),
    valDatasetDir=         os.path.join(ROOT_DIR, "cucu_train/project_dataset/valid_data"),
    testDatasetDir=        os.path.join(ROOT_DIR, "cucu_train/project_dataset/test_data"),
    trainResultContainer=  CURRENT_CONTAINER_DIR
)

try:
    original_umask = os.umask(0)
    os.makedirs(cucuPaths.trainedModelsDir, mode=0o777)
    os.makedirs(cucuPaths.visualizeEvaluationsDir, mode=0o777)

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

# seleect your weapon of choice
# list_of_trained_models = glob.glob(ROOT_DIR + "/trained_models" +'/*')
# latest_trained_model = sorted(list_of_trained_models, key=os.path.getctime)[-1]
model.load_weights(ROOT_DIR + "/cucu_train/trained_models/"+"cucuWheights_2019-01-05 19:39:10.350050.h5", by_name=True)


# model.load_weights(cucuPaths.cocoModelPath, by_name=True,
#                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
#                                 "mrcnn_bbox", "mrcnn_mask"])

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
        




# In[12]:


class InferenceConfig(cucumberConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()




# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=cucuPaths.TensorboardDir)

# Load trained weights
list_of_trained_models = glob.glob(cucuPaths.trainedModelsDir +'/*')
latest_trained_model = max(list_of_trained_models, key=os.path.getctime)

print("Loading weights from ", latest_trained_model)
model.load_weights(latest_trained_model, by_name=True)



# In[14]:
# DISPLAY_TOP_MASKS
#create container directories per function calls from Visualize module
os.mkdir(cucuPaths.visualizeEvaluationsDir + "/display_top_masks")
tests_location = cucuPaths.testDatasetDir + "/1024"
for filename in sorted(os.listdir(tests_location)):
    
    testImage = os.path.join(tests_location,filename)
    t = cv2.cvtColor(cv2.imread(testImage), cv2.COLOR_BGR2RGB)
    results = model.detect([t], verbose=1)
    r = results[0]
    # visualize.display_instances(t, r['rois'], r['masks'], r['class_ids'] ,dataset_train.class_names, r['scores'], ax=get_ax())
    visualize.display_top_masks(t, r['masks'], r['class_ids'] ,dataset_train.class_names, savePath=cucuPaths.visualizeEvaluationsDir + "/display_top_masks/"  + filename.split("/")[-1] )

    t= dataset_train.class_names
    print(t)


# DISPLAY_INSTANCES
#create container directories per function calls from Visualize module
os.mkdir(cucuPaths.visualizeEvaluationsDir + "/display_instances")
os.mkdir(cucuPaths.visualizeEvaluationsDir + "/plot_precision_recall")
os.mkdir(cucuPaths.visualizeEvaluationsDir + "/plot_overlaps")

image_ids = np.random.choice(dataset_val.image_ids, 2)
for image_id in image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, config, image_id, use_mini_mask=False)
    info = dataset_val.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                        dataset_val.image_reference(image_id)))
    # Run object detection
    results = model.detect([image], verbose=1)

    # Display results
    ax = get_ax(1)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names, r['scores'], ax=ax,title="Predictions", \
                                savePath=cucuPaths.visualizeEvaluationsDir + "/display_instances/" + "display_instances_" + "image_" + str(image_id) +".png")
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    # Load random image and mask.
    image = dataset_val.load_image(image_id)
    mask, class_ids = dataset_val.load_mask(image_id)
    # Compute Bounding box
    bbox = utils.extract_bboxes(mask)

    # Display image and additional stats
    print("image_id ", image_id, dataset_val.image_reference(image_id))
    log("image", image)
    log("mask", mask)
    log("class_ids", class_ids)
    log("bbox", bbox)
    # Display image and instances
    visualize.display_instances(image, bbox, mask, class_ids, dataset_val.class_names, savePath=cucuPaths.visualizeEvaluationsDir + "/display_instances/" + "display_instances2_" + "image_" + str(image_id) +".png")

    # Draw precision-recall curve
    AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                            r['rois'], r['class_ids'], r['scores'], r['masks'])
    visualize.plot_precision_recall(AP, precisions, recalls,savePath=cucuPaths.visualizeEvaluationsDir + "/plot_precision_recall/" + "plot_precision_recall_" + "image_" + str(image_id) +".png")

    # Grid of ground truth objects and their predictions
    visualize.plot_overlaps(gt_class_id, r['class_ids'], r['scores'],
                        overlaps, dataset_val.class_names,savePath=cucuPaths.visualizeEvaluationsDir + "/plot_overlaps/" + "plot_overlaps" + "image_" + str(image_id) +".png")
# In[ ]:




# # Compute VOC-style Average Precision
# def compute_batch_ap(image_ids):
#     APs = []
#     for image_id in image_ids:
#         # Load image
#         image, image_meta, gt_class_id, gt_bbox, gt_mask =\
#             modellib.load_image_gt(dataset_val, config,
#                                    image_id, use_mini_mask=False)
#         # Run object detection
#         results = model.detect([image], verbose=0)
#         # Compute AP
#         r = results[0]
#         AP, precisions, recalls, overlaps =\
#             utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
#                               r['rois'], r['class_ids'], r['scores'], r['masks'])
#         APs.append(AP)
#     return APs

# # Pick a set of random images
# image_ids = np.random.choice(dataset_val.image_ids, 10)
# APs = compute_batch_ap(image_ids)
# print("mAP @ IoU=50: ", np.mean(APs))

print("ASHERRRRRRRRRRR   CHANGE BACK TO COCO WEIGHTS!!!")





# In[ ]:










