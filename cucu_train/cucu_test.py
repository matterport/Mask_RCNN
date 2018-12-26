

# In[1]:




import os
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
from cucu_genDatasetClass import *
from cucu_config import cucumberConfig

# from cucu_realDatasetClass import *

import json

# ROOT_DIR = os.path.abspath("../")
ROOT_DIR = dirname(dirname(os.path.realpath(__file__)))
print(ROOT_DIR)

import faulthandler
# faulthandler.enable()
dumpTo = ROOT_DIR + "/cucu_train/Dumps/coreDump"
# dumpTo_fd = open(dumpTo, 'w')
# faulthandler.dump_traceback(file=dumpTo_fd, all_threads=True)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Directory to save logs 
TENSOR_BOARD_DIR = os.path.join(ROOT_DIR, "cucu_train/TensorBoardGraphs")

# Directory to save trained model:
TRAINED_MODELS_DIR = os.path.join(ROOT_DIR, "cucu_train/trained_models")


# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
  


# In[11]:



#create configurations for model instentiating
config = cucumberConfig()
config.display()


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=TENSOR_BOARD_DIR)




# In[12]:


class InferenceConfig(cucumberConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=TENSOR_BOARD_DIR)

# Load trained weights
list_of_trained_models = glob.glob(TRAINED_MODELS_DIR +'/*')
latest_trained_model = max(list_of_trained_models, key=os.path.getctime)

print("Loading weights from ", latest_trained_model)
model.load_weights(latest_trained_model, by_name=True)



# In[14]:


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# Training dataset
# asher todo: find a workaround to get rid of initiating a dataset object here
dataset_train = genDataset( ROOT_DIR + '/cucu_train/cucumbers_objects', 
                            ROOT_DIR + '/cucu_train/leaves_objects',
                            ROOT_DIR + '/cucu_train/flower_objects',
                        ROOT_DIR + '/cucu_train/background_folder', config)
dataset_train.load_shapes(100, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()
tests_location = ROOT_DIR + "/cucu_train/simple_test/"
for filename in sorted(os.listdir(tests_location)):
    
    testImage = os.path.join(tests_location,filename)
    t = cv2.cvtColor(cv2.imread(testImage), cv2.COLOR_BGR2RGB)
    results = model.detect([t], verbose=1)
    r = results[0]
    # visualize.display_instances(t, r['rois'], r['masks'], r['class_ids'] ,dataset_train.class_names, r['scores'], ax=get_ax())
    visualize.display_top_masks(t, r['masks'], r['class_ids'] ,dataset_train.class_names)

    t= dataset_train.class_names
    print(t)


# In[ ]:




# # Compute VOC-Style mAP @ IoU=0.5
# # Running on 10 images. Increase for better accuracy.
# image_ids = np.random.choice(dataset_val.image_ids, 100)
# APs = []
# for image_id in image_ids:
#     # Load image and ground truth data
#     image, image_meta, gt_class_id, gt_bbox, gt_mask =        modellib.load_image_gt(dataset_val, inference_config,
#                                image_id, use_mini_mask=False)
#     molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
#     # Run object detection
#     results = model.detect([image], verbose=0)
#     r = results[0]
#     # Compute AP
#     AP, precisions, recalls, overlaps =        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
#                          r["rois"], r["class_ids"], r["scores"], r['masks'])
#     APs.append(AP)
    
# print("mAP: ", np.mean(APs))






# In[ ]:










