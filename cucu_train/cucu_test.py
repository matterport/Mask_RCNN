

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
from project_assets.cucu_classes import *
from cucu_config import cucumberConfig

import json

# from cucu_realDatasetClass import *
ROOT_DIR = dirname(dirname(os.path.realpath(__file__)))

# create centralized class for used paths during training
cucuPaths = project_paths(
    projectRootDir=ROOT_DIR,
    TensorboardDir=os.path.join(ROOT_DIR, "cucu_train/TensorBoardGraphs"),
    trainedModelsDir=os.path.join(ROOT_DIR, "cucu_train/trained_models"),
    cocoModelPath=os.path.join(ROOT_DIR, "mask_rcnn_coco.h5"),
    trainDatasetDir=os.path.join(ROOT_DIR, "cucu_train/project_dataset/train_data"),
    valDatasetDir=os.path.join(ROOT_DIR, "cucu_train/project_dataset/valid_data"),
    testDatasetDir=os.path.join(ROOT_DIR, "cucu_train/project_dataset/test_data"),
    trainResultContainer=os.path.join(ROOT_DIR, "cucu_train/trainResultContainer")
)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# In[11]:



#create configurations for model instentiating
config = cucumberConfig()
config.display()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=cucuPaths.TensorboardDir)




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


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


dataset_train = genDataset( cucuPaths.trainDatasetDir + '/cucumbers_objects', 
                            cucuPaths.trainDatasetDir + '/leaves_objects',
                            cucuPaths.trainDatasetDir + '/flower_objects',
                            cucuPaths.trainDatasetDir + '/background_folder/1024', config)
dataset_train.load_shapes(100, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

tests_location = cucuPaths.testDatasetDir + "/1024"
for filename in sorted(os.listdir(tests_location)):
    
    testImage = os.path.join(tests_location,filename)
    t = cv2.cvtColor(cv2.imread(testImage), cv2.COLOR_BGR2RGB)
    results = model.detect([t], verbose=1)
    r = results[0]
    # visualize.display_instances(t, r['rois'], r['masks'], r['class_ids'] ,dataset_train.class_names, r['scores'], ax=get_ax())
    visualize.display_top_masks(t, r['masks'], r['class_ids'] ,dataset_train.class_names, savePath="/home/simon/Mask_RCNN/cucu_train/" + filename.split("/")[-1] )

    t= dataset_train.class_names
    print(t)



# Load random image and mask.
image_id = random.choice(dataset_train.image_ids)
image = dataset_train.load_image(image_id)
mask, class_ids = dataset_train.load_mask(image_id)
# Compute Bounding box
bbox = utils.extract_bboxes(mask)

# Display image and additional stats
print("image_id ", image_id, dataset_train.image_reference(image_id))
log("image", image)
log("mask", mask)
log("class_ids", class_ids)
log("bbox", bbox)
# Display image and instances
visualize.display_instances(image, bbox, mask, class_ids, dataset_train.class_names)


# In[ ]:










