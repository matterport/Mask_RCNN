#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN - Inspect Neurons Trained Model
# 
# Code and visualizations to test, debug, and evaluate the Mask R-CNN model.

#%%
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = '42'
matplotlib.rcParams['ps.fonttype'] = '42'
import matplotlib.pyplot as plt
import matplotlib.patches as patches
try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

# Root directory of the project
ROOT_DIR = os.path.abspath("/home/nel/Code/NEL_LAB/Mask_RCNN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.neurons import neurons

get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# ## Configurations

# %%
config = neurons.NeuronsConfig()
cross = 'cross3'
NEURONS_DIR = os.path.join(ROOT_DIR, ("datasets/neurons_3channel_cross1_all"))

# %%
# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.8
    IMAGE_RESIZE_MODE = "pad64"
    IMAGE_MAX_DIM=512
    #CLASS_THRESHOLD = 0.33
    #RPN_NMS_THRESHOLD = 0.7
    #DETECTION_NMS_THRESHOLD = 0.3
    #IMAGE_SHAPE=[512, 128,3]
    RPN_NMS_THRESHOLD = 0.5
    #TRAIN_ROIS_PER_IMAGE = 1000
    POST_NMS_ROIS_INFERENCE = 1000
    DETECTION_MAX_INSTANCES = 1000

config = InferenceConfig()
config.display()

# %%
# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

# ## Load Validation Dataset
# Load validation dataset
dataset = neurons.NeuronsDataset()
dataset.load_neurons(NEURONS_DIR, "val")

# Must call before using the dataset
dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))


# ## Load Model
# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

# %%
# Set path to balloon weights file

# Download file from the Releases page and set its path
# https://github.com/matterport/Mask_RCNN/releases
# weights_path = "/path/to/mask_rcnn_balloon.h5"

# Or, load the last model you trained
weights_path = model.find_last()
#weights_path = MODEL_DIR + "/neurons20191203T1603/mask_rcnn_neurons_0035.h5"
#weights_path = MODEL_DIR + "/neurons20191203T1330/mask_rcnn_neurons_0045.h5"
#weights_path = MODEL_DIR + "/neurons20191203T1748/mask_rcnn_neurons_0040.h5"
#weights_path = MODEL_DIR + "/neurons20191204T1134/mask_rcnn_neurons_0040.h5"
#weights_path = MODEL_DIR + "/neurons20191204T1315/mask_rcnn_neurons_0040.h5"
#weights_path = MODEL_DIR + "/neurons20191204T1346/mask_rcnn_neurons_0040.h5"
#weights_path = MODEL_DIR + "/neurons20191205T0021/mask_rcnn_neurons_0040.h5"     #cross3
#weights_path = MODEL_DIR + "/neurons20191204T2351/mask_rcnn_neurons_0040.h5"    #cross2
#weights_path = MODEL_DIR + "/neurons20191204T2307/mask_rcnn_neurons_0040.h5"    #cross1

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)


# ## Run Detection

# %%
prop = 'val'
dataset = neurons.NeuronsDataset()
dataset.load_neurons(NEURONS_DIR, prop)

# Must call before using the dataset
dataset.prepare()

import os 
folder = '/home/nel/Code/NEL_LAB/Mask_RCNN/output/'+os.path.split(os.path.split(weights_path)[0])[-1]+'/'+prop+'/'
try:
    os.makedirs(folder)
    print('make folder')
except:
    print('already exist')


# %%
image_id = random.choice(dataset.image_ids)
image_id = 1
image, image_meta, gt_class_id, gt_bbox, gt_mask =    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset.image_reference(image_id)))

# Run object detection

results = model.detect([image], verbose=1)

# Display results
ax = get_ax(1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)


# %%
from caiman.base.rois import nf_match_neurons_in_binary_masks
performance=[]
F1 = {}
recall = {}
precision = {}
number = {}
for image_id in dataset.image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask =        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                           dataset.image_reference(image_id)))
    
    # Run object detection    
    results = model.detect([image], verbose=1)
    
    # Display results
    _, ax = plt.subplots(1,1, figsize=(16,16))
    r = results[0]
    display_result = True
    if display_result:  
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    dataset.class_names, r['scores'], ax=ax,
                                    title="Predictions")
        #plt.savefig(folder +dataset.image_info[image_id]['id'][:-4]+'_mrcnn_result.pdf')
        plt.close()
    
    mask_pr = r['masks'].copy().transpose([2,0,1])*1.
    mask_gt = gt_mask.copy().transpose([2,0,1])*1.
    
    tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
            mask_gt, mask_pr, thresh_cost=0.7, min_dist=10, print_assignment=True,
            plot_results=True, Cn=image[:,:,0], labels=['GT', 'MRCNN'])
    #plt.savefig(folder +dataset.image_info[image_id]['id'][:-4]+'_compare.pdf')
    plt.close()
    performance.append(performance_cons_off)
    F1[dataset.image_info[image_id]['id'][:-4]] = performance_cons_off['f1_score']
    recall[dataset.image_info[image_id]['id'][:-4]] = performance_cons_off['recall']
    precision[dataset.image_info[image_id]['id'][:-4]] = performance_cons_off['precision']
    number[dataset.image_info[image_id]['id'][:-4]] = dataset.image_info[image_id]['polygons'].shape[0]

# %%
processed = {}
for i in ['F1','recall','precision','number']:
    result = eval(i)
    processed[i] = {}    
    for j in ['L1','TEG','HPC']:
        if j == 'L1':
            temp = [result[i] for i in result.keys() if 'Fish' not in i and 'IVQ' not in i]
            if i == 'number':
                processed[i]['L1'] = sum(temp)
            else:
                processed[i]['L1'] = sum(temp)/len(temp)
        if j == 'TEG':
            temp = [result[i] for i in result.keys() if 'Fish' in i]
            if i == 'number':
                processed[i]['TEG'] = sum(temp)
            else:
                processed[i]['TEG'] = sum(temp)/len(temp)
        if j == 'HPC':
            temp = [result[i] for i in result.keys() if 'IVQ' in i]
            if i == 'number':
                processed[i]['HPC'] = sum(temp)
            else:
                processed[i]['HPC'] = sum(temp)/len(temp)

# %%
results_all[cross+'_'+prop] = processed

# %%
r = results_all.copy()

# %%
prop = 'val'
f = {}
f['F1'] = []
f['recall'] = []
f['precision'] = []
f['number'] = []
for i in ['F1','recall','precision','number']:
    for j in ['cross1_'+prop,'cross2_'+prop,'cross3_'+prop]:
        f[i].append(list(r[j][i].values()))        

print(prop)
for i in ['F1','recall','precision','number']:
    print(i)
    print('Mean value')
    print(np.array(f[i]).mean(axis=0))
    print('Standard deviation')
    print(np.array(f[i]).std(axis=0))
    
    
    

# %%


# train
print('Ahrens',sum(ss[:2])/2)
print('Svoboda',sum(ss[2:8])/6)
print('Cohen',sum(ss[8:])/len(ss[8:]))


# In[24]:


# val
print('Ahrens',sum(ss[:1])/len(ss[:1]))
print('Svoboda',sum(ss[1:4])/len(ss[1:4]))
print('Cohen',sum(ss[4:])/len(ss[4:]))


# In[41]:


# val
print('Ahrens',sum(ss[:1])/len(ss[:1]))
print('Svoboda',sum(ss[1:4])/len(ss[1:4]))
print('Cohen',sum(ss[4:])/len(ss[4:]))


# In[88]:


precision


# In[20]:



image_id = 2
image, image_meta, gt_class_id, gt_bbox, gt_mask =    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset.image_reference(image_id)))

# Run object detection    
results = model.detect([image], verbose=1)

# Display results
_, ax = plt.subplots(1,1, figsize=(16,16))
r = results[0]
display_result = True
if display_result:  
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, r['scores'], ax=ax,
                                title="Predictions")
plt.savefig(folder +dataset.image_info[image_id]['id'][:-4]+'_mrcnn_result.pdf')
plt.close()

mask_pr = r['masks'].transpose([2,0,1])
mask_gt = gt_mask.transpose([2,0,1])

tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
        mask_gt, mask_pr, thresh_cost=0.99, min_dist=10, print_assignment=True,
        plot_results=True, Cn=image[:,:,0], labels=['GT', 'MRCNN'])
plt.savefig(folder +dataset.image_info[image_id]['id'][:-4]+'_compare.pdf')
plt.close()
performance.append(performance_cons_off)


# In[30]:


tp_gt


# In[17]:


# Override the training configurations with a few
# changes for inferencing.
ct = [0.2,0.3,0.4,0.5]
summary = {}
for i in ct:
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.01
        IMAGE_RESIZE_MODE = "pad64"
        IMAGE_MAX_DIM=512
        CLASS_THRESHOLD = i
        RPN_NMS_THRESHOLD = 0.7
        DETECTION_NMS_THRESHOLD = 0.3
        #IMAGE_SHAPE=[512, 128,3]
        #RPN_NMS_THRESHOLD = 0.7
        #TRAIN_ROIS_PER_IMAGE = 1000
    config = InferenceConfig()

    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=config)

    weights_path = MODEL_DIR + "/neurons20191008T1658/mask_rcnn_neurons_0075.h5"

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    from caiman.base.rois import nf_match_neurons_in_binary_masks
    performance=[]
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask =            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        info = dataset.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                               dataset.image_reference(image_id)))

        # Run object detection

        results = model.detect([image], verbose=1)

        # Display results
        _, ax = plt.subplots(1,1, figsize=(16,16))
        r = results[0]
        display_result = True
        if display_result:  
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                        dataset.class_names, r['scores'], ax=ax,
                                        title="Predictions")
        #plt.savefig(folder +dataset.image_info[image_id]['id'][:-4]+'_mrcnn_result.pdf')
        plt.close()

        mask_pr = r['masks'].transpose([2,0,1])
        mask_gt = gt_mask.transpose([2,0,1])

        tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
                mask_gt, mask_pr, thresh_cost=1, min_dist=10, print_assignment=True,
                plot_results=True, Cn=image[:,:,0], labels=['GT', 'MRCNN'])
        #plt.savefig(folder +dataset.image_info[image_id]['id'][:-4]+'_compare.pdf')
        plt.close()
        performance.append(performance_cons_off)
        
    summary[i] = performance



# In[138]:


for key, performance in summary.items():
    print((sum([performance[i]['f1_score'] for i in range(len(performance))][:3]))/3)
    print([performance[i]['f1_score'] for i in range(len(performance))][:3])


# In[18]:


for key, performance in summary.items():
    print((sum([performance[i]['f1_score'] for i in range(len(performance))][:3]))/3)
    print([performance[i]['f1_score'] for i in range(len(performance))][:3])


# ## Color Splash
# 
# This is for illustration. You can call `balloon.py` with the `splash` option to get better images without the black padding.

# In[ ]:





# In[19]:


splash = neurons.color_splash(image, r['masks'])
display_images([splash], cols=1)


# ## Step by Step Prediction

# ## Stage 1: Region Proposal Network
# 
# The Region Proposal Network (RPN) runs a lightweight binary classifier on a lot of boxes (anchors) over the image and returns object/no-object scores. Anchors with high *objectness* score (positive anchors) are passed to the stage two to be classified.
# 
# Often, even positive anchors don't cover objects fully. So the RPN also regresses a refinement (a delta in location and size) to be applied to the anchors to shift it and resize it a bit to the correct boundaries of the object.

# ### 1.a RPN Targets
# 
# The RPN targets are the training values for the RPN. To generate the targets, we start with a grid of anchors that cover the full image at different scales, and then we compute the IoU of the anchors with ground truth object. Positive anchors are those that have an IoU >= 0.7 with any ground truth object, and negative anchors are those that don't cover any object by more than 0.3 IoU. Anchors in between (i.e. cover an object by IoU >= 0.3 but < 0.7) are considered neutral and excluded from training.
# 
# To train the RPN regressor, we also compute the shift and resizing needed to make the anchor cover the ground truth object completely.

# In[59]:


# Generate RPN trainig targets
# target_rpn_match is 1 for positive anchors, -1 for negative anchors
# and 0 for neutral anchors.
target_rpn_match, target_rpn_bbox = modellib.build_rpn_targets(
    image.shape, model.anchors, gt_class_id, gt_bbox, model.config)
log("target_rpn_match", target_rpn_match)
log("target_rpn_bbox", target_rpn_bbox)

positive_anchor_ix = np.where(target_rpn_match[:] == 1)[0]
negative_anchor_ix = np.where(target_rpn_match[:] == -1)[0]
neutral_anchor_ix = np.where(target_rpn_match[:] == 0)[0]
positive_anchors = model.anchors[positive_anchor_ix]
negative_anchors = model.anchors[negative_anchor_ix]
neutral_anchors = model.anchors[neutral_anchor_ix]
log("positive_anchors", positive_anchors)
log("negative_anchors", negative_anchors)
log("neutral anchors", neutral_anchors)

# Apply refinement deltas to positive anchors
refined_anchors = utils.apply_box_deltas(
    positive_anchors,
    target_rpn_bbox[:positive_anchors.shape[0]] * model.config.RPN_BBOX_STD_DEV)
log("refined_anchors", refined_anchors, )


# In[60]:


# Display positive anchors before refinement (dotted) and
# after refinement (solid).
visualize.draw_boxes(image, boxes=positive_anchors, refined_boxes=refined_anchors, ax=get_ax())


# ### 1.b RPN Predictions
# 
# Here we run the RPN graph and display its predictions.

# In[61]:


# Run RPN sub-graph
pillar = model.keras_model.get_layer("ROI").output  # node to start searching from

# TF 1.4 and 1.9 introduce new versions of NMS. Search for all names to support TF 1.3~1.10
nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression:0")
if nms_node is None:
    nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV2:0")
if nms_node is None: #TF 1.9-1.10
    nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV3:0")

rpn = model.run_graph([image], [
    ("rpn_class", model.keras_model.get_layer("rpn_class").output),
    ("pre_nms_anchors", model.ancestor(pillar, "ROI/pre_nms_anchors:0")),
    ("refined_anchors", model.ancestor(pillar, "ROI/refined_anchors:0")),
    ("refined_anchors_clipped", model.ancestor(pillar, "ROI/refined_anchors_clipped:0")),
    ("post_nms_anchor_ix", nms_node),
    ("proposals", model.keras_model.get_layer("ROI").output),
])


# In[62]:


# Show top anchors by score (before refinement)
limit = 100
sorted_anchor_ids = np.argsort(rpn['rpn_class'][:,:,1].flatten())[::-1]
visualize.draw_boxes(image, boxes=model.anchors[sorted_anchor_ids[:limit]], ax=get_ax())


# In[63]:


sorted_anchor_ids.shape


# In[64]:


# Show top anchors with refinement. Then with clipping to image boundaries
limit = 1000
ax = get_ax(1, 2)
pre_nms_anchors = utils.denorm_boxes(rpn["pre_nms_anchors"][0], image.shape[:2])
refined_anchors = utils.denorm_boxes(rpn["refined_anchors"][0], image.shape[:2])
refined_anchors_clipped = utils.denorm_boxes(rpn["refined_anchors_clipped"][0], image.shape[:2])
visualize.draw_boxes(image, boxes=pre_nms_anchors[:limit],
                     refined_boxes=refined_anchors[:limit], ax=ax[0])
visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[:limit], ax=ax[1],cmap='gray',vmax=np.percentile(image,95))


# In[65]:


# Show refined anchors after non-max suppression
limit = 1000
ixs = rpn["post_nms_anchor_ix"][:limit]
visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[ixs], ax=get_ax(),cmap='gray',vmax=np.percentile(image,95))


# In[55]:


config.IMAGE_SHAPE = np.array([508,288,3])


# In[56]:


# Show final proposals
# These are the same as the previous step (refined anchors 
# after NMS) but with coordinates normalized to [0, 1] range.
limit = 2000
# Convert back to image coordinates for display
h, w = config.IMAGE_SHAPE[:2]
proposals = rpn['proposals'][0, :limit] * np.array([h, w, h, w])
visualize.draw_boxes(image, refined_boxes=proposals, ax=get_ax(), cmap='gray', vmax=np.percentile(image, 95))


# ## Stage 2: Proposal Classification
# 
# This stage takes the region proposals from the RPN and classifies them.

# ### 2.a Proposal Classification
# 
# Run the classifier heads on proposals to generate class propbabilities and bounding box regressions.

# In[117]:


# Get input and output to classifier and mask heads.
mrcnn = model.run_graph([image], [
    ("proposals", model.keras_model.get_layer("ROI").output),
    ("probs", model.keras_model.get_layer("mrcnn_class").output),
    ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),
    ("masks", model.keras_model.get_layer("mrcnn_mask").output),
    ("detections", model.keras_model.get_layer("mrcnn_detection").output),
])


# In[116]:


config.DETECTION_NMS_THRESHOLD=0.3


# In[118]:


# Get detection class IDs. Trim zero padding.
det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
det_count = np.where(det_class_ids == 0)[0][0]
det_class_ids = det_class_ids[:det_count]
detections = mrcnn['detections'][0, :det_count]

print("{} detections: {}".format(
    det_count, np.array(dataset.class_names)[det_class_ids]))

captions = ["{} {:.3f}".format(dataset.class_names[int(c)], s) if c > 0 else ""
            for c, s in zip(detections[:, 4], detections[:, 5])]
visualize.draw_boxes(
    image, 
    refined_boxes=utils.denorm_boxes(detections[:, :4], image.shape[:2]),
    visibilities=[2] * len(detections),
    captions=captions, title="Detections",
    ax=get_ax(),cmap='gray', vmax=np.percentile(image, 95))


# ### 2.c Step by Step Detection
# 
# Here we dive deeper into the process of processing the detections.

# In[119]:


# Proposals are in normalized coordinates. Scale them
# to image coordinates.
h, w = config.IMAGE_SHAPE[:2]
proposals = np.around(mrcnn["proposals"][0] * np.array([h, w, h, w])).astype(np.int32)

# Class ID, score, and mask per proposal

roi_class_ids = np.argmax(mrcnn["probs"][0], axis=1)
#roi_class_ids = ((mrcnn["probs"][0][:,1]>0.3)*1.).astype(np.int)

roi_scores = mrcnn["probs"][0, np.arange(roi_class_ids.shape[0]), roi_class_ids]
roi_class_names = np.array(dataset.class_names)[roi_class_ids]
roi_positive_ixs = np.where(roi_class_ids > 0)[0]

# How many ROIs vs empty rows?
print("{} Valid proposals out of {}".format(np.sum(np.any(proposals, axis=1)), proposals.shape[0]))
print("{} Positive ROIs".format(len(roi_positive_ixs)))

# Class counts
print(list(zip(*np.unique(roi_class_names, return_counts=True))))


# In[120]:


# Display a random sample of proposals.
# Proposals classified as background are dotted, and
# the rest show their class and confidence score.
limit = 1000
ixs = np.random.randint(0, proposals.shape[0], limit)
captions = ["{} {:.3f}".format(dataset.class_names[c], s) if c > 0 else ""
            for c, s in zip(roi_class_ids[ixs], roi_scores[ixs])]
visualize.draw_boxes(image, boxes=proposals[ixs],
                     visibilities=np.where(roi_class_ids[ixs] > 0, 2, 1),
                     captions=captions, title="ROIs Before Refinement",
                     ax=get_ax(), cmap='gray', vmax=np.percentile(image, 95))


# #### Apply Bounding Box Refinement

# In[126]:


# Class-specific bounding box shifts.
roi_bbox_specific = mrcnn["deltas"][0, np.arange(proposals.shape[0]), roi_class_ids]
log("roi_bbox_specific", roi_bbox_specific)

# Apply bounding box transformations
# Shape: [N, (y1, x1, y2, x2)]
refined_proposals = utils.apply_box_deltas(
    proposals, roi_bbox_specific * config.BBOX_STD_DEV).astype(np.int32)
log("refined_proposals", refined_proposals)

# Show positive proposals
#ids = np.arange(roi_boxes.shape[0])  # Display all
limit = 1000
#ids = np.random.randint(0, len(roi_positive_ixs), limit)  # Display random sample
captions = ["{} {:.3f}".format(dataset.class_names[c], s) if c > 0 else ""
            for c, s in zip(roi_class_ids[roi_positive_ixs][ids], roi_scores[roi_positive_ixs][ids])]
visualize.draw_boxes(image, boxes=proposals[roi_positive_ixs][ids],
                     refined_boxes=refined_proposals[roi_positive_ixs][ids],
                     visibilities=np.where(roi_class_ids[roi_positive_ixs][ids] > 0, 1, 0),
                     captions=captions, title="ROIs After Refinement",
                     ax=get_ax())


# #### Filter Low Confidence Detections

# In[110]:


# Remove boxes classified as background
keep = np.where(roi_class_ids > 0)[0]
print("Keep {} detections:\n{}".format(keep.shape[0], keep))


# In[111]:


# Remove low confidence detections
keep = np.intersect1d(keep, np.where(roi_scores >= config.DETECTION_MIN_CONFIDENCE)[0])
print("Remove boxes below {} confidence. Keep {}:\n{}".format(
    config.DETECTION_MIN_CONFIDENCE, keep.shape[0], keep))


# #### Per-Class Non-Max Suppression

# In[114]:


# Apply per-class non-max suppression
pre_nms_boxes = refined_proposals[keep]
pre_nms_scores = roi_scores[keep]
pre_nms_class_ids = roi_class_ids[keep]

nms_keep = []
for class_id in np.unique(pre_nms_class_ids):
    # Pick detections of this class
    ixs = np.where(pre_nms_class_ids == class_id)[0]
    # Apply NMS
    class_keep = utils.non_max_suppression(pre_nms_boxes[ixs], 
                                            pre_nms_scores[ixs],
                                          config.DETECTION_NMS_THRESHOLD)
    # Map indicies
    class_keep = keep[ixs[class_keep]]
    nms_keep = np.union1d(nms_keep, class_keep)
    print("{:22}: {} -> {}".format(dataset.class_names[class_id][:20], 
                                   keep[ixs], class_keep))

keep = np.intersect1d(keep, nms_keep).astype(np.int32)
print("\nKept after per-class NMS: {}\n{}".format(keep.shape[0], keep))


# In[113]:


config.DETECTION_NMS_THRESHOLD=0.3


# In[115]:


# Show final detections
ixs = np.arange(len(keep))  # Display all
# ixs = np.random.randint(0, len(keep), 10)  # Display random sample
captions = ["{} {:.3f}".format(dataset.class_names[c], s) if c > 0 else ""
            for c, s in zip(roi_class_ids[keep][ixs], roi_scores[keep][ixs])]
visualize.draw_boxes(
    image, boxes=proposals[keep][ixs],
    refined_boxes=refined_proposals[keep][ixs],
    visibilities=np.where(roi_class_ids[keep][ixs] > 0, 1, 0),
    captions=captions, title="Detections after NMS",
    ax=get_ax(),vmax=np.percentile(image, 95), cmap='gray')


# ## Stage 3: Generating Masks
# 
# This stage takes the detections (refined bounding boxes and class IDs) from the previous layer and runs the mask head to generate segmentation masks for every instance.

# ### 3.a Mask Targets
# 
# These are the training targets for the mask branch

# In[98]:


display_images(np.transpose(gt_mask, [2, 0, 1]), cmap="Blues")


# ### 3.b Predicted Masks

# In[28]:


# Get predictions of mask head
mrcnn = model.run_graph([image], [
    ("detections", model.keras_model.get_layer("mrcnn_detection").output),
    ("masks", model.keras_model.get_layer("mrcnn_mask").output),
])

# Get detection class IDs. Trim zero padding.
det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
det_count = np.where(det_class_ids == 0)[0][0]
det_class_ids = det_class_ids[:det_count]

print("{} detections: {}".format(
    det_count, np.array(dataset.class_names)[det_class_ids]))


# In[29]:


# Masks
det_boxes = utils.denorm_boxes(mrcnn["detections"][0, :, :4], image.shape[:2])
det_mask_specific = np.array([mrcnn["masks"][0, i, :, :, c] 
                              for i, c in enumerate(det_class_ids)])
det_masks = np.array([utils.unmold_mask(m, det_boxes[i], image.shape)
                      for i, m in enumerate(det_mask_specific)])
log("det_mask_specific", det_mask_specific)
log("det_masks", det_masks)


# In[30]:


display_images(det_mask_specific[:4] * 255, cmap="Blues", interpolation="none")


# In[31]:


display_images(det_masks[:4] * 255, cmap="Blues", interpolation="none")


# ## Visualize Activations
# 
# In some cases it helps to look at the output from different layers and visualize them to catch issues and odd patterns.

# In[32]:


# Get activations of a few sample layers
activations = model.run_graph([image], [
    ("input_image",        tf.identity(model.keras_model.get_layer("input_image").output)),
    ("res2c_out",          model.keras_model.get_layer("res2c_out").output),
    ("res3c_out",          model.keras_model.get_layer("res3c_out").output),
    ("res4w_out",          model.keras_model.get_layer("res4w_out").output),  # for resnet100
    ("rpn_bbox",           model.keras_model.get_layer("rpn_bbox").output),
    ("roi",                model.keras_model.get_layer("ROI").output),
])


# In[33]:


# Input image (normalized)
_ = plt.imshow(modellib.unmold_image(activations["input_image"][0],config))


# In[34]:


# Backbone feature map
display_images(np.transpose(activations["res2c_out"][0,:,:,:4], [2, 0, 1]), cols=4)


# In[ ]:




