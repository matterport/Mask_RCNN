import os
from os.path import dirname, abspath
import tensorflow as tf
import glob
import sys
import datetime
import random
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('QT5Agg')
from cucu_config import cucumberConfig
from cucu_config import cucuConfForTrainingSession as config
from project_assets.cucu_classes import genDataset, realDataset, CucuLogger, project_paths
from PIL import Image
import json
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from shutil import copyfile, copytree
from keras.callbacks import *
import math

def setRootPathForCurrentSession():
    return dirname(dirname(os.path.realpath(__file__)))
def constructContainerPath():
    # create a container for training result per exexution of cucu_train.py
    CONTAINER_ROOT_DIR = ROOT_DIR + "/cucu_train/trainResultContainers/"
    now = datetime.datetime.now()
    return CONTAINER_ROOT_DIR + 'trainResults_{month}-{day}-{hour}'.format(month=now.month, day=now.day, hour=now.hour)
def cloneDataSetIntoSessionFolderTree():
    copytree(ROOT_DIR+ '/cucu_train/project_dataset', currentContainerDir+ '/project_dataset')
    return
def initiateAllPathsForCurrentSession(currentContainerDir,currentSessionInitialWeights):
    # create centralized class for used paths during current session
    cucuPaths = project_paths(
    projectRootDir=ROOT_DIR,
    TensorboardDir=        os.path.join(currentContainerDir, "TensorBoardGraphs"),
    trainedModelsDir=      os.path.join(currentContainerDir, "trained_models"),
    visualizeEvaluationsDir = os.path.join(currentContainerDir, "visualizeEvaluations"),
    currSessionInitialModelWeights=         currentSessionInitialWeights,
    trainDatasetDir=       os.path.join(currentContainerDir, "project_dataset/generated/train_data"), #asher todo: generalize paths ( 'generated' inside pre-defined path is bad)
    valDatasetDir=         os.path.join(currentContainerDir, "project_dataset/generated/valid_data"),
    testDatasetDir=        os.path.join(currentContainerDir, "project_dataset/test_data"),
    trainResultContainer=  currentContainerDir,
    trainOutputLog      =  currentContainerDir)
    return cucuPaths

#todo: simplify
def createFoldersForModelWeightsAndVizualizations():
    try:
        original_umask = os.umask(0)
        os.makedirs(cucuPaths.trainedModelsDir, mode=0o777)
        os.makedirs(cucuPaths.visualizeEvaluationsDir, mode=0o777)
    finally:
        os.umask(original_umask)
    return
def prepareCallbackForCurrentSession():
    def scheduleLearningRate(epoch, lr):
        return lr*0.5
    
    return [EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1, mode='auto'),
            LearningRateScheduler(scheduleLearningRate, verbose=1)]

#handle paths,new folders and logger
ROOT_DIR = setRootPathForCurrentSession()
os.chmod(ROOT_DIR, mode=0o777)
currentContainerDir = constructContainerPath()
#todo: get it inside function below
print('Enter full path for initial model weights:')
currentSessionInitialWeights=input()
cucuPaths = initiateAllPathsForCurrentSession(currentContainerDir,currentSessionInitialWeights)
sys.path.append(cucuPaths.projectRootDir)  # To find local version of the library

createFoldersForModelWeightsAndVizualizations()

sys.stdout = CucuLogger(sys.stdout, cucuPaths.trainOutputLog + "/sessionLogger.txt")
sys.stdout.getFromUserCurrentSessionSpecs()
#print current session configuration to logger
config.display()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=cucuPaths.TensorboardDir)

# load initial weights
weightPath=cucuPaths.currSessionInitialModelWeights
model.load_weights(weightPath, by_name=True,
                exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                        "mrcnn_bbox", "mrcnn_mask"])
print("loaded weights from path:", weightPath)

#create directory to hold inside samples of images we pass to model during training
os.mkdir(cucuPaths.visualizeEvaluationsDir + "/SamplesOfTrainDataset")


#create path dictionaries
trainCategoryPathsDict = {}
trainCategoryPathsDict['BG']         = cucuPaths.trainDatasetDir + '/background_folder/1024'
trainCategoryPathsDict['cucumber']   = cucuPaths.trainDatasetDir + '/cucumbers_objects'
trainCategoryPathsDict['leaf']       = cucuPaths.trainDatasetDir + '/leaves_objects'
trainCategoryPathsDict['flower']     = cucuPaths.trainDatasetDir + '/flower_objects'
trainCategoryPathsDict['stem']       = cucuPaths.trainDatasetDir + '/stems_objects'

validCategoryPathsDict = {}
validCategoryPathsDict['BG']         = cucuPaths.valDatasetDir + '/background_folder/1024'
validCategoryPathsDict['cucumber']   = cucuPaths.valDatasetDir + '/cucumbers_objects'
validCategoryPathsDict['leaf']       = cucuPaths.valDatasetDir + '/leaves_objects'
validCategoryPathsDict['flower']     = cucuPaths.valDatasetDir + '/flower_objects'
validCategoryPathsDict['stem']       = cucuPaths.valDatasetDir + '/stems_objects'

# add custom callbacks if needed as a preparation to training model
custom_callbacks= prepareCallbackForCurrentSession()

# start training loop
for _ in range(config.EPOCHS_ROUNDS):
    # Training dataset
    # asher todo: add a choice from which dataset to generate
    dataset_train = realDataset()
    dataset_train.load_dataset(os.path.join(cucuPaths.trainDatasetDir, "annotations.json"), cucuPaths.trainDatasetDir)


    # Validation dataset
    dataset_val = realDataset()
    dataset_val.load_dataset(os.path.join(cucuPaths.valDatasetDir, "annotations.json"), cucuPaths.valDatasetDir)
    dataset_val.prepare()

    #show n random image&mask train examples
    n = 1
    image_ids = np.random.choice(dataset_train.image_ids, n)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        print(image.shape)
        visualize.display_top_masks( image, mask, class_ids, \
        dataset_train.class_names,cucuPaths.visualizeEvaluationsDir + "/SamplesOfTrainDataset/" + "image_" + str(image_id) +".png", 2)

    model.train(dataset_train, dataset_val, learning_rate= config.LEARNING_RATE, epochs=config.EPOCHS,\
                            custom_callbacks=custom_callbacks, layers="heads",verbose=1)

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

    # PREPARE NEW CONFIG FOR NEXT ROUND:
    config.OBJECTS_IOU_THRESHOLD = min(config.OBJECTS_IOU_THRESHOLD*3, 0.5)
    # config.MIN_GENERATED_OBJECTS = min(math.ceil(config.MIN_GENERATED_OBJECTS + 5), config.MAX_GT_INSTANCES)
    # config.MAX_GENERATED_OBJECTS = min(math.ceil(config.MAX_GENERATED_OBJECTS + 5), config.MAX_GT_INSTANCES)
    # config.LEARNING_RATE = 0.01



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
tests_location = cucuPaths.testDatasetDir
for filename in sorted(os.listdir(tests_location)):
    
    testImage = os.path.join(tests_location,filename)
    try:
        t = cv2.cvtColor(cv2.imread(testImage), cv2.COLOR_BGR2RGB)
    except Exception as e:
        print("error: {}, \n probably a non image file is in the directory".format(e))
        continue
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
os.mkdir(cucuPaths.visualizeEvaluationsDir + "/draw_boxes")
os.mkdir(cucuPaths.visualizeEvaluationsDir + "/masks_detections")
os.mkdir(cucuPaths.visualizeEvaluationsDir + "/activationsImages")


# in future we want to generate from dataset_test!
dataset = dataset_val

image_ids = np.random.choice(dataset.image_ids, 20)
for image_id in image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                        dataset.image_reference(image_id)))
    # Run object detection
    results = model.detect([image], verbose=1)

    # Display results
    ax = get_ax(1)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'], ax=ax,title="Predictions", \
                                savePath=cucuPaths.visualizeEvaluationsDir + "/display_instances/" + "display_instances_" + "image_" + str(image_id) +".png")
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    # Load random image and mask.
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    # Compute Bounding box
    bbox = utils.extract_bboxes(mask)

    # Display image and additional stats
    print("image_id ", image_id, dataset.image_reference(image_id))
    log("image", image)
    log("mask", mask)
    log("class_ids", class_ids)
    log("bbox", bbox)
    # Display image and instances
    visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names, savePath=cucuPaths.visualizeEvaluationsDir + "/display_instances/" + "display_instances2_" + "image_" + str(image_id) +".png")

    # Draw precision-recall curve
    AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                            r['rois'], r['class_ids'], r['scores'], r['masks'])
    visualize.plot_precision_recall(AP, precisions, recalls,savePath=cucuPaths.visualizeEvaluationsDir + "/plot_precision_recall/" + "plot_precision_recall_" + "image_" + str(image_id) +".png")

    # Grid of ground truth objects and their predictions
    visualize.plot_overlaps(gt_class_id, r['class_ids'], r['scores'],
                        overlaps, dataset.class_names,savePath=cucuPaths.visualizeEvaluationsDir + "/plot_overlaps/" + "plot_overlaps" + "image_" + str(image_id) +".png")


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
    # Display positive anchors before refinement (dotted) and
    # after refinement (solid).
    visualize.draw_boxes(image, boxes=positive_anchors,title="Display positive anchors before refinement (dotted)", refined_boxes=refined_anchors, ax=get_ax(),\
                        savePath=cucuPaths.visualizeEvaluationsDir + "/draw_boxes/" + "draw_boxes_beforeAndAfterRefine_" + "image_" + str(image_id) +".png")


    # asher todo: this module stilll doesn't work
    # Run RPN sub-graph
    pillar = model.keras_model.get_layer("mrcnn_bbox").output  # node to start searching from

    # TF 1.4 and 1.9 introduce new versions of NMS. Search for all names to support TF 1.3~1.10
    nms_node = model.ancestor(model.keras_model.get_layer("mrcnn_bbox").output, "ROI/rpn_non_max_suppression:0")
    if nms_node is None:
        nms_node = model.ancestor(model.keras_model.get_layer("mrcnn_bbox").output, "ROI/rpn_non_max_suppression/NonMaxSuppressionV2:0")
    # if nms_node is None: #TF 1.9-1.10
    #     nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV3:0")

    rpn = model.run_graph([image], [
        ("mrcnn_bbox", model.keras_model.get_layer("mrcnn_bbox").output),
        # ("pre_nms_anchors", model.ancestor(pillar, "ROI/pre_nms_anchors:0")),
        # ("refined_anchors", model.ancestor(pillar, "mrcnn_bbox_fc")),
        # ("refined_anchors_clipped", model.ancestor(pillar, "RPN/ROI/refined_anchors_clipped:0"))
        # ,("post_nms_anchor_ix", nms_node)
        # ,("rois", model.keras_model.get_layer("TrainGroundTruths/proposal_targets/rois").output),
    ])
    # Show top anchors by score (before refinement)
    limit = 100
    sorted_anchor_ids = np.argsort(rpn['mrcnn_bbox'][:,:,1].flatten())[::-1]
    visualize.draw_boxes(image, boxes=model.anchors[sorted_anchor_ids[:limit]], ax=get_ax(),savePath=cucuPaths.visualizeEvaluationsDir + "/draw_boxes/" + "draw_boxes_topAnchorsNotRefined_" + "image_" + str(image_id) +".png")

    # Show top anchors with refinement. Then with clipping to image boundaries
    # limit = 50
    # ax = get_ax(1, 2)
    # pre_nms_anchors = utils.denorm_boxes(rpn["pre_nms_anchors"][0], image.shape[:2])
    # refined_anchors = utils.denorm_boxes(rpn["refined_anchors"][0], image.shape[:2])
    # refined_anchors_clipped = utils.denorm_boxes(rpn["refined_anchors_clipped"][0], image.shape[:2])
    # visualize.draw_boxes(image, boxes=pre_nms_anchors[:limit],
    #                     refined_boxes=refined_anchors[:limit], ax=ax[0],savePath=cucuPaths.visualizeEvaluationsDir + "/draw_boxes/" + "draw_boxes_topAnchorsRefinedWithoutClip_" + "image_" + str(image_id) +".png")
    # visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[:limit], ax=ax[1],savePath=cucuPaths.visualizeEvaluationsDir + "/draw_boxes/" + "draw_boxes_topAnchorsRefinedWithClip_" + "image_" + str(image_id) +".png")

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
    # Masks
    det_boxes = utils.denorm_boxes(mrcnn["detections"][0, :, :4], image.shape[:2])
    det_mask_specific = np.array([mrcnn["masks"][0, i, :, :, c] for i, c in enumerate(det_class_ids)])
    det_masks = np.array([utils.unmold_mask(m, det_boxes[i], image.shape)for i, m in enumerate(det_mask_specific)])
    log("det_mask_specific", det_mask_specific)
    log("det_masks", det_masks)
    visualize.display_images(det_masks[:4] * 255, cmap="Blues", interpolation="none", savePath=cucuPaths.visualizeEvaluationsDir + "/masks_detections/" + "masks_detections_" + "image_" + str(image_id) +".png" )

    # Get activations of a few sample layers
    activations = model.run_graph([image], [
        ("input_image",        model.keras_model.get_layer("input_image").output),
        ("res2a_out",          model.keras_model.get_layer("res2a_out").output)  # for resnet100
        # ,("rpn_bbox",           model.keras_model.get_layer("rpn_bbox").output),
        # ("roi",                model.keras_model.get_layer("ROI").output),
    ])
    # Input image (normalized)
    _ = plt.imshow(modellib.unmold_image(activations["input_image"][0],config))
    plt.savefig(cucuPaths.visualizeEvaluationsDir + "/activationsImages/" + "normInputImage" + "image_" + str(image_id) +".png")
    # Backbone feature map
    visualize.display_images(np.transpose(activations["res2a_out"][0,:,:,:4], [2, 0, 1]), savePath=cucuPaths.visualizeEvaluationsDir + "/activationsImages/" + "activationRes2aImage" + "image_" + str(image_id) +".png")
    # Get activations of a few sample layers
    activations = model.run_graph([image], [
        ("input_image",        model.keras_model.get_layer("input_image").output),
        ("res3a_out",          model.keras_model.get_layer("res3a_out").output)  # for resnet100
        # ,("rpn_bbox",           model.keras_model.get_layer("rpn_bbox").output),
        # ("roi",                model.keras_model.get_layer("ROI").output),
    ])
    # Input image (normalized)
    _ = plt.imshow(modellib.unmold_image(activations["input_image"][0],config))
    plt.savefig(cucuPaths.visualizeEvaluationsDir + "/activationsImages/" + "normInputImage" + "image_" + str(image_id) +".png")
    # Backbone feature map
    visualize.display_images(np.transpose(activations["res3a_out"][0,:,:,:4], [2, 0, 1]), savePath=cucuPaths.visualizeEvaluationsDir + "/activationsImages/" + "activationRes3aImage" + "image_" + str(image_id) +".png")




# Compute VOC-style Average Precision
def compute_batch_ap(image_ids):
    APs = []
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, config,
                                   image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                              r['rois'], r['class_ids'], r['scores'], r['masks'])
        APs.append(AP)
    return APs

# Pick a set of random images
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = compute_batch_ap(image_ids)
print("mAP @ IoU=50: ", np.mean(APs))










