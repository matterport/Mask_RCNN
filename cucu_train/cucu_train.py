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
from cucu_config import cucumberConfig, InferenceConfig
from cucu_config import cucuConfForTrainingSession as config
from cucu_config import globalObjectCategories
from project_assets.cucu_classes import genDataset, CucuLogger, project_paths,HybridDataset, realDataset
from project_assets.cucu_utils import get_ax

from PIL import Image
import json
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from shutil import copyfile, copytree
from keras.callbacks import EarlyStopping,LearningRateScheduler
import math

def setRootPathForCurrentSession():
    rootDir = dirname(dirname(os.path.realpath(__file__)))
    os.chmod(rootDir, mode=0o777)
    return rootDir

def constructContainerPath():
    # create a container for training result per exexution of cucu_train.py
    CONTAINER_ROOT_DIR = rootDir + "/cucu_train/trainResultContainers/"
    now = datetime.datetime.now()
    return CONTAINER_ROOT_DIR + 'trainResults_{month}-{day}-{hour}'.format(month=now.month, day=now.day, hour=now.hour)
def cloneDataSetIntoSessionFolderTree():
    copytree(rootDir+ '/cucu_train/project_dataset', currentContainerDir+ '/project_dataset')
    return

def initiateAllPathsForCurrentSession(rootDir,currentContainerDir):
    #asher todo: change to get from user
    # currentSessionInitialWeights = input()
    currentSessionInitialWeights =rootDir +'/mask_rcnn_coco.h5'
    # create centralized class for used paths during current session
    cucuPaths = project_paths(
    projectRootDir=rootDir,
    currSessionInitialModelWeights=         currentSessionInitialWeights,
    TensorboardDir=        os.path.join(currentContainerDir, "TensorBoardGraphs"),
    trainedModelsDir=      os.path.join(currentContainerDir, "trained_models"),
    visualizeEvaluationsDir = os.path.join(currentContainerDir, "visualizeEvaluations"),

    #dataset paths
    GenDatasetDir=       os.path.join(currentContainerDir, "../../project_dataset/generated/"), 
    RealDatasetDir=     os.path.join(currentContainerDir, "../../project_dataset/real/"), 
    TestDatasetDir=        os.path.join(currentContainerDir, "../../project_dataset/test/test_data/"),
    #GenDatasetDir=       os.path.join(rootDir, "/cucu_train/project_dataset/generated/"), 
    #RealDatasetDir=     os.path.join(rootDir, "/cucu_train/project_dataset/real/"), 
    #TestDatasetDir=        os.path.join(rootDir, "/cucu_train/project_dataset/test/test_data/"),
    
    trainResultContainer=  currentContainerDir,
    trainOutputLog      =  currentContainerDir)

    sys.path.append(cucuPaths.projectRootDir)   # To find local version of the library

    return cucuPaths
def createFoldersForModelWeightsAndVizualizations():
    try:
        original_umask = os.umask(0)
        os.makedirs(cucuPaths.trainedModelsDir, mode=0o777)
        os.makedirs(cucuPaths.visualizeEvaluationsDir, mode=0o777)

        #create directory to hold inside samples of images we pass to model during training
        os.mkdir(cucuPaths.visualizeEvaluationsDir + "/SamplesOfTrainDataset")
        
        #create container directories per function calls from Visualize module
        os.mkdir(cucuPaths.visualizeEvaluationsDir + "/display_instances")
        os.mkdir(cucuPaths.visualizeEvaluationsDir + "/plot_precision_recall")
        os.mkdir(cucuPaths.visualizeEvaluationsDir + "/plot_overlaps")
        os.mkdir(cucuPaths.visualizeEvaluationsDir + "/draw_boxes")
        os.mkdir(cucuPaths.visualizeEvaluationsDir + "/masks_detections")
        os.mkdir(cucuPaths.visualizeEvaluationsDir + "/activationsImages")

        
    finally:
        os.umask(original_umask)
    return
def prepareCallbackForCurrentSession():
    def scheduleLearningRate(epoch, lr):
        return lr*0.8
    
    return [EarlyStopping(monitor='val_loss', min_delta=0.01, patience=2, verbose=1, mode='auto'),
            LearningRateScheduler(scheduleLearningRate, verbose=1)]

def createSessionLoggerToCollectPrintOutputs():
    sys.stdout = CucuLogger(sys.stdout, cucuPaths.trainOutputLog + "/sessionLogger.txt")
    sys.stdout.getFromUserCurrentSessionSpecs() 

def initiateTensorflowModel():
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=cucuPaths.TensorboardDir)

    # load initial weights
    weightPath=cucuPaths.currSessionInitialModelWeights
    model.load_weights(weightPath, by_name=True,
                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                            "mrcnn_bbox", "mrcnn_mask"])
    print("loaded weights from path:", weightPath)
    return model

if __name__ == "__main__":
    #handle paths,new folders and logger
    rootDir = setRootPathForCurrentSession()
    currentContainerDir = constructContainerPath()
    cucuPaths = initiateAllPathsForCurrentSession(rootDir,currentContainerDir)
    createFoldersForModelWeightsAndVizualizations()

    #cloneDataSetIntoSessionFolderTree()
    createSessionLoggerToCollectPrintOutputs()

    #print current session configuration to logger
    config.display()

    model = initiateTensorflowModel()
    # add custom callbacks if needed as a preparation to training model
    custom_callbacks= prepareCallbackForCurrentSession()

    # start training loop
    #asher todo: EPOCHS_ROUNDS can be deleted
    for _ in range(config.EPOCHS_ROUNDS):
        ## Training dataset
        #dataset_train = HybridDataset(config,cucuPaths.GenDatasetDir,cucuPaths.RealDatasetDir,dataSetType = 'train',augmentedCategory = 'cucumber')
        #dataset_train.load_dataset()
        #dataset_train.prepare()
        ## Validation dataset
        #dataset_val = HybridDataset(config,cucuPaths.GenDatasetDir,cucuPaths.RealDatasetDir,dataSetType = 'valid',augmentedCategory = 'cucumber')
        #dataset_val.load_dataset()
        #dataset_val.prepare()
        
        # Training dataset
        #dataset_train = genDataset(config,cucuPaths.GenDatasetDir,datasetType = 'train')
        #dataset_train.load_shapes(config.GEN_TRAIN_SET_SIZE, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
        #dataset_train.prepare()
        
        # Validation dataset
        #dataset_val = genDataset(config,cucuPaths.GenDatasetDir,datasetType = 'valid')
        #dataset_val.load_shapes(config.GEN_TRAIN_SET_SIZE, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
        #dataset_val.prepare()

        dataset_train = realDataset()
        dataset_train.load_dataset(cucuPaths.RealDatasetDir + '1024/cucumber/train/augmented/annotations.json',cucuPaths.RealDatasetDir + '/1024/cucumber/train/augmented')
        dataset_train.prepare()

        dataset_val = realDataset()
        dataset_val.load_dataset(cucuPaths.RealDatasetDir + '/1024/cucumber/valid/annotations.json',cucuPaths.RealDatasetDir + '/1024/cucumber/valid')
        dataset_val.prepare()
        # In[ ]:


        #store n random image&mask train examples
        n = 20
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



    ###################### TEST TRAINED MODEL PART########################################


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
    tests_location = cucuPaths.TestDatasetDir
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


    # in future we want to generate from dataset_test!
    dataset = dataset_val

    image_ids = np.random.choice(dataset.image_ids, 20)

    for image_id in image_ids:
        
        
        ################# todo: function ##########################################

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



        ################# todo: function ##########################################

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





        ################# todo: function ##########################################
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




        ################# todo: function ##########################################
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





        ################# todo: function ##########################################
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

    # Pick a set of random images


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

    image_ids = np.random.choice(dataset_val.image_ids, 10)
    APs = compute_batch_ap(image_ids)
    print("mAP @ IoU=5{}:{} ".format(config.DETECTION_NMS_THRESHOLD,np.mean(APs)) )















