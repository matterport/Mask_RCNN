"""
Mobile Mask R-CNN Train & Eval Script
for Training on the COCO Dataset

written by github.com/GustavZ

to use tensorboard run inside model_dir with file "events.out.tfevents.123":
tensorboard --logdir="$(pwd)"

keras h5 to tensorflow pb file:
python keras_to_tensorflow.py -input_model_file saved_model_mrcnn_eval -output_model_file model.pb -num_outputs=7
"""

## Import Packages
import os
import sys
import imgaug

## Import Mobile Mask R-CNN
from mmrcnn import model as modellib, utils
import coco

## Paths
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_DIR = os.path.join(ROOT_DIR, 'data/coco')
WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")
DEFAULT_WEIGHTS_DIR = "/home/gustav/workspace/Mobile_Mask_RCNN/logs/cocoperson20180425T1415/mask_rcnn_cocoperson_0160.h5"

## Dataset
class_names = ['person']  # all classes: None
dataset_train = coco.CocoDataset()
dataset_train.load_coco(COCO_DIR, "train", class_names=class_names)
dataset_train.prepare()
dataset_val = coco.CocoDataset()
dataset_val.load_coco(COCO_DIR, "val", class_names=class_names)
dataset_val.prepare()

## Model
config = coco.CocoConfig()
config.display()
model = modellib.MaskRCNN(mode="training", model_dir = MODEL_DIR, config=config)
model.keras_model.summary()

## Weights
#model_path = model.get_imagenet_weights()
#model_path = model.find_last()[1]
model_path = DEFAULT_WEIGHTS_DIR
print("> Loading weights from {}".format(model_path))
model.load_weights(model_path, by_name=True)

## Training - Config
starting_epoch = model.epoch
epoch = dataset_train.dataset_size // (config.STEPS_PER_EPOCH * config.BATCH_SIZE)
epochs_heads = 2 * epoch + starting_epoch
epochs_stage4 = 2 * epoch + starting_epoch
epochs_all = 2 * epoch + starting_epoch
augmentation = imgaug.augmenters.Fliplr(0.5)

## Training - Stage 1
print("> Training network heads")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=epochs_heads,
            layers='heads',
            augmentation=augmentation)

## Training - Stage 2
# Finetune layers  stage 4 and up
print("> Fine tune {} stage 4 and up".format(config.BACKBONE))
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=epoch_heads + epochs_stage4,
            layers="4+",
            augmentation=augmentation)

## Training - Stage 3
# Fine tune all layers
print("> Fine tune all layers")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=epochs_heads + epochs_stage4 + epochs_all,
            layers='all',
            augmentation=augmentation)

## Save Model
#if not os.path.isdir(WEIGHTS_DIR):
#    os.mkdirs(WEIGHTS_DIR)
#model_path = os.path.join(WEIGHTS_DIR, "mobile_mask_rcnn_cocoperson.h5")
#model.keras_model.save(model_path)
