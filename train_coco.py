"""
Mobile Mask R-CNN Train & Eval Script
for Training on the COCO Dataset

written by github.com/GustavZ

to use tensorboard run inside model_dir with file "events.out.tfevents.123":
tensorboard --logdir="$(pwd)"

keras h5 to tensorflow pb file:
python keras_to_tensorflow.py -input_model_file saved_model_mrcnn_eval -output_model_file model.pb -num_outputs=7
"""

# Import Packages
import os
import sys
import imgaug

# Import Mobile Mask R-CNN
from mmrcnn import model as modellib, utils
import coco

# Paths
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_DIR = os.path.join(ROOT_DIR, 'data/coco')
WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")
DEFAULT_WEIGHTS_DIR = os.path.join(MODEL_DIR, 'cocoperson20180423T1626/mask_rcnn_cocoperson_0160.h5')

# Model
config = coco.CocoConfig()
config.display()
model = modellib.MaskRCNN(mode="training", model_dir = MODEL_DIR, config=config)

# Weights
model_path = model.get_imagenet_weights()
#model_path = model.find_last()[1]
#model_path = DEFAULT_WEIGHTS_DIR
print("> Loading weights from {}".format(model_path))
model.load_weights(model_path, by_name=True)
model.keras_model.summary()

# Dataset
class_names = ['person']
dataset_train = coco.CocoDataset()
dataset_train.load_coco(COCO_DIR, "train", class_names=class_names)
dataset_train.prepare()
dataset_val = coco.CocoDataset()
dataset_val.load_coco(COCO_DIR, "val", class_names=class_names)
dataset_val.prepare()

# Training - Config
augmentation = imgaug.augmenters.Fliplr(0.5)

# Training - Stage 1
print("> Training network heads")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=160,
            layers='heads',
            augmentation=augmentation)

# Training - Stage 2
# Finetune layers  stage 4 and up
print("> Fine tune {} stage 4 and up".format(config.BACKBONE))
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=120,
            layers="11M+",
            augmentation=augmentation)

# Training - Stage 3
# Fine tune all layers
print("> Fine tune all layers")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=40,
            layers='all',
            augmentation=augmentation)

# Save Model
if not os.path.isdir(WEIGHTS_DIR):
    os.mkdirs(WEIGHTS_DIR)
model_path = os.path.join(WEIGHTS_DIR, "mobile_mask_rcnn_cocoperson.h5")
model.keras_model.save(model_path)
