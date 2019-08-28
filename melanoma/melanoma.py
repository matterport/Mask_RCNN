import os
import sys
import datetime
import numpy as np
import skimage.draw
from PIL import Image

sys.path.insert(0, ".")
ROOT_DIR="."

sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils

image_directory="./images/"
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
mask_dir="./masks/"
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

class MelanomaConfig(Config):

    NAME="Melanoma"
    STEPS_PER_EPOCH = 100
    NUM_CLASSES = 1+1
    IMAGE_RESIZE_MODE = "square"
#     IMAGE_MIN_DIM = 512
#     IMAGE_MAX_DIM = 4096
    IMAGE_CHANNEL_COUNT = 3
    DETECTION_MIN_CONFIDENCE = 0.9
    STEPS_PER_EPOCH = 200
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class MelanomaDataset(utils.Dataset):
   
    def load_melanoma(self,dataset_dir,subset):

        self.add_class("melanoma",1,"melanoma")
        assert subset in ["train", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)
        file_names=[]
        for val in os.listdir(dataset_dir):
            file_names.append(val)
            image_path = os.path.join(dataset_dir, val)
            try:
                im = Image.open(image_path)
                width, height = im.size
            except:
                pass

            self.add_image(
                "melanoma",
                image_id=val[:-4],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                number=len(file_names))

    def load_mask(self,image_id):

        # If not a melanoma dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "melanoma":
            return super(self.__class__, self).load_mask(image_id)
        
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], 1],
                        dtype=np.uint8)
        mask_path = os.path.join(mask_dir, info["id"].split('.')[0]+"_segmentation.png")
        mask = skimage.io.imread(mask_path, as_gray=True)
        mask=np.expand_dims(mask, axis=2)

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
#     """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "melanoma":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)



config = MelanomaConfig()
config.display()

model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=ROOT_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    weights_path = COCO_WEIGHTS_PATH
        # Download weights file
    if not os.path.exists(weights_path):
        utils.download_trained_weights(weights_path)
    model.load_weights(weights_path, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

dataset_train=MelanomaDataset()
dataset_train.load_melanoma(image_directory,"train")
dataset_train.prepare()

dataset_val=MelanomaDataset()
dataset_val.load_melanoma(image_directory,"test")
dataset_val.prepare()
print("Training network heads")

model.train(dataset_train,dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=2,
                layers='heads')
print("Training entire network")
model.train(dataset_train,dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='all')