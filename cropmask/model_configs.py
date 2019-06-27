from cropmask.mrcnn.config import Config
import numpy as np

############################################################
#  Configurations
############################################################
class LandsatConfig(Config):
    """Configuration for training on landsat imagery. 
     Overrides values specific to Landsat center pivot imagery.
    
    Descriptive documentation for each attribute is at
    https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py
    
    There are many more hyperparameters to edit than are set in this subclass"""

    def __init__(self, N):
        """Set values of computed attributes. Channel dimension is overriden, 
        replaced 3 with N as per this guideline: https://github.com/matterport/Mask_RCNN/issues/314
        THERE MAY BE OTHER CODE CHANGES TO ACCOUNT FOR 3 vs N channels. See other 
        comments."""
        # https://github.com/matterport/Mask_RCNN/wiki helpful for N channels
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, N])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, N])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES
        self.CHANNELS_NUM = N

    LEARNING_RATE = 0.0003

    # Image mean from inspect_data ipynb (preprocess.py differs for some reason, only slightly by 1os of digits or 1s of digits)
    MEAN_PIXEL = np.array([711.1, 995.51, 1097.56])

    # Give the configuration a recognizable name
    NAME = "landsat-512-cp"

    # Batch size is 4 (GPUs * images/GPU).
    # Keras 2.1.6 works for multi-gpu but takes longer than single GPU currently
    GPU_COUNT = 1
    IMAGES_PER_GPU = 3

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + ag

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0.7

    # Use small images for faster training. Determines the image shape.
    # From build() in model.py
    # Exception("Image size must be dividable by 2 at least 6 times "
    #     "to avoid fractions when downscaling and upscaling."
    #    "For example, use 256, 320, 384, 448, 512, ... etc. "
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    #     IMAGE_MIN_SCALE = 2.0

    # anchor side in pixels, determined using inspect_crop_data.ipynb. can specify more or less scales
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # for cp

    # Aim to allow ROI sampling to pick 33% positive ROIs. This is always 33% in inspect_data nb, unsure if that is accurate.
    TRAIN_ROIS_PER_IMAGE = 600

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128  # 64

    # Unsure what best step size is but nucleus used 100
    STEPS_PER_EPOCH = 100

    # reduces the max number of field instances
    # MAX_GT_INSTANCES = 29 # for smallholder determined using inspect_crop_data.ipynb
    MAX_GT_INSTANCES = 195  # for cp determined using inspect_crop_data.ipynb

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet50"

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (28, 28)  # (height, width) of the mini-mask

    # Loss weights for more precise optimization. It has been suggested that mrcnn_mask_loss should be weighted higher
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.0,
        "rpn_bbox_loss": 1.0,
        "mrcnn_class_loss": 1.0,
        "mrcnn_bbox_loss": 1.0,
        "mrcnn_mask_loss": 3.0,
    }


class LandsatInferenceConfig(LandsatConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imagery for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7
    #must be set to what pretrained resnet model expects, see https://github.com/matterport/Mask_RCNN/issues/1291
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)


class WV2Config(Config):
    """Configuration for training on worldview-2 imagery. 
     Overrides values specific to WV2.
    
    Descriptive documentation for each attribute is at
    https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py
    
    There are many more hyperparameters to edit than are set in this subclass"""

    def __init__(self, N):
        """Set values of computed attributes. Channel dimension is overriden, 
        replaced 3 with N as per this guideline: https://github.com/matterport/Mask_RCNN/issues/314
        THERE MAY BE OTHER CODE CHANGES TO ACCOUNT FOR 3 vs N channels. See other 
        comments."""
        # https://github.com/matterport/Mask_RCNN/wiki helpful for N channels
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, N])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, N])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES
        self.CHANNELS_NUM = N

    LEARNING_RATE = 0.00005

    # Image mean (RGBN RGBN) from WV2_MRCNN_PRE.ipynb
    # filling with N values, need to compute mean of each channel
    # values are for gridded wv2 no partial grids
    MEAN_PIXEL = np.array([225.25, 308.74, 184.93])

    # Give the configuration a recognizable name
    NAME = "wv2-1024-cp"

    # Batch size is 4 (GPUs * images/GPU).
    # Keras 2.1.6 works for multi-gpu but takes longer than single GPU currently
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + ag

    # Use small images for faster training. Determines the image shape.
    # From build() in model.py
    # Exception("Image size must be dividable by 2 at least 6 times "
    #     "to avoid fractions when downscaling and upscaling."
    #    "For example, use 256, 320, 384, 448, 512, ... etc. "
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    # anchor side in pixels, determined using inspect_crop_data.ipynb. can specify more or less scales
    RPN_ANCHOR_SCALES = (100, 150, 250, 375)  # for cp
    # RPN_ANCHOR_SCALES = (20, 60, 100, 140) # for smallholder

    # Aim to allow ROI sampling to pick 33% positive ROIs. This is always 33% in inspect_data nb, unsure if that is accurate.
    TRAIN_ROIS_PER_IMAGE = 300

    # Unsure what best step size is but nucleus used 100. Doubling because smallholder is more complex
    STEPS_PER_EPOCH = 400

    # reduces the max number of field instances
    # MAX_GT_INSTANCES = 29 # for smallholder determined using inspect_crop_data.ipynb
    MAX_GT_INSTANCES = 7  # for cp determined using inspect_crop_data.ipynb

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 100

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet50"

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Loss weights for more precise optimization. It has been suggested that mrcnn_mask_loss should be weighted higher
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.0,
        "rpn_bbox_loss": 1.0,
        "mrcnn_class_loss": 1.0,
        "mrcnn_bbox_loss": 1.0,
        "mrcnn_mask_loss": 3.0,
    }


class WV2InferenceConfig(WV2Config):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imagery for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7
