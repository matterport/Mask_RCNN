

from mrcnn.config import Config
import numpy as np
class cucumberConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """

    """MODEL HYPER PARAMETERS"""
    # Give the configuration a recognizable name
    NAME = "cucumbers"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 16
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 3 # background + cucumber, leaf, flower

    # anchor side in pixels, for each of RPN layer
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  
    
    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32
    
    #asher todo: can we utilize it better?
    # ROI_POSITIVE_RATIO = 66  
    

    VALIDATION_STEPS = 100
     # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    MAX_SAVED_TRAINED_MODELS = 10
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.5
    # each EPOCHS times we save the weights of the net
    EPOCHS = 12
    # EPOCHS_ROUNDS determines how many weighst of the net we will save
    EPOCHS_ROUNDS = 1


    """ DATA GENERATION HYPER PARAMETERS """
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    #THIS PARTICULAR HYPERPARAMETER IS BOTH FOR SETTING DIMS FOR NET AND FOR DATA GENERATION AS WELL!
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    
    #SCALES OF GENERATED OBJECTS
    MIN_SCALE_OBJ = 0.3
    MAX_SCALE_OBJ = 0.5
    # this hyper parameter varifies that object is not generated outside boundries of image being generated
    BOUNDING_DELTA = 0.2

    TRAIN_SET_SIZE = 10000
    VALID_SET_SIZE = 1000

    #in case images are synthesized
    MIN_GENERATED_OBJECTS = 5
    MAX_GENERATED_OBJECTS = 25

    #in case we want to generate new dataset each iteratation in EPOCH_ROUNDS
    SCALE_OBJECT_NUM_NEXT_EPOCH_ROUND = 1

    # this threshold determines how much objects will cover each other
    OBJECTS_IOU_THRESHOLD = 0.05

    #this is an optional scaler when starting new epochs run
    SCALE_OBJECTS_IOU_THRESHOLD = 1

    def __init__(self):
        super().__init__()
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM,
                self.IMAGE_CHANNEL_COUNT])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,
                self.IMAGE_CHANNEL_COUNT])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES
        self.STEPS_PER_EPOCH  = self.TRAIN_SET_SIZE // self.IMAGES_PER_GPU



cucuConf = cucumberConfig()