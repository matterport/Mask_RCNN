

from mrcnn.config import Config
import numpy as np



# here you can add more object shapes 
# globalObjectCategories= ['BG', 'cucumber', 'flower', 'leaf']
globalObjectCategories= ['BG', 'stem']

# objectsDistribution = ["cucumber","cucumber","cucumber", "leaf","leaf","leaf","leaf", "flower", "flower"]
objectsDistribution = ['stem']

class cucumberConfig(Config):
    """Configuration for training on dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    

    """FREQUENTLY TUNED HYPER PARAMETERS"""


    """ ARCHITECTURE HYPER-PARAMETERS"""
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = len(globalObjectCategories) # background + cucumber, leaf, flower, stems

    # anchor side in pixels, for each of RPN layer
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  
    
    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    
    # Maximum number of ground truth instances to use in one image
    # Pay attention to these when you realize you have "too much" annotations on one image
    MAX_GT_INSTANCES = 300
    
    # Max number of final detections
    DETECTION_MAX_INSTANCES = 300


    '''ACTIVE TRAINING HYPER PARAMETERS'''

    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9


    # each EPOCHS times we save the weights of the net
    EPOCHS = 10

    # STEPS_PER_EPOCH is set in __init__ procedure


    VALIDATION_STEPS = 4
    
    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.5

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.5
    
    # Skip detections with lower than DETECTION_MIN_CONFIDENCE percents confidence
    DETECTION_MIN_CONFIDENCE = 0.7

    """ DATA GENERATION HYPER PARAMETERS """
    # make sure namings inside are compatible with globalObjectCategories
    RANDOM_DISTRIBUTION = objectsDistribution

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    #THIS PARTICULAR HYPERPARAMETER IS BOTH FOR SETTING DIMS FOR NET AND FOR DATA GENERATION AS WELL!
    SQUARED_IMAGES_DIM_FOR_CURRENT_SESSION = 1024

    IMAGE_MIN_DIM = SQUARED_IMAGES_DIM_FOR_CURRENT_SESSION
    IMAGE_MAX_DIM = SQUARED_IMAGES_DIM_FOR_CURRENT_SESSION
    
    #SCALE BOUNDRIES OF GENERATED OBJECTS
    MIN_SCALE_OBJ = 0.1
    MAX_SCALE_OBJ = 0.3
    # MIN_SCALE_OBJ = 0.5
    # MAX_SCALE_OBJ = 0.8

    # this hyper parameter varifies that object is not generated outside boundries of image being generated
    BOUNDING_DELTA = 0.2

    GEN_TRAIN_SET_SIZE = 2000
    REAL_TRAIN_SET_SIZE = 0
    # VALID_SET_SIZE = 5
    TEST_SET_SIZE = 12

    #Boundries for num of objects in image in case images are synthesized,
    MIN_GENERATED_OBJECTS = 2
    MAX_GENERATED_OBJECTS = 10

    # this threshold determines how much objects will cover each other
    OBJECTS_IOU_THRESHOLD = 0.05



    """SELDOM TUNED HYPER PARAMETERS"""

    # Give the configuration a recognizable name
    NAME = "agriculture"

    # Train on 1 GPU and X images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is  (GPUs * (imagesPerGPU)).
    GPU_COUNT = 1


    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False

    # EPOCHS_ROUNDS is the number of times we generate new training data with different stats.
    EPOCHS_ROUNDS = 1

    # control disk space on computer, non relevant when EPOCH_ROUNDS is 1
    MAX_SAVED_TRAINED_MODELS = 30

    #in case we want to generate new dataset each iteratation in EPOCH_ROUNDS
    SCALE_OBJECT_NUM_NEXT_EPOCH_ROUND = 1
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
        self.STEPS_PER_EPOCH  = (self.GEN_TRAIN_SET_SIZE + self.REAL_TRAIN_SET_SIZE)  // self.IMAGES_PER_GPU


class InferenceConfig(cucumberConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
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
        self.STEPS_PER_EPOCH  = self.TEST_SET_SIZE // self.IMAGES_PER_GPU


cucuConfForTrainingSession = cucumberConfig()
