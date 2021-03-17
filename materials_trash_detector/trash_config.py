# -*- coding: utf-8 -*-


from mrcnn.config import Config

############################################################
#  Configurations
############################################################

class TrashConfig(Config):
    """Configuration for training on MS trash.
    Derives from the base Config class and overrides values specific
    to the trash dataset.
    """
    # Give the configuration a recognizable name
    NAME = "trash"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 7  # trash has 6 classes

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    # LOSS_WEIGHTS = {
    #     "rpn_class_loss": 1.,
    #     "rpn_bbox_loss": 1.,
    #     "mrcnn_class_loss": 1.,
    #     "mrcnn_bbox_loss": 1.,
    #     "mrcnn_mask_loss": 1.
    # }

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet101"