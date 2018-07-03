import os
import sys

import geopandas as gpd
import cv2
import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import shapely.affinity
import shapely.geometry
from shapely.geometry import MultiPolygon
import tensorflow as tf

# Import Mask RCNN
sys.path.append('../../Mask_RCNN')  # Get the mrcnn library onto the python path
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from mrcnn.config import Config


# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Directory to save logs and trained model
MODEL_DIR = '../../../weights/'
model_path = os.path.join(MODEL_DIR, "mask_rcnn_airstrips1.h5")


def unmold_detections(detections, mrcnn_mask, original_image_shape,
                      image_shape, window):
    """Reformats the detections of one image from the format of the neural
    network output to a format suitable for use in the rest of the
    application.

    detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
    mrcnn_mask: [N, height, width, num_classes]
    original_image_shape: [H, W, C] Original image shape before resizing
    image_shape: [H, W, C] Shape of the image after resizing and padding
    window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
            image is excluding the padding.

    Returns:
    boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
    class_ids: [N] Integer class IDs for each bounding box
    scores: [N] Float probability scores of the class_id
    masks: [height, width, num_instances] Instance masks
    """
    # How many detections do we have?
    # Detections array is padded with zeros. Find the first class_id == 0.
    zero_ix = np.where(detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

    # Extract boxes, class_ids, scores, and class-specific masks
    boxes = detections[:N, :4]
    class_ids = detections[:N, 4].astype(np.int32)
    scores = detections[:N, 5]
    masks = mrcnn_mask[np.arange(N), :, :, class_ids]

    # Translate normalized coordinates in the resized image to pixel
    # coordinates in the original image before resizing
    window = norm_boxes(window, image_shape[:2])
    wy1, wx1, wy2, wx2 = window
    shift = np.array([wy1, wx1, wy1, wx1])
    wh = wy2 - wy1  # window height
    ww = wx2 - wx1  # window width
    scale = np.array([wh, ww, wh, ww])
    # Convert boxes to normalized coordinates on the window
    boxes = np.divide(boxes - shift, scale)
    # Convert boxes to pixel coordinates on the original image
    boxes = denorm_boxes(boxes, original_image_shape[:2])

    # Filter out detections with zero area. Happens in early training when
    # network weights are still random
    exclude_ix = np.where(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        boxes = np.delete(boxes, exclude_ix, axis=0)
        class_ids = np.delete(class_ids, exclude_ix, axis=0)
        scores = np.delete(scores, exclude_ix, axis=0)
        masks = np.delete(masks, exclude_ix, axis=0)
        N = class_ids.shape[0]

    # Resize detections to original image size and set boundary threshold.
    mask_polys = []
    for i in range(N):
        # Convert polygon to full size mask
        mask_poly = unmold_mask_poly(masks[i], boxes[i])
        mask_polys.append(mask_poly)

    return boxes, class_ids, scores, mask_polys


def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


def unmold_mask_poly(mask, bbox):
    image, contours, hierarchy = cv2.findContours((mask > 0.5).astype('uint8'), cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)
    polys = [cont[:, 0, :] for cont in contours if cont.shape[0] > 2]
    polys = [resize_cont(p, (0, 0, mask.shape[1], mask.shape[0]), bbox) for p in polys]
    polys = [shapely.geometry.Polygon(p).buffer(0).simplify(0) for p in polys]

    if len(polys) == 1:
        return polys[0]
    else:
        return MultiPolygon(polys)


def resize_cont(cont, old_box, new_box):
    old_width = old_box[3] - old_box[1]
    old_height = old_box[2] - old_box[0]
    new_width = new_box[3] - new_box[1]
    new_height = new_box[2] - new_box[0]

    cont[:, 0] = (cont[:, 0] * new_width / old_width) + (new_box[1] - old_box[1])
    cont[:, 1] = (cont[:, 1] * new_height / old_height) + (new_box[0] - old_box[0])

    return cont


def resize_poly(poly, old_box, new_box):
    old_width = old_box[3] - old_box[1]
    old_height = old_box[2] - old_box[0]
    new_width = new_box[3] - new_box[1]
    new_height = new_box[2] - new_box[0]

    new_poly = shapely.affinity.scale(poly, new_width / old_width, new_height / old_height, origin=(0, 0))
    new_poly = shapely.affinity.translate(new_poly, new_box[1] - old_box[1], new_box[0] - old_box[0])
    return new_poly


class InferenceConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "airstrips"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", config=InferenceConfig(),
                          model_dir=MODEL_DIR)

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

keras_model = model.keras_model

input_size = 256
nchannels = 3
batch_size = 1

# These are sizes used by the function that decodes the network output.  I did not need them for this example
input_shape = (input_size, input_size, nchannels)  # shape of the network input
image_shape = (input_size, input_size, nchannels)  # shape of the image
window = [0, 0, input_size, input_size]  # portion of the image used for detection

# Set some constant inputs
meta = np.array([[0, 256, 256, 3, 256, 256, 3, 0, 0, 256, 256, 1, 0, 0]])

# Compute the anchors.  This is pretty complicated and probably not worth porting to C++
anchors = model.get_anchors((256, 256))
anchors = np.broadcast_to(anchors, (batch_size,) + anchors.shape)
# np.save('anchors256.npy', anchors)
# anchors = np.load('anchors256.npy')

#########################
## REPLACE THIS PART WITH CHIPPING
# Get a list of test images
image_dir = 'PATH TO IMAGES'
image_list = os.listdir(image_dir)

# Pick an image
image_file_path = os.path.join(image_dir, np.random.choice(image_list))
original_image = np.array(PIL.Image.open(image_file_path).resize((input_size, input_size)))
######################

# Normalize the image
molded_image = (original_image - np.array([123.7, 116.8, 103.9]))[None, :, :, :]  # 124, np.array([123.7, 116.8, 103.9])

# Run the model
detections, _, _, mrcnn_mask, _, _, _ = keras_model.predict([molded_image, meta, anchors], verbose=0)

# Decode the output
rois, class_ids, scores, polys = \
    unmold_detections(detections[0], mrcnn_mask[0], image_shape, input_shape, window)

# Filter out empty detections
if len(polys):
    rois, class_ids, scores, polys = zip(
        *[(b, i, s, p) for b, i, s, p in zip(rois, class_ids, scores, polys) if not p.is_empty])

detects = gpd.GeoDataFrame(list(zip(rois, class_ids, scores, polys)), columns=['roi', 'obj_class', 'conf', 'geometry'])
