import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

#custom package for loading configs, model class and detection func
from cropmask import model_configs
from cropmask.mrcnn import model as modellib
import tensorflow as tf
import skimage.io as skio
import rasterio as rio
from rasterio.plot import reshape_as_image

# Core detection functions

def open_image(image_bytes):
    """ Open an image in binary format using PIL.Image and convert to RGB mode
    Args:
        image_bytes: an image in binary format read from the POST request's body

    Returns:
        an PIL image object in RGB mode
    """
    # image = Image.open(image_bytes)
    # if image.mode not in ('RGBA', 'RGB'):
    #     raise AttributeError('Input image not in RGBA or RGB mode and cannot be processed.')
    # if image.mode == 'RGBA':
    #     # Image.convert() returns a converted copy of this image
    #     image = image.convert(mode='RGB')
    # return image

    img = rio.open(image_bytes)
    arr = reshape_as_image(img.read())
    # returns the array for detection and the PIL img object for drawing since model trianed on flaot 32
    # and PIL can't read tiff (float32) and png and jpeg don't support float 32
    return arr, Image.fromarray(np.unint16(arr), mode='RGB')



def generate_detections(arr):
    """ Generates a set of bounding boxes with confidence and class prediction for one input image file.

    Args:
        detection_model: a mrcnn model class object containing already loaded segmentation weights
        image_file: a PIL Image object

    Returns:
        boxes, scores, classes, and the image loaded from the input image_file - for one image
    """
        # Load the model
    # The model was copied to this location when the container was built; see ../Dockerfile
    config = model_configs.LandsatInferenceConfig(3)
    print('keras_detector.py: Loading model weights...')
    LOGS_DIR = "/app/keras_iNat_api/logs"
    with tf.device("/cpu:0"):
      model = modellib.MaskRCNN(mode="inference",
                                model_dir=LOGS_DIR,
                                config=config)
    model_path = '/app/keras_iNat_api/mask_rcnn_landsat-512-cp_0042.h5'
    model.load_weights(model_path, by_name=True)
    print('keras_detector.py: Model weights loaded.')

    result = model.detect([arr], verbose=1)
    r = result[0]
    return r['rois'], r['class_ids'], r['scores'], r['masks']# these are lists of bboxes, scores etc

# Rendering functions
def render_bounding_boxes(boxes, scores, classes, image, label_map={}, confidence_threshold=0.5):
    """Renders bounding boxes, label and confidence on an image if confidence is above the threshold.

    Args:
        boxes, scores, classes:  outputs of generate_detections.
        image: PIL.Image object, output of generate_detections.
        label_map: optional, mapping the numerical label to a string name.
        confidence_threshold: threshold above which the bounding box is rendered.

    image is modified in place!

    """
    display_boxes = []
    display_strs = []  # list of list, one list of strings for each bounding box (to accommodate multiple labels)

    for box, score, clss in zip(boxes, scores, classes):
        if score > confidence_threshold:
            print('Confidence of detection greater than threshold: ', score)
            display_boxes.append(box)
            clss = int(clss)
            label = label_map[clss] if clss in label_map else str(clss)
            displayed_label = '{}: {}%'.format(label, round(100*score))
            display_strs.append([displayed_label])

    display_boxes = np.array(display_boxes)
    draw_bounding_boxes_on_image(image, display_boxes, display_str_list_list=display_strs)

# the following two functions are from https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py

def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 color='LimeGreen',
                                 thickness=4,
                                 display_str_list_list=()):
  """Draws bounding boxes on image.

  Args:
    image: a PIL.Image object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
  boxes_shape = boxes.shape
  if not boxes_shape:
    return
  if len(boxes_shape) != 2 or boxes_shape[1] != 4:
    raise ValueError('Input must be of size [N, 4]')
  for i in range(boxes_shape[0]):
    display_str_list = ()
    if display_str_list_list:
      display_str_list = display_str_list_list[i]
    draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                               boxes[i, 3], color, thickness, display_str_list)


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin
