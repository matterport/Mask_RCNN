import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from os.path import exists
import csv

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Main directory that contains project and data
MAIN_DIR = os.path.abspath("../../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append('../../mrcnn')  # ...Actually use local version
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

# Specify whether images are RGB or OCN.
IMAGE_TYPE = "OCN"

class StrawberryConfig(Config):
    """Configuration for training on the strawberry dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "strawberry"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    if IMAGE_TYPE == "RGB":
      NUM_CLASSES = 1 + 1  # Background + strawberry
    else:
      NUM_CLASSES = 1 + 3 # Background + strawberry + flower + note

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Validation steps per epoch
    # VALIDATION_STEPS = 50

    # Skip detections with < 80% confidence
    DETECTION_MIN_CONFIDENCE = 0.8


############################################################
#  Dataset
############################################################

class StrawberryDataset(utils.Dataset):

    def load_strawberry(self, dataset_dir, subset):
        """Load a subset of the Strawberry dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. For RGB, we have only one class to add.
        self.add_class("strawberry", 1, "strawberry")
        if IMAGE_TYPE == "OCN":
          # For OCN, also add the flower and note classes.
          self.add_class("flower", 2, "flower")
          self.add_class("note", 3, "note")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, "my_labels.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # Add images
        for annotation in annotations[0]:
            polygons = annotation['polygon']
            # Skip empty labels or annotations with very little data
            if len(polygons) < 20:
                continue
            width, height = 4000, 3000
            image_name = annotation['file'].split("/")[-1]
            image_path = os.path.join(dataset_dir, image_name)
            # Skip non-existing images
            file_exists = exists(image_path)
            if not file_exists:
                continue

            if IMAGE_TYPE == "RGB":
              self.add_image(
                  "strawberry",
                  image_id=image_name,  # use file name as a unique image id
                  path=image_path,
                  width=width, height=height,
                  polygons=polygons)
            else:
              labels = [label.lower() for label in annotation['label']]
              self.add_image(
                  "strawberry",
                  image_id=image_name,  # use file name as a unique image id
                  path=image_path,
                  width=width, height=height,
                  polygons=polygons,
                  annotations=labels)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a strawberry dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "strawberry":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, polygon in enumerate(info["polygons"]):
            # Avoid single-dimension polygons 
            # (i.e. [x, y] instead of [[x, y], [x, y], ...])
            try:
                len(polygon[0])
            except:
                continue
            # Get indexes of pixels inside the polygon and set them to 1
            x = [coord[0] for coord in polygon]
            y = [coord[1] for coord in polygon]
            rr, cc = skimage.draw.polygon(y, x)
            mask[rr, cc, i] = 1

        if IMAGE_TYPE == "RGB":
          # Return mask, and array of class IDs of each instance. Since we have
          # one class ID only, we return an array of 1s
          return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        else:
          # Return mask, and array of class IDs of each instance.
          class_ids = np.array([1 if a == 'strawberry' else (2 if a == 'flower' else 3) for a in info['annotations']]).astype(np.int32)
          return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "strawberry":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = StrawberryDataset()
    dataset_train.load_strawberry(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = StrawberryDataset()
    dataset_val.load_strawberry(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')

def crop(image, roi):
    """Crop an image to a region of interest."""
    x1, y1, x2, y2 = roi
    return image[x1:x2, y1:y2]


def segment(image, mask, roi):
    """Segment the image.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Transpose to (nSegments, height, width)
    mask_transpose = np.transpose(mask, [2, 0, 1])

    segments = []

    for mask_, roi_ in zip(mask_transpose, roi):
      # Crop image and mask
      image_ = crop(image, roi_)
      mask_ = crop(mask_, roi_)

      # Make a grayscale copy of the image. The grayscale copy still
      # has 4 RGBA channels, though.
      black = skimage.color.gray2rgba(skimage.color.rgb2gray(image_), alpha=0) * 255

      # Add alpha dim to image
      image_ = np.dstack((image_, np.full((image_.shape[0], image_.shape[1]), 255)))

      # Convert (width, height) to (width, height, 4)
      arr_new = np.ones((*mask_.shape, 4))
      for i, x in enumerate(mask_):
          for j, y in enumerate(x):
              arr_new[i][j] = [y for _ in range(4)]
      mask_ = arr_new

      # Use image values on mask, black otherwise
      res = np.where(mask_, image_, black).astype(np.uint8)
      segments.append(res)
    return segments


def calculate_iou(seg, seg_bbox, mask, mask_bbox_array, track_id_array):
  # Keep track of IoU and track id of best overlapping bbox
  highest_iou = 0
  best_track_id = -1
  
  for mask_bbox, track_id in zip(mask_bbox_array, track_id_array):
      # If bboxes dont overlap, continue
      if seg_bbox[2] < mask_bbox[0] or seg_bbox[0] > mask_bbox[2] or seg_bbox[3] < mask_bbox[1] or seg_bbox[1] > mask_bbox[3]:
        continue

      # Calculate intersection
      x_range = [max(seg_bbox[0], mask_bbox[0]), min(seg_bbox[2], mask_bbox[2])]
      y_range = [max(seg_bbox[1], mask_bbox[1]), min(seg_bbox[3], mask_bbox[3])]
      intersection = 0
      for x in range(x_range[0], x_range[1]):
        for y in range(y_range[0], y_range[1]):
          if mask[y, x] == seg[y, x]:
            intersection += 1

      if intersection == 0:
        print('no intersection')
        continue

      # Calculate union
      # Union = area of seg + area of mask - intersection
      seg_union = 0
      for x in range(seg_bbox[0], seg_bbox[2]):
        for y in range(seg_bbox[1], seg_bbox[3]):
          if mask[y, x] == seg[y, x]:
            seg_union += 1
      mask_union = 0
      for x in range(mask_bbox[0], mask_bbox[2]):
        for y in range(mask_bbox[1], mask_bbox[3]):
          if mask[y, x] == seg[y, x]:
            mask_union += 1
      union = seg_union + mask_union - intersection     

      if union == 0:
        print('no union')
        continue

      cur = intersection / union
      if cur > highest_iou:
        highest_iou = cur
        best_track_id = track_id
    
  return highest_iou, best_track_id


def detect_and_segment(model, image_paths, output_dir='output'):
    # Create output directory
    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, date)
    os.mkdir(path)

    # Create output csv
    # f = open('/Users/ceesjol/Documents/csv_output/' + date + '.csv', 'w')
    # f = open(os.path.join(output_dir, date, 'output.csv'), 'w')
    # writer = csv.writer(f)
    # writer.writerow(["Name", "Track ID", "Pixel width"])

    for image_path in image_paths:
      # Skip JSON, DS_Store, etc.
      if not (image_path.lower().endswith("jpg") or \
        image_path.lower().endswith("png")):
        continue

      #
      # 1. Run model detection
      #
      print('1. Running model on image...')
      print("Running on {}".format(image_path))
      # Read image
      image = skimage.io.imread(image_path)
      # Detect objects
      r = model.detect([image], verbose=1)[0]

      #
      # 2. Calculate accuracy
      #
      print('2. Calculating accuracy...')
      if not len(r['masks']):
        print('No strawberries were detected, exiting')
        return
      transposed_masks = np.transpose(r['masks'], [2, 0, 1])

      # Get image annotation
      if args.labels:
        test_annotations = json.load(open(args.labels))
      else:
        test_annotations = json.load(open(os.path.join(MAIN_DIR, "data/Images/RGBCAM1_split/test/my_labels.json")))
      test_annotations = list(test_annotations.values())  # don't need the dict keys
      image_name = image_path.split("/")[-1]
      ann = [x for x in test_annotations[0] if image_name in x['file']]
      if len(ann) == 0:
        raise Exception('No annotation found for image', image_name)
      ann = ann[0]

      # Convert polygons to a bitmap mask of shape
      # [height, width, instance_count]
      polygons = ann['polygon']
      mask = np.zeros([3000, 4000], dtype=np.int32)
      for polygon in polygons:
          # Avoid single-dimension polygons 
          # (i.e. [x, y] instead of [[x, y], [x, y], ...])
          try:
              len(polygon[0])
          except:
              continue
          # Get indexes of pixels inside the polygon and set them to 1
          x = [coord[0] for coord in polygon]
          y = [coord[1] for coord in polygon]
          rr, cc = skimage.draw.polygon(y, x)
          mask[rr, cc] = 1
      mask_bbox_array = ann['bbox']
      track_id_array = ann['track_id']

      iou_sum = 0
      track_id_estimations = []
      for seg, seg_bbox in zip(transposed_masks, r['rois']):
          # flip x and y
          seg_bbox = [seg_bbox[1], seg_bbox[0], seg_bbox[3], seg_bbox[2]]

          # Crop predicted bbox
          # Commented out since it doesn't seem to improve IoU
          if False:
            x, y = np.nonzero(seg)
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            seg_bbox = [y_min, x_min, y_max, x_max]

          iou, track_id = calculate_iou(seg, seg_bbox, mask, mask_bbox_array, track_id_array)
          iou_sum += iou
          track_id_estimations.append(track_id)

      print('average IoU', iou_sum / len(transposed_masks))

      #
      # 3. Generate segmentation images and crop them.
      #
      print('3. Creating segmented images...')
      segments = segment(image, r['masks'], r['rois'])
      for i, s in enumerate(segments):
        x, y = np.nonzero(s[:,:,-1])
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        segments[i] = s[x_min:x_max + 1, y_min:y_max + 1, 0:4]

      #
      # 4. Save output.
      #
      print('4. Outputting results...')

      # Output each segment
      for idx, (seg, track_id) in enumerate(zip(segments, track_id_estimations)):
        # Image
        image_name = image_path.split('/')[-1].split('.')[0]
        seg_name = '_seg' + str(idx) + '_' + str(track_id)
        file_name = image_name + seg_name + '.png'
        skimage.io.imsave(os.path.join(output_dir, date, file_name), seg)

        # csv
        # writer.writerow([image_name, track_id, seg.shape[1]])

      # f.close()
    print('Done')


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect strawberries.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'segment'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/strawberry/dataset/",
                        help='Directory of the Strawberry dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the segmentation on')
    parser.add_argument('--folder', required=False,
                        metavar="path or URL to folder",
                        help='Folder to apply the segmentation on')                    
    parser.add_argument('--labels', required=False,
                        metavar="path or URL to labels JSON file",
                        help='Labels')
    parser.add_argument('--output', required=False,
                        metavar="Segmentation images output folder",
                        help='Output')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "segment":
        assert args.image or args.folder,\
               "Provide --image or --folder to apply segmentation"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = StrawberryConfig()
    else:
        class InferenceConfig(StrawberryConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "segment":
        if args.image:
          image_paths = [args.image]
        else:
          image_paths = os.listdir(args.folder)
          image_paths = [os.path.join(args.folder, image_path) for image_path in image_paths]
        print('image_paths: ', image_paths)
        detect_and_segment(model, image_paths, args.output)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'segment'".format(args.command))
