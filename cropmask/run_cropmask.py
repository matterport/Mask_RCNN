"""
Original nucleus.py example written by Waleed Abdulla at 
https://github.com/matterport/Mask_RCNN/blob/master/samples/nucleus/nucleus.py
------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 crop_mask.py train --dataset=data/wv2 --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 crop_mask.py train --dataset=data/wv2 --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 crop_mask.py train --dataset=data/wv2 --subset=train --weights=last

    # Generate submission file
    python3 crop_mask.py detect --dataset=data/wv2 --subset=train --weights=<last or /path/to/weights.h5>
"""


############################################################
#  Pre-processing and train/test split
############################################################
# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == "__main__":
    import matplotlib

    # Agg backend runs without a display
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

############################################################
#  Set model paths and imports
############################################################
import sys
import os
import datetime
from imgaug import augmenters as iaa

# Import cropmask and mrcnn
from cropmask.preprocess import PreprocessWorkflow
from cropmask import datasets, model_configs
from cropmask.mrcnn import model as modellib
from cropmask.mrcnn import visualize
import numpy as np

# Path to trained weights file
ROOT_DIR = "/home/ryan/work/CropMask_RCNN"
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

#contains paths as instance attributes
wflow = PreprocessWorkflow(os.path.join(ROOT_DIR,"cropmask/preprocess_config.yaml"))


############################################################
#  Training
############################################################


def train(model, dataset_dir, subset, config):
    """Train the model."""
    # Training dataset.
    dataset_train = datasets.ImageDataset()
    dataset_train.load_imagery(
        dataset_dir, "train", image_source="landsat", class_name="agriculture"
    )
    dataset_train.prepare()

    # Validation dataset
    dataset_val = datasets.ImageDataset()
    dataset_val.load_imagery(
        dataset_dir, "test", image_source="landsat", class_name="agriculture"
    )
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf(
        (0, 2),
        [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf(
                [iaa.Affine(rotate=90), iaa.Affine(rotate=180), iaa.Affine(rotate=270)]
            ),
        ],
    )

    # *** This training schedule is an example. Update to your needs ***

    print("Train all layers")
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=100,
        augmentation=augmentation,
        layers="all",
    )


############################################################
#  RLE Encoding
############################################################


def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################


def detect(model, dataset_dir, subset, wflow):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(wflow.RESULTS):
        os.makedirs(wflow.RESULTS)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(wflow.RESULTS, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = datasets.ImageDataset(3)
    dataset.load_imagery(
        dataset_dir, subset, image_source="landsat", class_name="agriculture", train_test_split_dir=wflow.RESULTS
    )
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        # commented out lines to only return output of detection, no viz or saves for api calls
#         visualize.display_instances(
#             image,
#             r["rois"],
#             r["masks"],
#             r["class_ids"],
#             dataset.class_names,
#             r["scores"],
#             show_bbox=False,
#             show_mask=False,
#             title="Predictions",
#         )
#         plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
#     file_path = os.path.join(submit_dir, "submit.csv")
#     with open(file_path, "w") as f:
#         f.write(submission)
#     print("Saved to ", submit_dir)
    return submission


############################################################
#  Command Line
############################################################

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Mask R-CNN for fields counting and segmentation"
    )
    parser.add_argument(
        "command",
        metavar="<command>",
        help="'preprocess' or 'train' or 'detect. preprocess takes no arguments.'",
    )
    parser.add_argument(
        "--dataset",
        required=False,
        metavar="/path/to/dataset/",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--weights",
        required=False,
        metavar="/path/to/weights.h5",
        help="Path to weights .h5 file or 'coco'",
    )
    parser.add_argument(
        "--logs",
        required=False,
        default=DEFAULT_LOGS_DIR,
        metavar="/path/to/logs/",
        help="Logs and checkpoints directory (default=logs/)",
    )
    parser.add_argument(
        "--subset",
        required=False,
        metavar="Dataset sub-directory",
        help="Subset of dataset to run prediction on",
    )
    args = parser.parse_args()

    if args.command == "preprocess":
        wflow = PreprocessWorkflow()
        wflow.run_single_scene()
    else:
        # Validate arguments
        if args.command == "train":
            assert args.dataset, "Argument --dataset is required for training"
        elif args.command == "detect":
            assert args.subset, "Provide --subset to run prediction on"

        print("Weights: ", args.weights)
        print("Dataset: ", args.dataset)
        if args.subset:
            print("Subset: ", args.subset)
        print("Logs: ", args.logs)

        # Configurations
        if args.command == "train":
            config = model_configs.LandsatConfig(3)
        else:
            config = model_configs.LandsatInferenceConfig(3)
        config.display()

        # Create model
        if args.command == "train":
            model = modellib.MaskRCNN(
                mode="training", config=config, model_dir=args.logs
            )
        else:
            model = modellib.MaskRCNN(
                mode="inference", config=config, model_dir=args.logs
            )
        if args.weights is not None:
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
                model.load_weights(
                    weights_path,
                    by_name=True,
                    exclude=[
                        "mrcnn_class_logits",
                        "mrcnn_bbox_fc",
                        "mrcnn_bbox",
                        "mrcnn_mask",
                    ],
                )
            else:
                model.load_weights(weights_path, by_name=True)

        # Train or evaluate
        if args.command == "train":
            os.chdir(ROOT_DIR)
            print(os.getcwd(), "current working dir")
            train(model, args.dataset, args.subset, config)
        elif args.command == "detect":
            detect(model, args.dataset, args.subset, wflow)
        else:
            print(
                "'{}' is not recognized. "
                "Use 'train' or 'detect'".format(args.command)
            )
