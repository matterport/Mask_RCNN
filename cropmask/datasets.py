from mrcnn import utils
import os
import pandas as pd
import skimage.io as skio
import numpy as np
import preprocess as pp

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# Results directory
# Save submission files and test/train split csvs here
RESULTS_DIR = pp.RESULTS

############################################################
#  Dataset
############################################################


class ImageDataset(utils.Dataset):
    """Generates the Imagery dataset."""

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,N] Numpy array.
        Channels are ordered [B, G, R, ...]. This is called by the 
        Keras data_generator function
        """
        # Load image
        image = skio.imread(self.image_info[image_id]["path"])

        assert image.ndim == 3

        return image

    def load_imagery(self, dataset_dir, subset, image_source, class_name):
        """Load a subset of the fields dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load.
                * train: training images/masks excluding testing
                * test: testing images moved by train/test split func
        image_source: string identifier for imagery. "wv2" or "planet"
        class_name: string name for class. "agriculture" or another name 
                depending on labels. self.add_class for multi class model.
        """
        # Add classes. We have one class.
        self.add_class(image_source, 1, class_name)
        assert subset in ["train", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)
        train_ids = pd.read_csv(os.path.join(RESULTS_DIR, "train_ids.csv"))
        train_list = list(train_ids["train"])
        test_ids = pd.read_csv(os.path.join(RESULTS_DIR, "test_ids.csv"))
        test_list = list(test_ids["test"])
        if subset == "test":
            image_ids = test_list
        else:
            image_ids = train_list

        # Add images
        for image_id in image_ids:
            self.add_image(
                image_source,
                image_id=image_id,
                path=os.path.join(
                    dataset_dir, str(image_id), "image/{}.tif".format(str(image_id))
                ),
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info["path"])), "mask")

        # Read mask files from image
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".tif"):
                m = skio.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)
                assert m.ndim == 2
        mask = np.stack(mask, axis=-1)
        # assert mask.ndim == 3
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "field":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)
