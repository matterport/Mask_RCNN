from cropmask.mrcnn import utils
import os
import pandas as pd
import skimage.io as skio
import numpy as np
import yaml

class ImageDataset(utils.Dataset):
    """Generates the Imagery dataset used by mrcnn."""
       
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,N] Numpy array.
        Channels are ordered [B, G, R, ...]. This is called by the 
        Keras data_generator function
        """
        # Load image
        image = skio.imread(self.image_info[image_id]["path"])

        assert image.ndim == 3

        return image

    def load_imagery(self, dataset_dir, subset, image_source, class_name, train_test_split_dir):
        """Load a subset of the fields dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load.
                * train: training images/masks excluding testing
                * test: testing images moved by train/test split func
        image_source: string identifier for imagery. "wv2" or "planet" or "landsat"
        class_name: string name for class. "agriculture" or another name 
                depending on labels. use self.add_class for a multi class model.
        train_test_split_dir: the directory to hold the train_ids and test_ids from PreprocessWorflow
        """
        # Add classes here
        self.add_class(image_source, 1, class_name)
        assert subset in ["train", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)
        train_ids = pd.read_csv(os.path.join(train_test_split_dir, "train_ids.csv"))
        train_list = list(train_ids["train"])
        test_ids = pd.read_csv(os.path.join(train_test_split_dir, "test_ids.csv"))
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
        tile_folder_path = '/'.join(self.image_info[image_id]['path'].split('/')[:-2])
        mask_name = os.listdir(os.path.join(tile_folder_path,'mask'))[0]
        m = skio.imread(os.path.join(tile_folder_path, 'mask',mask_name)).astype(np.bool)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        if len(m.shape) < 3:
            m = np.expand_dims(m,2) # this conditional has had to be placed throughout to deal with images without labels. Need a better, less fragile way, could do this in the preprocess step.
        return m, np.ones([m.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "field":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)
