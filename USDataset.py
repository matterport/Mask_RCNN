import random

import cv2
import numpy as np
import skimage
import skimage.io

import utils

images_path = 'I:\\Mask_RCNN\\data\\images\\'
masks_path = 'I:\\Mask_RCNN\\data\\masks\\'


class USDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def __init__(self, path):
        super(USDataset, self).__init__()
        print('file list %s' % path)

        self.path = path
        self.add_class("", 1, 'mass')

        with open(path, 'r') as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                line = line.strip()
                self.add_image(source='', path=images_path + line, image_id=count)
                count = count + 1
        print('load image: %d' % len(self.image_info))

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        return info['source']

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        path = self.image_info[image_id]['path']
        path = path.split('\\')
        mask_path = masks_path + path[-1]
        mask = skimage.io.imread(mask_path)
        # mask = mask[:, :, 1]
        mask = np.array([mask], dtype=np.uint8)
        mask = np.transpose(mask, [1, 2, 0])
        class_ids = np.array([1], dtype=np.int32)
        return mask, class_ids
