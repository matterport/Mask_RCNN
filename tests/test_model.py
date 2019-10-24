import mrcnn.model as modellib
import numpy as np
import skimage.io

from fixtures import model_data
from fixtures import ROOT_DIR
from mrcnn.config import Config


TEST_IMAGE_PATH = ROOT_DIR/'images/3627527276_6fe8cd9bfe_z.jpg'


class UnittestConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "unittest"
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes
    GPU_COUNT = 1


def test_inference_detect(tmpdir, model_data):
    config = UnittestConfig()
    model = modellib.MaskRCNN(mode="inference", model_dir=tmpdir, config=config)
    # Load weights trained on MS-COCO
    model.load_weights(model_data, by_name=True)
    image = skimage.io.imread(TEST_IMAGE_PATH)
    result = model.detect([image], verbose=1)[0]
    assert np.all([result['class_ids'], [24, 23, 23, 23]])
    assert np.all([np.greater(result['scores'], [0.99, 0.99, 0.99, 0.99]), [True, True, True, True]])
