# -*- coding: utf-8 -*-
import os.path
import pytest
import urllib

from pathlib import Path


ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
CACHE_DIR = ROOT_DIR/"cache"


@pytest.fixture
def model_data():
    """ Fixture for downloading mask_rcnn_coco training data
    """
    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    test_model_path = str((CACHE_DIR / "mask_rcnn_coco.h5").resolve())
    if not os.path.isfile(test_model_path):
        urllib.request.urlretrieve(
            "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5",
            test_model_path)
    return test_model_path
