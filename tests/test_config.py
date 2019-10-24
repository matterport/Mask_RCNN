from mrcnn.config import Config
import numpy as np


def test_config_to_dict():
    config_dict = Config().to_dict()
    assert config_dict["BATCH_SIZE"] == 2
    assert config_dict["LOSS_WEIGHTS"] == \
           {'rpn_class_loss': 1.0,
            'rpn_bbox_loss': 1.0,
            'mrcnn_class_loss': 1.0,
            'mrcnn_bbox_loss': 1.0,
            'mrcnn_mask_loss': 1.0}
    assert config_dict["MEAN_PIXEL"].all() == np.array([123.7, 116.8, 103.9]).all()


def test_config_display(capsys):
    config = Config()
    config.display()
    captured = capsys.readouterr()
    assert "LEARNING_MOMENTUM              0.9" in captured.out
