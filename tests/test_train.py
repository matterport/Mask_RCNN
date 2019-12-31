import mrcnn.model as modellib
import numpy as np

from samples.shapes.shapes import ShapesConfig
from samples.shapes.shapes import ShapesDataset
from fixtures import model_data


class UnittestTrainConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def test_training(tmpdir, model_data):
    config = UnittestTrainConfig()
    # Training dataset
    dataset_train = ShapesDataset()
    dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ShapesDataset()
    dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_val.prepare()
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=tmpdir)
    model.load_weights(model_data, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=2,
                layers="all")

    model = modellib.MaskRCNN(mode="inference",
                              config=config,
                              model_dir=tmpdir)
    model_path = model.find_last()
    # Load trained weights
    model.load_weights(model_path, by_name=True)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, config,
                               0, use_mini_mask=False)

    results = model.detect([original_image], verbose=0)
    r = results[0]

    assert gt_class_id == r['class_ids']


class FixShapesDataset(ShapesDataset):

    def random_image(self, height, width):
        """ Return fixed image for testing
        """
        # Pick random background color
        bg_color = np.array([202, 3, 25])
        shapes = [('square', (28, 39, 158), (99, 65, 21)), ('circle', (158, 237, 77), (57, 26, 29))]
        return bg_color, shapes


def test_load_image_gt(tmpdir, model_data):
    config = ShapesConfig()
    dataset_train = FixShapesDataset()
    dataset_train.load_shapes(5, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_train.prepare()
    image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(dataset_train, config, 0)
    assert (image[0][0] == [202, 3, 25]).all()
    assert (image_meta == [0, 128, 128, 3, 128, 128, 3, 0, 0, 128, 128, 1, 1, 1, 1, 1]).all()
    assert (class_ids == [1, 2]).all()
    assert (bbox == [[44, 78, 87, 121], [0, 28, 56, 87]]).all()
    assert (mask[0][0] == [0, 0]).all()


def test_data_sequence():
    config = ShapesConfig()
    dataset = FixShapesDataset()
    dataset.load_shapes(1, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset.prepare()
    dg = modellib.data_generator(dataset, config)
    inputs, output = next(dg)

    assert output == []
    batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox, batch_gt_class_ids, batch_gt_boxes, batch_gt_masks = inputs

    assert (batch_gt_boxes[0][:3] == [[44, 78, 87, 121], [0, 28, 56, 87], [0, 0, 0, 0]]).all()
    assert (batch_image_meta[0] == [0, 128, 128, 3, 128, 128, 3, 0, 0, 128, 128, 1, 1, 1, 1, 1]).all()


