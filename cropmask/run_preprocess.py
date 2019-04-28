import os
import random
import yaml

random.seed(1)

if __name__ == "__main__":
    make_dirs()
    reorder_images(params)
    negative_buffer_and_small_filter(params)
    grid_images(params)
    open_labels(params)
    move_img_to_folder(params)
    connected_comp(params)
    train_test_split(params)
    print("preprocessing complete, ready to run model.")

    print("channel means, put these in model_configs.py subclass")
    band_indices = yaml_to_band_index(params)
    for i, v in enumerate(band_indices):
        get_arr_channel_mean(i)
def parse_yaml(input_file):
    """Parse yaml file of configuration parameters."""
    with open(input_file, "r") as yaml_file:
        params = yaml.safe_load(yaml_file)
    return params


params = parse_yaml("preprocess_config.yaml")

ROOT = params["dirs"]["root"]

REGION = os.path.join(ROOT, params["dirs"]["region_name"])

DATASET = os.path.join(REGION, params["dirs"]["dataset"])

STACKED = os.path.join(DATASET, params["dirs"]["stacked"])

TRAIN = os.path.join(DATASET, params["dirs"]["train"])

TEST = os.path.join(DATASET, params["dirs"]["test"])

GRIDDED_IMGS = os.path.join(DATASET, params["dirs"]["gridded_imgs"])

GRIDDED_LABELS = os.path.join(DATASET, params["dirs"]["gridded_labels"])

OPENED = os.path.join(DATASET, params["dirs"]["opened"])

NEG_BUFFERED = os.path.join(DATASET, params["dirs"]["neg_buffered_labels"])

RESULTS = os.path.join(
    ROOT, params["dirs"]["results"], params["dirs"]["dataset"]
)

SOURCE_IMGS = os.path.join(ROOT, params["dirs"]["region_name"])

SOURCE_LABELS = os.path.join(ROOT, params["dirs"]["region_labels"])

DIRECTORY_LIST = [
        REGION,
        DATASET,
        STACKED,
        TRAIN,
        TEST,
        GRIDDED_IMGS,
        GRIDDED_LABELS,
        OPENED,
        NEG_BUFFERED,
        RESULTS,
    ]