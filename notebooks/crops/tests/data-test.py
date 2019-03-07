def parse_yaml(input_file):
    """Parse yaml file of configuration parameters."""
    with open(input_file, "r") as yaml_file:
        params = yaml.load(yaml_file)
    return params


params = parse_yaml("preprocess_config.yaml")

ROOT = params["dirs"]["root"]

DATASET = os.path.join(ROOT, params["dirs"]["dataset"])

REORDER = os.path.join(DATASET, params["dirs"]["reorder"])

TRAIN = os.path.join(DATASET, params["dirs"]["train"])

TEST = os.path.join(DATASET, params["dirs"]["test"])

GRIDDED_IMGS = os.path.join(DATASET, params["dirs"]["gridded_imgs"])

GRIDDED_LABELS = os.path.join(DATASET, params["dirs"]["gridded_labels"])

OPENED = os.path.join(DATASET, params["dirs"]["opened"])

INSTANCES = os.path.join(DATASET, params["dirs"]["instances"])

RESULTS = os.path.join(
    ROOT, "../", params["dirs"]["results"], params["dirs"]["dataset"]
)

SOURCE_IMGS = os.path.join(ROOT, params["dirs"]["source_imgs"])

SOURCE_LABELS = os.path.join(ROOT, params["dirs"]["source_labels"])

# all files, including ones we don't care about
file_ids_all = next(os.walk(SOURCE_IMGS))[2]
# all multispectral on and off season tifs
image_ids_all = [
    image_id for image_id in file_ids_all if "MS" in image_id and ".aux" not in image_id
]

# check for duplicates
assert len(image_ids_all) == len(set(image_ids_all))

image_ids_gs = [image_id for image_id in image_ids_all if "GS" in image_id]
image_ids_os = [image_id for image_id in image_ids_all if "OS" in image_id]

# check for equality
assert len(image_ids_os) == len(image_ids_gs)

# only select growing season images
image_ids_short = [image_id[0:9] for image_id in image_ids_gs]

for imid in image_ids_short:
    load_merge_wv2(imid, WV2_DIR)

image_list = next(os.walk(REORDERED_DIR))[2]
