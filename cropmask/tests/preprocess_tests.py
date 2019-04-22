import cropmask.preprocess as pp

def test_make_dirs():
    
    ROOT = "/az-ml-container"

    DATASET = os.path.join(ROOT, "pytest_dataset")

    STACKED = os.path.join(DATASET, "stacked")

    TRAIN = os.path.join(DATASET, "train")

    TEST = os.path.join(DATASET, "test")

    GRIDDED_IMGS = os.path.join(DATASET, "gridded_imgs")

    GRIDDED_LABELS = os.path.join(DATASET, "gridded_labels")

    OPENED = os.path.join(DATASET, "opened")

    NEG_BUFFERED = os.path.join(DATASET, "neg_buffered_labels")

    RESULTS = os.path.join(ROOT, "pytest_results", DATASET)

    SOURCE_IMGS = os.path.join(ROOT, "pytest_source_imgs")

    SOURCE_LABELS = os.path.join(ROOT, "pytest_source_labels")

    directory_list = [
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
    pp.make_dirs(directory_list)
    
    for i in directory_list:
        assert os.exists(i)
    
    
    
