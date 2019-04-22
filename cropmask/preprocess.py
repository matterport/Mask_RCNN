import random
import os
import shutil
import copy
from skimage import measure
from skimage import morphology as skim
import skimage.io as skio
import warnings
import pandas as pd
import numpy as np
import pathlib
import yaml
import geopandas as gpd
from rasterio import features, coords
import rasterio
from shapely.geometry import shape
import gdal

random.seed(1)


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

NEG_BUFFERED = os.path.join(DATASET, params["dirs"]["neg_buffered_labels"])

RESULTS = os.path.join(
    ROOT, "../", params["dirs"]["results"], params["dirs"]["dataset"]
)

SOURCE_IMGS = os.path.join(ROOT, params["dirs"]["source_imgs"])

SOURCE_LABELS = os.path.join(ROOT, params["dirs"]["source_labels"])


def make_dirs():

    dirs = [
        DATASET,
        REORDER,
        TRAIN,
        TEST,
        GRIDDED_IMGS,
        GRIDDED_LABELS,
        OPENED,
        NEG_BUFFERED,
        RESULTS,
    ]

    # Make directory and subdirectories
    for d in dirs:
        pathlib.Path(d).mkdir(parents=False, exist_ok=False)

    # Change working directory to project directory
    os.chdir(dirs[0])


def yaml_to_band_index(params):
    """Parses config booleans to a list of band indexes to be stacked.

    Landsat has 6 bands (7 if you count band 6, thermal) that we can use
    for masking.

    Args:
        params (dict): The configuration dictionary that is read with yaml.

    Returns:
        list: A list of ints for the band numbers, starting from 0. 0 would
        represent the blue band, 1 green, and so on. 
        
        See https://landsat.usgs.gov/what-are-band-designations-landsat-satellites

    .. _PEP 484:
        https://www.python.org/dev/peps/pep-0484/

    """
    band_list = []
    if params["image_vals"]["dataset"] == "landsat":
        bands = params["landsat_bands_to_include"]
    for i, band in enumerate(bands):
        if list(band.values())[0] == True:
            band_list.append(i)
    return band_list


def load_gs_wv2(scene_dir_path, band_list):
    """Load the landsat bands specified by yaml_to_band_index and returns 
    a [H,W,N] Numpy array for a single scene, where N is the number of bands 
    and H and W are the height and width of the original band arrays. 
    Channels are ordered in band order.

    Args:
        scene_dir_path (str): The path to the scene directory. The dir name should be the standard scene id that is the same as
        as the blob name of the folder that has the landsat product bands downloaded using lsru or
        download_utils.
        band_list (str): a list of band indices to include

    Returns:
        ndarray:  
        
    .. _PEP 484:
        https://www.python.org/dev/peps/pep-0484/

    """
    # Load image
    product_list = os.listdir(scene_dir_path)
    # below works because only products that are bands have a int in the 4th to last position
    filtered_product_list = [band for band in product_list if band[-4] in band_list]
    filtered_product_list = sorted(filtered_product_list)
    filtered_product_paths = [os.path.join(scene_dir_path, fname) for fname in filtered_product_list]
    arr_list = [skio.imread(product_path) for product_path in filtered_product_paths]
    stacked_arr = np.dstack(arr_list)
    stacked_arr[stacked_arr <= 0]=0
    reorder_path = os.path.join(REORDER_DIR,image_id+'.tif')
    skio.imsave(reorder_path,stacked_arr, plugin='tifffile')

def negative_buffer_and_small_filter(params):
    """
    Applies a negative buffer to labels since some are too close together and 
    produce conjoined instances when connected components is run (even after 
    erosion/dilation). This may not get rid of all conjoinments and should be adjusted.
    It relies too on the source projection of the label file to calculate distances for
    the negative buffer. It's assumed that the projection is in meters and that a negative buffer in meter units 
    will work with this projection.

    Returns rasterized labels that are ready to be gridded
    """

    class_int = params["label_vals"]["class"]
    neg_buffer = float(params["label_vals"]["neg_buffer"])
    small_area_filter = float(params["label_vals"]["small_area_filter"])
    big_area_filter = float(params["label_vals"]["big_area_filter"])
    # This is a helper  used with sorted for a list of strings by specific indices in
    # each string. Was used for a long path that ended with a file name
    # Not needed here but may be with different source imagery and labels
    # def takefirst_two(elem):
    #     return int(elem[-12:-10])

    items = os.listdir(SOURCE_LABELS)
    labels = []
    for name in items:
        if name.endswith(".shp") or name.endswith(".geojson"):
            labels.append(os.path.join(SOURCE_LABELS, name))

    shp_list = sorted(labels)
    # need to use Source imagery for geotransform data for rasterized shapes, didn't preserve when save imgs to reorder
    scenes = os.listdir(SOURCE_IMGS)

    scenes = [scene for scene in scenes if ".tif" in scene and ".aux" not in scene]
    img_list = []
    for name in scenes:
        img_list.append(os.path.join(SOURCE_IMGS, name))

    img_list = sorted(img_list)
    for shp_path, img_path in zip(shp_list, img_list):
        shp_frame = gpd.read_file(shp_path)
        # keeps the class of interest if it is there and the polygon of raster extent
        with rasterio.open(img_path) as rast:
            meta = rast.meta.copy()
            meta.update(compress="lzw")
            meta["count"] = 1
            tifname = os.path.splitext(os.path.basename(shp_path))[0] + ".tif"
            rasterized_name = os.path.join(NEG_BUFFERED, tifname)
            with rasterio.open(rasterized_name, "w+", **meta) as out:
                out_arr = out.read(1)
                shp_frame = shp_frame.loc[shp_frame.area > small_area_filter]
                shp_frame = shp_frame.loc[shp_frame.area < big_area_filter]
                shp_frame["geometry"] = shp_frame["geometry"].buffer(neg_buffer)
                # https://gis.stackexchange.com/questions/151339/rasterize-a-shapefile-with-geopandas-or-fiona-python#151861
                shapes = (
                    (geom, value)
                    for geom, value in zip(shp_frame.geometry, shp_frame.ObjectID)
                )
                burned = features.rasterize(
                    shapes=shapes,
                    fill=0,
                    out_shape=rast.shape,
                    transform=out.transform,
                    default_value=1,
                )
                burned[burned < 0] = 0
                burned[burned > 0] = 1
                burned = burned.astype(np.int16, copy=False)
                out.write(burned, 1)
    print(
        "Done applying negbuff of {negbuff} and filtering small labels of area less than {area}".format(
            negbuff=neg_buffer, area=small_area_filter
        )
    )


def rm_mostly_empty(scene_path, label_path):
    """
    Removes a grid that is mostly (over 1/4th) empty and corrects bad no data value to 0.
    Ignor ethe User Warning, unsure why it pops up but doesn't seem to impact the array shape
    """

    usable_data_threshold = params["image_vals"]["usable_thresh"]
    arr = skio.imread(scene_path)
    arr[arr < 0] = 0
    skio.imsave(scene_path, arr)
    pixel_count = arr.shape[0] * arr.shape[1]
    nodata_pixel_count = (arr == 0).sum()
    if nodata_pixel_count / pixel_count > usable_data_threshold:

        os.remove(scene_path)
        os.remove(label_path)
        print("removed scene and label, {}% bad data".format(usable_data_threshold))


def grid_images(params):
    """
    Grids up imagery to a variable size. Filters out imagery with too little usable data.
    appends a random unique id to each tif and label pair, appending string 'label' to the 
    mask.
    """
    if params["image_vals"]["img_id"] is str:
        img_list = [params["image_vals"]["img_id"]]
        label_list = sorted(next(os.walk(NEG_BUFFERED))[2])
        print(
            "label list def for single id should change later to specifically reference the id!"
        )
    else:
        img_list = sorted(next(os.walk(REORDER))[2])
        label_list = sorted(next(os.walk(NEG_BUFFERED))[2])
    for img_name, label_name in zip(img_list, label_list):
        img_path = os.path.join(REORDER, img_name)
        label_path = os.path.join(NEG_BUFFERED, label_name)
        # assign unique name to each gridded tif, keeping season suffix
        # assigning int of same length as ZA0932324 naming convention

        tile_size_x = params["image_vals"]["grid_size"]
        tile_size_y = params["image_vals"]["grid_size"]
        ds = gdal.Open(img_path)
        band = ds.GetRasterBand(1)
        xsize = band.XSize
        ysize = band.YSize

        for i in range(0, xsize, tile_size_x):
            for j in range(0, ysize, tile_size_y):
                unique_id = str(random.randint(100000000, 999999999))
                out_path_img = os.path.join(GRIDDED_IMGS, unique_id) + ".tif"
                out_path_label = os.path.join(GRIDDED_LABELS, unique_id) + "_label.tif"
                com_string = (
                    "gdal_translate -of GTIFF -srcwin "
                    + str(i)
                    + ", "
                    + str(j)
                    + ", "
                    + str(tile_size_x)
                    + ", "
                    + str(tile_size_y)
                    + " "
                    + str(img_path)
                    + " "
                    + str(out_path_img)
                )
                os.system(com_string)
                com_string = (
                    "gdal_translate -of GTIFF -srcwin "
                    + str(i)
                    + ", "
                    + str(j)
                    + ", "
                    + str(tile_size_x)
                    + ", "
                    + str(tile_size_y)
                    + " "
                    + str(label_path)
                    + " "
                    + str(out_path_label)
                )
                os.system(com_string)
                rm_mostly_empty(out_path_img, out_path_label)


def open_labels(params):
    """
    Opens labels with kernel as defined in config.
    """
    k = params["label_vals"]["kernel"]
    label_list = next(os.walk(GRIDDED_LABELS))[2]
    if params["label_vals"]["open"] == True:
        for name in label_list:
            arr = skio.imread(os.path.join(GRIDDED_LABELS, name))
            arr[arr < 0] = 0
            opened_path = os.path.join(OPENED, name)
            kernel = np.ones((k, k))
            arr = skim.binary_opening(arr, kernel)
            arr = 1 * arr
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                skio.imsave(opened_path, 1 * arr)

        print("Done opening with kernel of h and w {size}".format(size=k))

    else:
        for name in label_list:
            arr = skio.imread(os.path.join(GRIDDED_LABELS, name))
            arr[arr < 0] = 0
            opened_path = os.path.join(OPENED, name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                skio.imsave(opened_path, 1 * arr)


def move_img_to_folder(params):
    """Moves a file with identifier pattern 760165086_OSGS.tif 
    (or just760165086.tif) to a 
    folder path ZA0165086/image/ZA0165086.tif
    Also creates a mask folder at ZA0165086/masks
    """

    image_list = os.listdir(GRIDDED_IMGS)
    for img in image_list:

        folder_name = os.path.join(TRAIN, img[:9])
        os.mkdir(folder_name)
        new_path = os.path.join(folder_name, "image")
        mask_path = os.path.join(folder_name, "mask")
        os.mkdir(new_path)
        file_path = os.path.join(GRIDDED_IMGS, img)
        os.rename(file_path, os.path.join(new_path, img[:9] + ".tif"))
        os.mkdir(mask_path)


def connected_comp(params):
    """
    Extracts individual instances into their own tif files. Saves them
    in each folder ID in train folder. If an image has no instances,
    saves it with a empty mask.
    """
    label_list = next(os.walk(OPENED))[2]
    # save connected components and give each a number at end of id
    for name in label_list:
        arr = skio.imread(os.path.join(OPENED, name))
        blob_labels = measure.label(arr, background=0)
        blob_vals = np.unique(blob_labels)
        # for imgs with no isntances, create empty mask
        if len(blob_vals) == 1:
            img_folder = os.path.join(TRAIN, name[:9], "image")
            img_name = os.listdir(img_folder)[0]
            img_path = os.path.join(img_folder, img_name)
            arr = skio.imread(img_path)
            mask = np.zeros_like(arr[:, :, 0])
            mask_folder = os.path.join(TRAIN, name[:9], "mask")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                label_stump = os.path.splitext(os.path.basename(name))[0]
                skio.imsave(os.path.join(mask_folder, label_stump + "_0.tif"), mask)
        # only run connected comp if there is at least one instance
        for blob_val in blob_vals[blob_vals != 0]:
            labels_copy = blob_labels.copy()
            labels_copy[blob_labels != blob_val] = 0
            labels_copy[blob_labels == blob_val] = 1

            label_stump = os.path.splitext(os.path.basename(name))[0]
            label_name = label_stump + "_" + str(blob_val) + ".tif"
            mask_path = os.path.join(TRAIN, name[:9], "mask")
            label_path = os.path.join(mask_path, label_name)
            assert labels_copy.ndim == 2
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                skio.imsave(label_path, labels_copy)


def train_test_split(params):
    """Takes a sample of folder ids and copies them to a test directory
    from a directory with all folder ids. Each sample folder contains an 
    images and corresponding masks folder."""

    k = params["image_vals"]["split"]
    sample_list = next(os.walk(TRAIN))[1]
    k = round(k * len(sample_list))
    test_list = random.sample(sample_list, k)
    for test_sample in test_list:
        shutil.copytree(
            os.path.join(TRAIN, test_sample), os.path.join(TEST, test_sample)
        )
    train_list = list(set(next(os.walk(TRAIN))[1]) - set(next(os.walk(TEST))[1]))
    train_df = pd.DataFrame({"train": train_list})
    test_df = pd.DataFrame({"test": test_list})
    train_df.to_csv(os.path.join(RESULTS, "train_ids.csv"))
    test_df.to_csv(os.path.join(RESULTS, "test_ids.csv"))


def get_arr_channel_mean(channel):
    """
    Calculate the mean of a given channel across all training samples.
    """

    means = []
    train_list = list(set(next(os.walk(TRAIN))[1]) - set(TEST))
    for i, fid in enumerate(train_list):
        im_folder = os.path.join(TRAIN, fid, "image")
        im_path = os.path.join(im_folder, os.listdir(im_folder)[0])
        arr = skio.imread(im_path)
        arr = arr.astype(np.float32, copy=False)
        # added because no data values different for wv2 and landsat, need to exclude from mean
        nodata_value = arr.min() if arr.min() < 0 else -9999
        arr[arr == nodata_value] = np.nan
        means.append(np.nanmean(arr[:, :, channel]))
    print(np.mean(means))


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
