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
from rasterio.plot import reshape_as_raster
import rasterio
from shapely.geometry import shape
import gdal




def make_dirs(directory_list):

    # Make directory and subdirectories
    for d in directory_list:
        try: 
            pathlib.Path(d).mkdir(parents=False, exist_ok=False)
        except: 
            FileExistsError
    # Change working directory to project directory
    os.chdir(directory_list[0])


def yaml_to_band_index(params):
    """Parses config booleans to a list of band indexes to be stacked.

    For example, Landsat 5 has 6 bands (7 if you count band 6, thermal) 
    that we can use for masking.

    Args:
        params (dict): The configuration dictionary that is read with yaml.

    Returns:
        list: A list of strings for the band numbers, starting from 1. For Landsat 5 1 would
        represent the blue band, 2 green, and so on. For Landsat 8, band 1 would be coastal blue,
        band 2 would be blue, and so on.
        
        See https://landsat.usgs.gov/what-are-band-designations-landsat-satellites

    .. _PEP 484:
        https://www.python.org/dev/peps/pep-0484/

    """
    band_list = []
    if params["image_vals"]["dataset"] == "landsat":
        bands = params["landsat_bands_to_include"]
    for i, band in enumerate(bands):
        if list(band.values())[0] == True:
            band_list.append(i+1)
    return [str(b) for b in band_list]



def stack_and_save_all(scenes_source_dir, band_list, out_dir):
    """Runs stack_and_save_bands for all ordered scenes in SOURCE_IMGS. SOURCE_IMGS should contain a set of unpacked tar archives, and can be a single region order or a couple of concatenated region orders for training and testing on multiple geographies.

    Args:
        scenes_source_dir (str): the path to the dir with the source scenes
        band_list (str): a list of band indices to include
        out_dir (str): the path to the stacked directory

    Returns:
        ndarray:  
        
    .. _PEP 484:
        https://www.python.org/dev/peps/pep-0484/

    """
    
    
    
    









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
