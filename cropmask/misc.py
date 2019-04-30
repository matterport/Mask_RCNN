### currently random, useful functions
from skimage import exposure
import numpy as np
import yaml
import shutil
import os

def percentile_rescale(arr):
    """
    Rescales and applies other exposure functions to improve image vis. 
    http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.rescale_intensity
    """
    rescaled_arr = np.zeros_like(arr)
    for i in range(0, arr.shape[-1]):
        val_range = (np.percentile(arr[:, :, i], 1), np.percentile(arr[:, :, i], 99))
        rescaled_channel = exposure.rescale_intensity(arr[:, :, i], val_range)
        rescaled_arr[:, :, i] = rescaled_channel
        # rescaled_arr= exposure.adjust_gamma(rescaled_arr, gamma=1) #adjust from 1 either way
    #     rescaled_arr= exposure.adjust_sigmoid(rescaled_arr, cutoff=.50) #adjust from .5 either way
    return rescaled_arr

def remove_dirs(directory_list):
    """
    Removes all files and sub-folders in each folder.
    """

    for f in directory_list:
        if os.path.exists(f):
            shutil.rmtree(f)

def max_normalize(arr):
    arr *= 255.0 / arr.max()
    return arr

def parse_yaml(input_file):
    """Parse yaml file of configuration parameters."""
    with open(input_file, "r") as yaml_file:
        params = yaml.safe_load(yaml_file)
    return params

def make_dirs(directory_list):

    # Make directory and subdirectories
    for d in directory_list:
        try: 
            os.mkdir(d)
        except:
            print("Whole directory list: ", directory_list)
            print("The directory "+d+" exists already. Check it and maybe delete it or change config.")
            raise FileExistsError
