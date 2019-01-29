### currently random, useful functions
from skimage import exposure
import numpy as np

def percentile_rescale(arr):
    '''
    Rescales and applies other exposure functions to improve image vis. 
    http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.rescale_intensity
    '''
    rescaled_arr = np.zeros_like(arr)
    for i in range(0,arr.shape[-1]):
        val_range = (np.percentile(arr[:,:,i], 1), np.percentile(arr[:,:,i], 99))
        rescaled_channel = exposure.rescale_intensity(arr[:,:,i], val_range)
        rescaled_arr[:,:,i] = rescaled_channel
        #rescaled_arr= exposure.adjust_gamma(rescaled_arr, gamma=1) #adjust from 1 either way
#     rescaled_arr= exposure.adjust_sigmoid(rescaled_arr, cutoff=.50) #adjust from .5 either way 
    return rescaled_arr

def remove_dir_folders(directory):
    '''
    Removes all files and sub-folders in a folder and keeps the folder.
    '''

    folderlist = [ f for f in os.listdir(directory)]
    for f in folderlist:
        shutil.rmtree(os.path.join(directory,f))
        
def max_normalize(arr):
    arr *= (255.0/arr.max())
    return arr