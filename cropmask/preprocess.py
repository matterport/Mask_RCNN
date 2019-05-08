import random
import os
import shutil
import copy
import skimage.io as skio
import warnings
import pandas as pd
import numpy as np
import geopandas as gpd
from rasterio import features, coords
from rasterio.plot import reshape_as_raster
import rasterio
from shapely.geometry import shape
from osgeo import gdal
from itertools import product
from rasterio import windows
from multiprocessing.pool import ThreadPool
from cropmask.label_prep import rio_bbox_to_polygon
from cropmask.misc import parse_yaml, make_dirs
from cropmask import grid, label_prep

random.seed(42)

class PreprocessWorkflow():
    """
    Worflow for loading and gridding a single satellite image and reference dataset of the same extent.
    """
    
    def __init__(self, param_path, scene_dir_path, source_label_path):
        params = parse_yaml(param_path)
        self.params = params
        self.source_label_path = source_label_path # if there is a referenc label
        self.scene_dir_path = scene_dir_path # path to the unpacked tar archive on azure storage
        self.scene_id = self.scene_dir_path.split("/")[-2] # gets the name of the folder the bands are in, the scene_id
        
         # the folder structure for the unique run
        self.ROOT = params['dirs']["root"]
        assert os.path.exists(self.ROOT)
        self.DATASET = os.path.join(self.ROOT, params['dirs']["dataset"])
        self.STACKED = os.path.join(self.DATASET, params['dirs']["stacked"])
        self.TRAIN = os.path.join(self.DATASET, params['dirs']["train"])
        self.TEST = os.path.join(self.DATASET, params['dirs']["test"])
        self.GRIDDED_IMGS = os.path.join(self.DATASET, params['dirs']["gridded_imgs"])
        self.GRIDDED_LABELS = os.path.join(self.DATASET, params['dirs']["gridded_labels"])
        self.NEG_BUFFERED = os.path.join(self.DATASET, params['dirs']["neg_buffered_labels"])
        self.RESULTS = os.path.join(self.ROOT, params['dirs']["results"], params['dirs']["dataset"])
        
        # scene specific paths and variables
        self.rasterized_label_path = ''
        self.band_list = [] # the band indices
        self.meta = {} # meta data for the stacked raster
        self.chip_ids = [] # list of chip ids of form [scene_id]_[random number]
        self.small_area_filter = params['label_vals']['small_area_filter']
        self.neg_buffer = params['label_vals']['neg_buffer']
        self.ag_class_int = params['label_vals']['ag_class_int'] # TO DO, not implemented but needs to be for multi class
        self.dataset_name = params['image_vals']['dataset_name']
        self.grid_size = params['image_vals']['grid_size']
        self.usable_threshold = params['image_vals']['usable_thresh']
        self.split = params['image_vals']['split']

    def yaml_to_band_index(self):
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
        if self.params["image_vals"]["dataset_name"] == "landsat-5":
            bands = self.params["landsat_bands_to_include"]
        for i, band in enumerate(bands):
            if list(band.values())[0] == True:
                self.band_list.append(str(i+1))
        return self.band_list

    def setup_dirs(self):
        """
        This folder structure is used for each unique pre processing and modeling
        workflow and is made unique by specifying a unique DATASET name
        or ROOT path (if working on a different container.). 
        
        ROOT should be the path to the azure container mounted with blobfuse, 
        and should already exist. The RESULTS folder should be created in a folder named from param["results"], and this should also already exist.
        """

        directory_list = [
                self.DATASET,
                self.STACKED,
                self.TRAIN,
                self.TEST,
                self.GRIDDED_IMGS,
                self.GRIDDED_LABELS,
                self.NEG_BUFFERED,
                self.RESULTS,
            ]
        if os.path.exists(os.path.join(self.ROOT, self.params['dirs']["results"])) == False:
            os.mkdir(os.path.join(self.ROOT, self.params['dirs']["results"]))
        make_dirs(directory_list)
        return directory_list
    
    def get_product_paths(self, band_list):
        # Load image
        product_list = os.listdir(self.scene_dir_path)
        # below works because only products that are bands have a int in the 5th to last position
        filtered_product_list = [band for band in product_list if band[-5] in band_list and 'band' in band]
        filtered_product_list = sorted(filtered_product_list)
        filtered_product_paths = [os.path.join(self.scene_dir_path, fname) for fname in filtered_product_list]
        return filtered_product_paths
    
    def load_and_stack_bands(self, product_paths):
        arr_list = [skio.imread(product_path) for product_path in product_paths]
        # get metadata and edit meta obj for stacked raster
        with rasterio.open(product_paths[0]) as rast:
                meta = rast.meta.copy()
                meta.update(compress="lzw")
                meta["count"] = len(arr_list)
                self.meta=meta
                self.bounds = rast.bounds
        stacked_arr = np.dstack(arr_list)
        stacked_arr[stacked_arr <= 0]=0
        return stacked_arr
        
    def stack_and_save_bands(self):
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
            ndarray:k 

        .. _PEP 484:k 
            https://www.python.org/dev/peps/pep-0484/

        """
        
        product_paths = self.get_product_paths(self.band_list)
        stacked_arr = self.load_and_stack_bands(product_paths)
        stacked_name = os.path.basename(product_paths[0])[:-10] + ".tif"
        stacked_path = os.path.join(self.STACKED, stacked_name)
        self.stacked_path = stacked_path
        with rasterio.open(stacked_path, "w+", **self.meta) as out:
            out.write(reshape_as_raster(stacked_arr))
            
    def preprocess_labels(self):
        """For preprcoessing reference dataset"""
        shp_frame = gpd.read_file(self.source_label_path)
        # keeps the class of interest if it is there and the polygon of raster extent
        shp_frame = shp_frame.to_crs(self.meta['crs'].to_dict()) # reprojects to landsat's utm zone
        tif_polygon = rio_bbox_to_polygon(self.bounds) 
        shp_series = shp_frame.intersection(tif_polygon) # clips by landsat's bounds including nodata corners
        shp_series = shp_series.loc[shp_series.area > self.small_area_filter]
        shp_series = shp_series.buffer(self.neg_buffer)
        return shp_series.loc[shp_series.is_empty==False]
            
    def negative_buffer_and_small_filter(self, neg_buffer, small_area_filter):
        """
        Applies a negative buffer to labels since some are too close together and 
        produce conjoined instances when connected components is run (even after 
        erosion/dilation). This may not get rid of all conjoinments and should be adjusted.
        It relies too on the source projection of the label file to calculate distances for
        the negative buffer. It's assumed that the projection is in meters and that a negative buffer in meter units 
        will work with this projection.
        
        Args:
            source_label_path (str): the path to the reference shapefile dataset. Should be the same extent as a Landsat scene
            neg_buffer (float): The distance in meters to use for the negative buffer. Should at least be 1 pixel width.
            small_area_filter (float): The area thershold to remove spurious small fields. Particularly useful to remove fields                                         to small to be commercial agriculture

        Returns rasterized labels that are ready to be gridded
        """
        shp_series = self.preprocess_labels()
        meta = self.meta.copy()
        meta.update({'count':1})
        tifname = os.path.splitext(os.path.basename(self.source_label_path))[0] + ".tif"
        self.rasterized_label_path = os.path.join(self.NEG_BUFFERED, tifname)
        with rasterio.open(self.rasterized_label_path, "w+", **meta) as out:
            out_arr = out.read(1)
            # https://gis.stackexchange.com/questions/151339/rasterize-a-shapefile-with-geopandas-or-fiona-python#151861
            shapes = shp_series.values.tolist()
            burned = features.rasterize(
                shapes=shapes,
                fill=0,
                out_shape=(meta['height'],meta['width']),
                transform=out.transform,
                default_value=1,
            )
            burned[burned < 0] = 0
            burned[burned > 0] = 1
            burned = burned.astype(np.int16, copy=False)
            out.write(burned, 1)
        print(
            "Done applying negbuff of {negbuff} and filtering small labels of area less than {area}".format(
                negbuff=self.neg_buffer, area=self.small_area_filter
                )
            )
        return burned # for testing to confirm it worked
        
    def grid_images(self):
        """
        Grids up imagery to a variable size. Filters out imagery with too little usable data.
        appends a random unique id to each tif and label pair, appending string 'label' to the 
        mask.
        """
        chip_img_paths = grid.grid_images_rasterio_controlled_threads(self.stacked_path, self.GRIDDED_IMGS, output_name_template='tile_{}-{}.tif', grid_size=self.grid_size)
        chip_label_paths = grid.grid_images_rasterio_controlled_threads(self.rasterized_label_path, self.GRIDDED_LABELS, output_name_template='tile_{}-{}_label.tif', grid_size=self.grid_size)
        return (chip_img_paths, chip_label_paths)
                
                
    def rm_mostly_empty(self, scene_path, label_path):
        """
        Removes a grid that is emptier than the usable data threshold and corrects bad no data value to 0.
        Ignore the User Warning, unsure why it pops up but doesn't seem to impact the array shape. Used because
        very empty grid chips seem to throw off the model by increasing detections at the edges between good data 
        and nodata.
        """

        arr = skio.imread(scene_path)
        arr[arr < 0] = 0
        skio.imsave(scene_path, arr)
        pixel_count = arr.shape[0] * arr.shape[1]
        nodata_pixel_count = (arr == 0).sum()

        if nodata_pixel_count / pixel_count > self.usable_threshold:

            os.remove(scene_path)
            os.remove(label_path)
            print("removed scene and label, {}% bad data".format(self.usable_threshold))
                
    def remove_from_gridded(self, chip_img_paths, chip_label_paths):
                
        def get_chip_id(string,ignore_string):
            return string.replace(ignore_string, '')

        for img, label in zip(sorted(chip_img_paths, key=lambda x: get_chip_id(x,'.tif')), sorted(chip_label_paths, key=lambda x: get_chip_id(x,'_label.tif'))):
            self.rm_mostly_empty(img,label)
                
    def move_chips_to_folder(self):
        """Moves a file with identifier pattern 760165086.tif to a 
        folder path ZA0165086/image/ZA0165086.tif
        
        """

        for chip_id in os.listdir(self.GRIDDED_IMGS):
            chip_id = os.path.splitext(chip_id)[0]
            chip_folder_path = os.path.join(self.TRAIN, chip_id)
            if os.path.exists(chip_folder_path) == False:
                os.mkdir(chip_folder_path)
            else:
                raise Exception('{} should not exist prior to being created in this function, it has not been deleted properly prior to a new run'.format(folder_path)) 
            new_chip_path = os.path.join(chip_folder_path, "image")
            mask_path = os.path.join(chip_folder_path, "mask")
            os.mkdir(new_chip_path)
            os.mkdir(mask_path)
            old_chip_path = os.path.join(self.GRIDDED_IMGS, chip_id+'.tif')
            os.rename(old_chip_path, os.path.join(new_chip_path, chip_id + ".tif")) # moves the chips   
    
    def connected_components(self):
        """
        Extracts individual instances into their own tif files. Saves them
        in each folder ID in train folder. If an image has no instances,
        saves it with a empty mask.
        """
        for chip_id in os.listdir(self.TRAIN):
            chip_label_path = os.path.join(self.GRIDDED_LABELS, chip_id+"_label.tif")
            arr = skio.imread(chip_label_path)

            blob_labels = label_prep.connected_components(arr)
            # for imgs with no instances, create empty mask
            if len(np.unique(blob_labels)) == 1:
                mask_folder = os.path.join(self.TRAIN, chip_id, "mask")
                skio.imsave(os.path.join(mask_folder, chip_id + "_0.tif"), blob_labels)
            else:
                # only run connected comp if there is at least one instance
                label_list = label_prep.extract_labels(blob_labels)
                label_arrs = np.stack(label_list, axis=-1)
                label_name = chip_id + "_labels.tif"
                mask_path = os.path.join(self.TRAIN, chip_id, "mask")
                label_path = os.path.join(mask_path, label_name)
                skio.imsave(label_path, label_arrs)
            
    def train_test_split(self):
        """Takes a sample of folder ids and copies them to a test directory
        from a directory with all folder ids. Each sample folder contains an 
        images and corresponding masks folder."""

        sample_list = next(os.walk(self.TRAIN))[1]
        k = round(self.split * len(sample_list))
        test_list = random.sample(sample_list, k)
        for test_sample in test_list:
            os.rename(
                os.path.join(self.TRAIN, test_sample), os.path.join(self.TEST, test_sample)
            )
        train_list = list(set(next(os.walk(self.TRAIN))[1]) - set(next(os.walk(self.TEST))[1]))
        train_df = pd.DataFrame({"train": train_list})
        test_df = pd.DataFrame({"test": test_list})
        train_df.to_csv(os.path.join(self.RESULTS, "train_ids.csv")) # used for reading in trainging and testing in the modeling step
        test_df.to_csv(os.path.join(self.RESULTS, "test_ids.csv"))

    def get_arr_channel_mean(self, channel):
        """
        Calculate the mean of a given channel across all training samples.
        """

        means = []
        train_list = next(os.walk(self.TRAIN))[1]
        for i, fid in enumerate(train_list):
            im_folder = os.path.join(self.TRAIN, fid, "image")
            im_path = os.path.join(im_folder, os.listdir(im_folder)[0])
            arr = skio.imread(im_path)
            arr = arr.astype(np.float32, copy=False)
            # added because no data values different for wv2 and landsat, need to exclude from mean
            nodata_value = 0 # best to do no data masking up front and set bad qa bands to 0 rather than assuming 0 is no data. This is assumed from looking at no data values at corners being equal to 0
            arr[arr == nodata_value] = np.nan
            means.append(np.nanmean(arr[:, :, channel]))
        print(np.mean(means))
