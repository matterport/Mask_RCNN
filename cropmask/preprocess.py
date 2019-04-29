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
import geopandas as gpd
from rasterio import features, coords
from rasterio.plot import reshape_as_raster
import rasterio
from shapely.geometry import shape
import gdal

from cropmask.misc import parse_yaml, make_dirs

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
        self.scene_id = self.scene_dir_path.split("/")[-1] # gets the name of the folder the bands are in, the scene_id
        self.stacked_path = '' # path to the stacked bands
        self.rasterized_label_path = ''
        self.gridded_imgs_dir = ''
        self.gridded_labels_dir = ''
        self.results_dir = params['dirs']['results']
        self.band_list = [] # the band indices
        self.meta = # meta data for the stacked raster
        self.chip_ids = [] # list of chip ids of form [scene_id]_[random number]
        self.small_area_filter = params['label_vals']['small_area_filter']
        self.neg_buffer = params['label_vals']['neg_buffer']
        self.ag_class_int = params['label_vals']['ag_class_int'] # TO DO, not implemented but needs to be for multi class
        self.dataset_name = params['image_vals']['dataset_name']
        self.grid_size = params['image_vals']['grid_size']
        self.usable_threshold = params['image_vals']['usable_thresh']
        self.split = params['image_vals']['split']
O
    def yaml_to_band_index():
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
        if self.params["image_vals"]["dataset_name"] == "landsat5":
            bands = self.params["landsat_bands_to_include"]
        for i, band in enumerate(bands):
            if list(band.values())[0] == True:
                self.band_list.append(i+1)
        return [str(b) for b in self.band_list]

    def setup_dirs():
        """
        A path to a yaml config file that will be parsed as a dictionary 
        containing the directory paths and names to create.

        This folder structure is used for each unique pre processing and modeling
        workflow and is made unique by specifying a unique DATASET name
        or ROOT path (if working on a different container.). 
        
        ROOT should be the path to the azure container mounted with blobfuse, 
        and should already exist.
        """
        
        params = self.params['dirs']

        ROOT = params["container"]

        assert os.path.exists(ROOT)

        DATASET = os.path.join(ROOT, params["dataset"])

        STACKED = os.path.join(DATASET, params["stacked"])

        TRAIN = os.path.join(DATASET, params["train"])

        TEST = os.path.join(DATASET, params["test"])

        GRIDDED_IMGS = os.path.join(DATASET, params["gridded_imgs"])

        GRIDDED_LABELS = os.path.join(DATASET, params["gridded_labels"])

        NEG_BUFFERED = os.path.join(DATASET, params["neg_buffered_labels"])

        RESULTS = os.path.join(ROOT, params["results"], DATASET)

        SOURCE_IMGS = os.path.join(ROOT, params["source_imgs"])

        SOURCE_LABELS = os.path.join(ROOT, params["source_labels"])k 

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
        make_dirs(directory_list)        
        
    def stack_and_save_bands(self, out_dir):
        """Load the landsat bands specified by yaml_to_band_index and returns 
        a [H,W,N] Numpy array for a single scene, where N is the number of bands 
        and H and W are the height and width of the original band arrays. 
        Channels are ordered in band order.

        Args:
            scene_dir_path (str): The path to the scene directory. The dir name should be the standard scene id that is the same as
            as the blob name of the folder that has the landsat product bands downloaded using lsru or
            download_utils.
            band_list (str): a list of band indices to include
            out_dir (str): the path to the stacked directory

        Returns:
            ndarray:k 

        .. _PEP 484:k 
            https://www.python.org/dev/peps/pep-0484/

        """
        # Load image
        product_list = os.listdir(self.scene_dir_path)
        # below works because only products that are bands have a int in the 5th to last position
        filtered_product_list = [band for band in product_list if band[-5] in self.band_list]
        filtered_product_list = sorted(filtered_product_list)
        filtered_product_paths = [os.path.join(self.scene_dir_path, fname) for fname in filtered_product_list]
        arr_list = [skio.imread(product_path) for product_path in filtered_product_paths]
        # get metadata and edit meta obj for stacked raster
        with rasterio.open(filtered_product_paths[0]) as rast:
                meta = rast.meta.copy()
                meta.update(compress="lzw")
                meta["count"] = len(arr_list)
                self.meta=meta
        stacked_arr = np.dstack(arr_list)
        stacked_arr[stacked_arr <= 0]=0
        stacked_name = filtered_product_list[0][:-10] + ".tif"
        stacked_path = os.path.join(out_dir, stacked_name)
        self.stacked_path = stacked_path
        with rasterio.open(stacked_path, "w+", **meta) as out:
            out.write(reshape_as_raster(stacked_arr))
            
    def negative_buffer_and_small_filter(dest_path, neg_buffer, small_area_filter):
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

        shp_frame = gpd.read_file(self.source_label_path)
        # keeps the class of interest if it is there and the polygon of raster extent
        with rasterio.open(self.stacked_path) as rast:
            meta = rast.meta.copy()
            meta.update(compress="lzw")
            meta["count"] = 1
            self.meta = meta
            tifname = os.path.splitext(os.path.basename(self.source_label_path))[0] + ".tif"
            self.rasterized_label_path = os.path.join(dest_path, tifname)
        with rasterio.open(rasterized_label_path, "w+", **meta) as out:
            out_arr = out.read(1)
            shp_frame = shp_frame.loc[shp_frame.area > self.small_area_filter]
            shp_frame["geometry"] = shp_frame["geometry"].buffer(self.neg_buffer)
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
                negbuff=self.neg_buffer, area=self.small_area_filter
                )
            )

    def grid_images():
        """
        Grids up imagery to a variable size. Filters out imagery with too little usable data.
        appends a random unique id to each tif and label pair, appending string 'label' to the 
        mask.
        """
        
        def rm_mostly_empty(scene_path, label_path):
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
        

        ds = gdal.Open(self.stacked_path) # TO DO can get this info earlier and with less file io
        band = ds.GetRasterBand(1)
        xsize = band.XSize
        ysize = band.YSize
        ### TO DO MAKE GRID ID DIRECTORY CREATION HAPPEN IN THE MAKE DIRS FUNCTION, MOVE ALL DIR SETUP TO THAT FUNCTION INSTEAD OF HAVING
        ### IT BE SCATTERED AROUND MULTIPLE FUNCS
        for i in range(0, xsize, self.grid_size):
            for j in range(0, ysize, self.grid_size):
                chip_id = str(random.randint(100000000, 999999999))+'_'+self.scene_id
                self.chip_ids.append(chip_id)
                out_path_img = os.path.join(self.gridded_imgs_dir, chip_id) + ".tif"
                out_path_label = os.path.join(self.gridded_labels_dir, chip_id) + "_label.tif"
                com_string = (
                    "gdal_translate -of GTIFF -srcwin "
                    + str(i)
                    + ", "
                    + str(j)
                    + ", "
                    + str(self.grid_size)
                    + ", "
                    + str(self.grid_size)
                    + " "
                    + str(self.stacked_path)
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
                    + str(self.grid_size)
                    + ", "
                    + str(self.grid_size)
                    + " "
                    + str(self.rasterized_label_path)
                    + " "
                    + str(out_path_label)
                )
                os.system(com_string)
                rm_mostly_empty(out_path_img, out_path_label)
                
    def move_img_to_folder():
        """Moves a file with identifier pattern 760165086.tif to a 
        folder path ZA0165086/image/ZA0165086.tif
        
        """

        for chip_id in self.chip_ids:
            chip_folder_path = os.path.join(self.train_dir, chip_id)
            if os.path.exists(chip_folder_path) == False:
                os.mkdir(chip_folder_path)
            else:
                raise Exception('{} should not exist prior to being created in this function, it has not been deleted properly prior to a new run'.format(folder_path)) 
            new_chip_path = os.path.join(chip_folder_path, "image")
            mask_path = os.path.join(chip_folder_path, "mask")
            os.mkdir(new_path)
            os.mkdir(mask_path)
            old_chip_path = os.path.join(self.gridded_imgs_dir, self.chip_id+'.tif')
            os.rename(old_chip_path, os.path.join(new_chip_path, self.chip_id+'_'+self.scene_id + ".tif")) #names each gridded chip with randomID and scene_id
        
    def connected_comp():
    """
    Extracts individual instances into their own tif files. Saves them
    in each folder ID in train folder. If an image has no instances,
    saves it with a empty mask.
    """
        label_list = next(os.walk(self.rasterized_label_path))[2]
        # save connected components and give each a number at end of id
        for label_chip in label_list:
            arr = skio.imread(os.path.join(self.rasterized_label_path, label_chip))
            blob_labels = measure.label(arr, background=0)
            blob_vals = np.unique(blob_labels)
            # for imgs with no instances, create empty mask
            if len(blob_vals) == 1:
                img_chip_folder = os.path.join(self.train_dir, label_chip[:-11], "image")
                img_chip_name = os.listdir(img_chip_folder)[0]
                img_chip_path = os.path.join(img_chip_folder, img_chip_name)
                arr = skio.imread(img_chip_path)
                mask = np.zeros_like(arr[:, :, 0])
                mask_folder = os.path.join(self.train_dir, label_chip[:-11], "mask")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    label_stump = os.path.splitext(os.path.basename(label_chip))[0]
                    skio.imsave(os.path.join(mask_folder, label_stump + "_0.tif"), mask)
            else:
                # only run connected comp if there is at least one instance
                for blob_val in blob_vals[blob_vals != 0]:
                    labels_copy = blob_labels.copy()
                    labels_copy[blob_labels != blob_val] = 0
                    labels_copy[blob_labels == blob_val] = 1

                    label_stump = os.path.splitext(os.path.basename(label_chip))[0]
                    label_name = label_stump + "_" + str(blob_val) + ".tif"
                    mask_path = os.path.join(self.train_dir, label_chip[:-11], "mask")
                    label_path = os.path.join(mask_path, label_name)
                    assert labels_copy.ndim == 2
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        skio.imsave(label_path, labels_copy)
            
    def train_test_split():
        """Takes a sample of folder ids and copies them to a test directory
        from a directory with all folder ids. Each sample folder contains an 
        images and corresponding masks folder."""

        sample_list = next(os.walk(self.train_dir))[1]
        k = round(self.split * len(sample_list))
        test_list = random.sample(sample_list, k)
        for test_sample in test_list:
            shutil.copytree(
                os.path.join(self.train_dir, test_sample), os.path.join(self.test_dir, test_sample)
            )
            shutil.rmtree(
                os.path.join(self.train_dir, test_sample)
            )
        train_list = list(set(next(os.walk(self.train_dir))[1]) - set(next(os.walk(self.test_dir))[1]))
        train_df = pd.DataFrame({"train": train_list})
        test_df = pd.DataFrame({"test": test_list})
        train_df.to_csv(os.path.join(self.results_dir, "train_ids.csv"))
        test_df.to_csv(os.path.join(self.results_dir, "test_ids.csv"))

    def get_arr_channel_mean(channel):
        """
        Calculate the mean of a given channel across all training samples.
        """

        means = []
        train_list = next(os.walk(self.train_dir))[1]
        for i, fid in enumerate(train_list):
            im_folder = os.path.join(self.train_dir, fid, "image")
            im_path = os.path.join(im_folder, os.listdir(im_folder)[0])
            arr = skio.imread(im_path)
            arr = arr.astype(np.float32, copy=False)
            # added because no data values different for wv2 and landsat, need to exclude from mean
            nodata_value = arr.min() if arr.min() < 0 else -9999
            arr[arr == nodata_value] = np.nan
            means.append(np.nanmean(arr[:, :, channel]))
        print(np.mean(means))
