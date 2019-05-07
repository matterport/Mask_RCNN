import geopandas as gpd
import rasterio
from shapely.geometry import Polygon
import skimage.io as skio
import numpy as np
import os
from skimage import measure

def get_epsg(tif_path):
    "Gets the epsg code of the tif."
    with rasterio.open(tif_path) as src:
        meta = src.meta.copy()
        epsg_dict = meta['crs'].to_dict()
    return epsg_dict

def rio_bbox_to_polygon(tif_bounds):
    """Converts rasterio Bounding Box to a shapely geometry. access rasterio bounding box with
    rasterio.open and then src.bounds"""
    return Polygon([[tif_bounds.left, tif_bounds.bottom],[tif_bounds.left, tif_bounds.top],
    [tif_bounds.right,tif_bounds.top],[tif_bounds.right,tif_bounds.bottom]])

def connected_components(arr):
    """
    Extracts individual instances into their own tif files. Saves them
    in each folder ID in train folder. If an image has no instances,
    saves it with a empty mask. In this function geometry info is discarded, need to address.
    """
    
    unique_vals = np.unique(arr)
    # for imgs with no instances, create empty mask
    if len(unique_vals) == 1:
        return np.zeros_like(arr)
    else:
        # only run connected comp if there is at least one instance
        blob_labels = measure.label(arr, background=0)
        return blob_labels
    
def extract_labels(blob_labels):
    """
    Takes the output of connected_componenets and returns a list of
    arrays where each array is 1 where the instance is present and 0
    where it is not.
    """
    blob_vals = np.unique(blob_labels)
    label_list = []
    for blob_val in blob_vals[blob_vals != 0]:
        labels_copy = blob_labels.copy()
        labels_copy[blob_labels != blob_val] = 0
        labels_copy[blob_labels == blob_val] = 1
        label_list.append(labels_copy)
    return label_list