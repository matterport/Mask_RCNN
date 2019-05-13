import shapely as shp
import geojson
import geopandas as gpd
import os
from lsru import Usgs
from lsru import Espa
from pprint import pprint
import datetime
import time

# Instantiate Usgs class and login. requires setting config with credentials in home dir
login_path = os.path.expanduser("~/.lsru")
usgs = Usgs(conf=login_path)
usgs.login()
espa = Espa(conf=login_path)


def get_bbox_from_geojson(path):
    """
    Reads in a geojson, takes envelope,
    and returns the bounds. Might need to
    add in logic to support differently
    formatted geojsons (like multipolygons).

    Args:
        path (str): the path to the geojson file

    Returns: 
        List containing bbox coordinates
    """

    gdf = gpd.io.file.read_file(path)
    bbox = list(watershed.envelope.boundary.bounds.iloc[0])
    return bbox


def get_bbox_from_wbd(wbd_national_path, huc_layer_level, huc_id, name=None):
    """
    Gets bbox from National WBD Dataset, which can be downloaded from
    http://prd-tnm.s3-website-us-west-2.amazonaws.com/?prefix=StagedProducts/Hydrography/WBD/National/GDB/
    
    Find your huc with https://water.usgs.gov/wsc/map_index.html

    Reads in and parses the geodatabase with fiona and geopandas and
    subsets to the watershed boundary of interest. Possible LayerIDs include:
    'WBDHU2', 'WBDHU4', 'WBDHU6', 'WBDHU8', 'WBDHU10', 'WBDHU12', 'WBDHU14', 'WBDHU16',

    Once the geodataframe has been read in based on a HUC level, it can be subset by any of the 
    values in the following columns (tested with HUC8):
    'NAME', 'GLOBALID', 'HUC8' (this is an integer), 'STATES' (can contain multiple two letter codes),
    or geometry. You can filter by either NAME or HUC (not both).

    Args:
        wbd_national_path (str): National WBD .gdb file for USA.
        huc_layer_level (int): the HUC level to filter on, defaults to '8' for 'HUC8'.
        huc_id (int): the HUC ID to subset on. Only matches within the HUC level. Number of 
            digits in the huc_id is the huc_layer_level.
        name (str): the name of the watershed to filter on with column "NAME"

    Returns:
        Returns a list of bbox coordinates.

    Raises:
        KeyError
    """
    layer_id = "WBDHU" + str(huc_layer_level)
    WBD = gpd.read_file(wbd_national_path, driver="FileGDB", layer=layer_id)
    if name != None:
        bbox = list(WBD[WBD["NAME"] == name].envelope.boundary.bounds.iloc[0])
    else:
        huc_level = "HUC" + str(huc_layer_level)
        bbox = list(WBD[WBD[huc_level] == str(huc_id)].envelope.boundary.bounds.iloc[0])
    return bbox


def get_scene_list(collection, bbox, begin, end, max_results, max_cloud_cover):
    """
    Uses EROS ESPA api via the lsru package to
    get the list of scene ids for the given param
    ranges.

    Args:
        collection (str): Collection ID str like "LANDSAT_8_C1".
        bbox (list): list of coordinates from get_bbox.
        begin (datetime): lkke datetime.datetime(2014,1,1)
        end (datetime):
        max_results (int):
        max_cloud_cover (int):

    Returns:
        Returns a list of scene strings

    Raises:
        KeyError: Raises an exception.
    """
    # Query the Usgs api to find scene intersecting with the spatio-temporal window
    # help(usgs.search)
    scene_list = usgs.search(
        collection=collection,
        bbox=bbox,
        begin=begin,
        end=end,
        max_results=max_results,
        max_cloud_cover=max_cloud_cover,
    )

    # Extract Landsat scene ids for each hit from the metadata
    scene_list = [x["displayId"] for x in scene_list]
    # TODO nested dict needs to be parsed a level and return value for 'products' key
    # print(espa.get_available_products(scene_list))
    return scene_list


def filter_scenes_by_path_row(scene_list, path_row_list):
    """
    Takes a scene list and list of path/rows and returns the 
    correct scenes. Prints the available products for the scenes

    Args:
        scene_list: list of scene strings from get_scene_list
        path_row_list: user supplied lis tof path/row strings like ['032031','033031']

    Returns:
        Returns a list of scene strings

    Raises:
        KeyError: Raises an exception.
    """

    filtered = [
        scene
        for scene in scene_list
        if any(good_pathrow in scene for good_pathrow in path_row_list)
    ]
    # print(espa.get_available_products(filtered))
    return filtered


def submit_order(filtered_scene_list, product_list):
    """
    Submits order to espa and returns the order.
    """

    order = espa.order(scene_list=filtered_scene_list, products=product_list)
    return order


def local_download_order(download_path):
    """
    Checks the status of an order made with submit_order until it is complete, then
    downloads the order. Waits for 5 minutes if the order is not ready yet. Checks to 
    see if it is already downloaded by file size and name.
    """
    for order in espa.orders:
        while order.is_complete == False:
            # Orders have their own class with attributes and methods
            print("%s: %s" % (order.orderid, order.status))
            time.sleep(300)
        order.download_all_complete(path=download_path, unpack=True)
        print("Order downloaded")


def azure_download_order(order, configs):
    while order.is_complete == False:
        time.sleep(600)
    order.download_all_complete_azure(
        configs["storage"]["container"],
        configs["storage"]["region_name"],
        configs["storage"]["storage_name"],
        configs["storage"]["storage_key"],
    )
    print("Finished downloading order to azure")
