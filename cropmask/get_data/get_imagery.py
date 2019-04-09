import shapely as shp
import geojson
import geopandas as gpd
import os
from lsru import Usgs
from lsru import Espa
from pprint import pprint
import datetime
import time
import yaml

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

def get_bbox_from_wbd(wbd_national_path, huc_layer_level=8, huc_id==None, name==None):
    """
    Gets bbox from National WBD Dataset, which can be downloaded from
    http://prd-tnm.s3-website-us-west-2.amazonaws.com/?prefix=StagedProducts/Hydrography/WBD/National/GDB/

    Reads in and parses the geodatabase with fiona and geopandas and
    subsets to the watershed boundary of interest. HUC8 is probably a good 
    size for Landsat tiles. Possible LayerIDs include:
    'WBDHU2', 'WBDHU4', 'WBDHU6', 'WBDHU8', 'WBDHU10', 'WBDHU12', 'WBDHU14', 'WBDHU16',

    Once the geodataframe has been read in based on a HUC level, it can be subset by any of the 
    values in the following columns (tested with HUC8):
    'NAME', 'GLOBALID', 'HUC8' (this is an integer), 'STATES' (can contain multiple two letter codes),
    or geometry. Right now, only filtering by name is implemented.
    
    Can only be filtered on name or huc_id not both. 

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
    layer_id = "WBDHU"+str(huc_layer_level)
    WBD = gpd.read_file(wbd_national_path, driver='FileGDB', layer=layer_id)
    if name != None:
        bbox = list(WBD[WBD['NAME']==name].envelope.boundary.bounds.iloc[0])
    if huc_id != None:
        huc_level = "HUC" + str(huc_layer_level)
        bbox = list(WBD[WBD[huc_level]==str(huc_id)].envelope.boundary.bounds.iloc[0])
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
    scene_list = usgs.search(collection=collection,
                    bbox=bbox,
                    begin=begin,
                    end=end,
                    max_results=max_results,
                    max_cloud_cover=max_cloud_cover)

    # Extract Landsat scene ids for each hit from the metadata
    scene_list = [x['displayId'] for x in scene_list]
    print(espa.get_available_products(filtered))
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

    filtered = [scene for scene in scene_list if any(good_pathrow \
        in scene for good_pathrow in path_row_list)]
    espa = Espa()
    print(espa.get_available_products(filtered))
    return filtered

def submit_order(filtered_scene_list, product_list):
    """
    Submits order to espa and returns the order.
    """

    order = espa.order(scene_list=filtered_scene_list, products=product_list)
    return order

def download_order(order):
    """
    Checks the status of an order made with submit_order until it is complete, then
    downloads the order. Waits for 5 minutes if the order is not ready yet.
    """
    for order in espa.orders:
        while order.is_complete==False:
        # Orders have their own class with attributes and methods
            print('%s: %s' % (order.orderid, order.status))
            time.sleep(300)
        


if __name__ == "__main__":

    with open('../../azure_configs.yaml') as f:
    configs = yaml.safe_load(f)

    DATA_DIR = configs['storage']['vm_temp_path']
    GEOJSON_DIR = configs['storage']['geojsons_path']
    LANDSAT_DIR = configs['storage']['landsat_path']
    LANDSAT_PREFIX = configs['storage']['l5_prefix']
    WBD_PATH = os.path.join(GEOJSON_DIR, "WBD_National_GDB.gdb")

    blob_service = BlockBlobService(configs['storage']['account_name'], 
                                configs['storage']['account_key'])

    # Instantiate Usgs class and login. requires setting config with credentials
    usgs = Usgs()
    usgs.login()

    # Parse the WBD Boundary dataset to get bbox for a single watershed
    # Pick a watershed using https://water.usgs.gov/wsc/map_index.html
    # the number of digits in the HUC ID is the second argument to the func
    bbox = get_bbox_from_wbd(WBD_PATH, 8, 15050202)

    # Query the Usgs api to find scene intersecting with the spatio-temporal window
    # help(usgs.search)
    scene_list = get_scene_list(collection='LANDSAT_8_C1',
                         bbox=bbox,
                         begin=datetime.datetime(2005,1,1),
                         end=datetime.datetime(2006,1,1),
                         max_results=300,
                         max_cloud_cover=10)  



