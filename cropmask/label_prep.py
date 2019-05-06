import geopandas as gpd
import rasterio
from shapely.geometry import Polygon

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