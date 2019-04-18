from cropmask.download import download_utils as du
import datetime
import yaml
import time

config_path = "/home/ryan/work/azure_configs.yaml"

with open(config_path) as f:
    configs = yaml.safe_load(f)
    

bbox = du.get_bbox_from_wbd(configs['download']['huc_level'], configs['download']['huc_id'])

scene_list = du.get_scene_list(collection=configs['download']['collection'],
                     bbox=bbox,
                     begin=datetime.datetime(configs['download']['year_start'],configs['download']['month_start'],configs['download']['day_start']),
                     end=datetime.datetime(configs['download']['year_end'],configs['download']['month_end'],configs['download']['day_end']),
                     max_results=300,
                     max_cloud_cover=10) 

pathrow_list_western_nb = configs['download']['path_row_list']

scene_list = du.filter_scenes_by_path_row(scene_list, pathrow_list_western_nb)

product_list = configs['download']['product_list']

order = du.submit_order(scene_list, product_list)

du.azure_download_order(order)
#du.download_order()