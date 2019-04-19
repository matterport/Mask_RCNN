from cropmask.download import download_utils as du
import datetime
import yaml
import time
import click


@click.command()
@click.argument("config_path")
def run(config_path):
    """
    Runs the download process for azure. Reads the config file in order
    to subset the order by geographic bounds, date, or path row.

    Args:
        config_path (str): Takes the path to the config file, which contains 
        credentials for azure, storage paths, and other info (see template).

    Returns: Nothing is returned. Used for its side effect of downloading to azure.

    """

    with open(config_path) as f:
        configs = yaml.safe_load(f)

    bbox = du.get_bbox_from_wbd(
        configs["download"]["huc_level"], configs["download"]["huc_id"]
    )

    scene_list = du.get_scene_list(
        collection=configs["download"]["collection"],
        bbox=bbox,
        begin=datetime.datetime(
            configs["download"]["year_start"],
            configs["download"]["month_start"],
            configs["download"]["day_start"],
        ),
        end=datetime.datetime(
            configs["download"]["year_end"],
            configs["download"]["month_end"],
            configs["download"]["day_end"],
        ),
        max_results=300,
        max_cloud_cover=10,
    )

    pathrow_list_western_nb = configs["download"]["path_row_list"]

    scene_list = du.filter_scenes_by_path_row(scene_list, pathrow_list_western_nb)

    product_list = configs["download"]["product_list"]

    order = du.submit_order(scene_list, product_list)
    print("Order status: " + order.status)
    print("Order ID: " + order.orderid)
    du.azure_download_order(order)
    print("Order status: " + order.status)
    print("Order ID: " + order.orderid)


run()
