import os
from lsru import Usgs
from lsru import Espa
import yaml
from cropmask.download import download_utils as du
import datetime
import time
import pytest


@pytest.fixture
def configs():
    config_path = "/home/rave/azure_configs.yaml"

    with open(config_path) as f:
        configs = yaml.safe_load(f)
    return configs


@pytest.fixture
def setup_order(configs):
    """Orders a single scene from path/row for testing, or uses existing order"""

    usgs = Usgs(conf=configs["download"]["lsru_config"])
    usgs.login()
    espa = Espa(conf=configs["download"]["lsru_config"])

    try:
        order = espa.orders[-1]
        order.urls_completed = order.urls_completed[-1]  # speeds up testing
    except:
        bbox = [-102.255, 40.76, -101.255, 41.76]

        scene_list = du.get_scene_list(
            collection="LANDSAT_TM_C1",
            bbox=bbox,
            begin=datetime.datetime(2005, 1, 1),
            end=datetime.datetime(2006, 1, 1),
            max_results=10,
            max_cloud_cover=10,
        )

        pathrow_list_western_nb = ["032031"]

        scene_list = du.filter_scenes_by_path_row(scene_list, pathrow_list_western_nb)

        product_list = ["sr"]

        order = du.submit_order(scene_list, product_list)

    return order


def test_order_complete(setup_order):
    while setup_order.is_complete == False:
        time.sleep(10)
    assert setup_order.status == 200


def test_url_retrieve(setup_order):
    url = setup_order.urls_completed[-1]
    r = url_retrieve(url)
    assert r.status == 200


def test_azure_download(setup_order, configs):
    from azure.storage.blob import BlockBlobService as blob

    # if there is an existing order, we just want to use that and not set up a test order.
    setup_order.urls_completed = setup_order.urls_completed[-1]  # speeds up testing
    setup_order.download_all_complete_azure(
        configs["storage"]["container"],
        configs["storage"]["storage_name"],
        configs["storage"]["storage_key"],
    )

    url = setup_order.urls_completed[-1]
    blob_name = url.split("/")[-1].split(".")[0]

    assert blob.exists(
        container=configs["storage"]["storage_name"], blob_name=blob_name
    )
