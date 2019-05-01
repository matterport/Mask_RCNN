import cropmask.preprocess as pp
from cropmask.misc import make_dirs, remove_dirs
import os
import pytest

@pytest.fixture
def wflow():
    wflow = pp.PreprocessWorkflow("/az-ml-container/configs/preprocess_config_pytest.yaml", 
                                 "/az-ml-container/western_nebraska_landsat_scenes_pytest/LT050320312005011601T1-SC20190418222311/",
                                 "/az-ml-container/external_pytest/nebraska-center-pivots-2005/nbextent-clipped-to-western.geojson")
    return wflow

def test_init(wflow):
    
    assert wflow
    
def test_make_dir():
    
    directory_list = ["/az-ml-container/pytest_dir2"]
    make_dirs(directory_list)
    try: 
        assert os.path.exists(directory_list[0])
    except AssertionError: 
        remove_dirs(directory_list)
        print("The directory was not created.")
    remove_dirs(directory_list)

def test_make_dirs(wflow):
    
    directory_list = wflow.setup_dirs()
    
    for i in directory_list:
        try: 
            assert os.path.exists(i)
        except AssertionError:
            remove_dirs(directory_list)
            print("The directory "+i+" was not created.")
    
    remove_dirs(directory_list)
    
def test_yaml_to_band_index(wflow):

    band_list = wflow.yaml_to_band_index()
    try: 
        assert band_list == ['1','2','3']
    except AssertionError:
        print("The band list "+band_list+" is not "+['1','2','3'])
        
def test_list_products():
    
    path = "/az-ml-container/western_nebraska_landsat_scenes_pytest/LT050320312005011601T1-SC20190418222311/"
    
    try: 
        product_list = os.listdir(path)
        assert product_list
    except AssertionError:
        print("The product list is empty, check this path: "+ path)
    
def test_get_product_paths(wflow):
   
    band_list = wflow.yaml_to_band_index()
    
    product_list = wflow.get_product_paths(band_list)
    
    assert product_list
    assert len(product_list) == len(band_list)
    
def test_load_and_stack_bands(wflow):
    # fails because product list empty
   
    band_list = wflow.yaml_to_band_index()
    
    product_list = wflow.get_product_paths(band_list)
    
    stacked_arr = wflow.load_and_stack_bands(product_list)
    
    assert stacked_arr.shape[-1] == len(product_list)
    
def test_stack_and_save_bands(wflow):
    
    directory_list = wflow.setup_dirs()
    
    band_list = wflow.yaml_to_band_index()
    
    product_list = wflow.get_product_paths(band_list)
    
    stacked_arr = wflow.load_and_stack_bands(product_list)
    
    try: 
        wflow.stack_and_save_bands()
    except:
        remove_dirs(directory_list)
        print("The function didn't complete.")
    
    try: 
        assert os.path.exists(wflow.stacked_path)
        remove_dirs(directory_list)
    except AssertionError:
        remove_dirs(directory_list)
        print("The stacked tif was not saved at the location "+wflow.stacked_path)

def test_negative_buffer_and_small_filter(wflow):
    
    directory_list = wflow.setup_dirs()
    
    band_list = wflow.yaml_to_band_index()
    
    product_list = wflow.get_product_paths(band_list)
    
    stacked_arr = wflow.load_and_stack_bands(product_list)
    
    wflow.stack_and_save_bands()
    
    try: 
        wflow.negative_buffer_and_small_filter(-31, 100)
    except:
        remove_dirs(directory_list)
        print("The function didn't complete.")
    
    try: 
        assert os.path.exists(wflow.rasterized_label_path)
        remove_dirs(directory_list)
    except AssertionError:
        remove_dirs(directory_list)
        print("The rasterized label tif was not saved at the location "+wflow.rasterized_label_path)
        
def test_grid_images(wflow):
    
    directory_list = wflow.setup_dirs()
    
    band_list = wflow.yaml_to_band_index()
    
    product_list = wflow.get_product_paths(band_list)
    
    stacked_arr = wflow.load_and_stack_bands(product_list)
    
    wflow.stack_and_save_bands()
    
    wflow.negative_buffer_and_small_filter(-31, 100)
    try: 
        wflow.grid_images()
        assert len(os.listdir(wflow.GRIDDED_IMGS)) > 1
    except AssertionError:
        remove_dirs(directory_list)
        print("Less than one chip directory was made") 
        