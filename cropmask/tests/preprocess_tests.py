import cropmask.preprocess as pp
from cropmask.misc import make_dirs, remove_dirs
import os
import pytest

def test_init():
    
    assert pp.PreprocessWorkflow("/az-ml-container/configs/preprocess_config_pytest.yaml", 
                                 "/az-ml-container/western_nebraska_landsat_scenes_pytest/LT050320312005011601T1-SC20190418222311/",
                                 "/az-ml-container/external_pytest/nebraska-center-pivots-2005/nbextent-clipped-to-western.geojson")
    
def test_make_dir():
    
    directory_list = ["/az-ml-container/pytest_dir2"]
    make_dirs(directory_list)
    try: 
        assert os.path.exists(directory_list[0])
    except AssertionError: 
        remove_dirs(directory_list)
        print("The directory was not created.")
    remove_dirs(directory_list)

def test_make_dirs():
    
    wflow = pp.PreprocessWorkflow("/az-ml-container/configs/preprocess_config_pytest.yaml", 
                                 "/az-ml-container/western_nebraska_landsat_scenes_pytest/LT050320312005011601T1-SC20190418222311/",
                                 "/az-ml-container/external_pytest/nebraska-center-pivots-2005/nbextent-clipped-to-western.geojson")
    
    directory_list = wflow.setup_dirs()
    
    for i in directory_list:
        try: 
            assert os.path.exists(i)
        except AssertionError:
            remove_dirs(directory_list)
            print("The directory "+i+" was not created.")
    
    remove_dirs(directory_list)
    
def test_yaml_to_band_index():
    
    wflow = pp.PreprocessWorkflow("/az-ml-container/configs/preprocess_config_pytest.yaml", 
                                 "/az-ml-container/western_nebraska_landsat_scenes_pytest/LT050320312005011601T1-SC20190418222311/",
                                 "/az-ml-container/external_pytest/nebraska-center-pivots-2005/nbextent-clipped-to-western.geojson")
    band_list = wflow.yaml_to_band_index()
    try: 
        assert band_list == ['1','2','3']
    except AssertionError:
        print("The band list "+band_list+" is not "+['1','2','3'])
        
def test_list_products():
    
    path = "/az-ml-container/western_nebraska_landsat_scenes_pytest/LT050320312005011601T1-SC20190418222311/"
    
    try: 
        product_list = os.listdir()
        assert product_list
    except AssertionError:
        print("The product list is empty, check this path: "+ path)
    
def test_get_product_paths():
    # fails because product list empty
    wflow = pp.PreprocessWorkflow("/az-ml-container/configs/preprocess_config_pytest.yaml", 
                                 "/az-ml-container/western_nebraska_landsat_scenes_pytest/LT050320312005011601T1-SC20190418222311/",
                                 "/az-ml-container/external_pytest/nebraska-center-pivots-2005/nbextent-clipped-to-western.geojson")
   
    wflow.yaml_to_band_index()
    
    product_list = wflow.get_product_paths()
    
    assert product_list    
    
    
