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
    
    
    
    
    
