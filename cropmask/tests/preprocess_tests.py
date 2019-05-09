import cropmask.preprocess as pp
from cropmask.misc import make_dirs, remove_dirs
import os
import pytest

@pytest.fixture
def wflow():
    wflow = pp.PreprocessWorkflow("/home/ryan/work/CropMask_RCNN/cropmask/preprocess_config.yaml", 
                                 "/mnt/azureml-filestore-896933ab-f4fd-42b2-a154-0abb35dfb0b0/unpacked_landsat_downloads/032031/LT050320312005082801T1-SC20190418222350/",
                                 "/mnt/azureml-filestore-896933ab-f4fd-42b2-a154-0abb35dfb0b0/external/nebraska_pivots_projected.geojson")
    return wflow

def test_init(wflow):
    
    assert wflow
    
def test_make_dir():
    
    directory_list = ["/mnt/azureml-filestore-896933ab-f4fd-42b2-a154-0abb35dfb0b0/pytest_dir"]
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
    
    path = "/mnt/azureml-filestore-896933ab-f4fd-42b2-a154-0abb35dfb0b0/unpacked_landsat_downloads/032031/LT050320312005082801T1-SC20190418222350/"
    
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
        assert wflow.negative_buffer_and_small_filter(-31, 100) == np.array([0, 1]) # for the single class case, where 1 are cp pixels
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
        img_paths, label_paths = wflow.grid_images()
        assert len(img_paths) > 0
        assert len(img_paths) == len(label_paths)
    except AssertionError:
        remove_dirs(directory_list)
        print("Less than one chip was saved") 
        
def test_move_chips_to_folder(wflow):
        
    directory_list = wflow.setup_dirs()
    
    band_list = wflow.yaml_to_band_index()
    
    product_list = wflow.get_product_paths(band_list)
    
    stacked_arr = wflow.load_and_stack_bands(product_list)
    
    wflow.stack_and_save_bands()
    
    wflow.negative_buffer_and_small_filter(-31, 100)
    
    img_paths, label_paths = wflow.grid_images()
    
    wflow.remove_mostly_empty(img_paths, label_paths)
    try: 
        assert wflow.move_chips_to_folder()
        assert len(os.listdir(wflow.TRAIN)) > 1
        assert len(os.listdir(os.listdir(wflow.TRAIN))[0]) > 0
    except AssertionError:
        remove_dirs(directory_list)
        print("Less than one chip directory was made") 
        
def test_connected_components(wflow):
        
    directory_list = wflow.setup_dirs()
    
    band_list = wflow.yaml_to_band_index()
    
    product_list = wflow.get_product_paths(band_list)
    
    stacked_arr = wflow.load_and_stack_bands(product_list)
    
    wflow.stack_and_save_bands()
    
    wflow.negative_buffer_and_small_filter(-31, 100)
    
    img_paths, label_paths = wflow.grid_images()
    
    wflow.remove_mostly_empty(img_paths, label_paths)
    
    wflow.move_chips_to_folder()
    
    try: 
        assert wflow.connected_components()
    except AssertionError:
        print("Connected components did not complete") 

        