from cropmask.preprocess import *
import time

start = time.time()

wflow = PreprocessWorkflow("/home/ryan/work/CropMask_RCNN/cropmask/preprocess_config.yaml", 
                                 "/mnt/azureml-filestore-896933ab-f4fd-42b2-a154-0abb35dfb0b0/unpacked_landsat_downloads/032031/LT050320312005082801T1-SC20190418222350/",
                                 "/mnt/azureml-filestore-896933ab-f4fd-42b2-a154-0abb35dfb0b0/external/nebraska_pivots_projected.geojson")

if __name__ == "__main__":
    
    wflow.setup_dirs()
    
    band_list = wflow.yaml_to_band_index()
    
    product_list = wflow.get_product_paths(band_list)
    
    wflow.load_and_stack_bands(product_list)
    
    wflow.stack_and_save_bands()
    
    wflow.negative_buffer_and_small_filter(-31, 100)
    
    img_paths, label_paths = wflow.grid_images()
    
    wflow.remove_from_gridded(img_paths, label_paths)
    
    wflow.move_chips_to_folder()
    
    wflow.connected_components()
    
    wflow.train_test_split()
    
    print("channel means, put these in model_configs.py subclass")
    for i in band_list:
        print("Band index {} mean for normalization: ".format(i), get_arr_channel_mean(int(i)-1))
              
    print("preprocessing complete, ready to run model.")
    
    stop = time.time()
    
    print(stop-start, " seconds")