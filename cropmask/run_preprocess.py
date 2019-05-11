from cropmask.preprocess import *
import time

wflow = PreprocessWorkflow("/home/ryan/work/CropMask_RCNN/cropmask/preprocess_config.yaml", 
                                 "/mnt/azureml-filestore-896933ab-f4fd-42b2-a154-0abb35dfb0b0/unpacked_landsat_downloads/032031/LT050320312005082801T1-SC20190418222350/",
                                 "/mnt/azureml-filestore-896933ab-f4fd-42b2-a154-0abb35dfb0b0/external/nebraska_pivots_projected.geojson")

if __name__ == "__main__":
    
    wflow.run_single_scene()