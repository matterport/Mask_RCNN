[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

CropMask is a project to train and deploy instance segmentation models for mapping center pivot agriculture from multispectral satellite imagery. It extends [matterport's module](https://github.com/matterport/Mask_RCNN) , which is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. CropMask work with multispectral satellite imagery, contains infrastructure-as-code via terraform to provision a testing cluster on Azure, and will eventually contain a Leaflet or OpenLayers web app to expose maps of crop water use in drylands across the globe. 

See [matterport's module](https://github.com/matterport/Mask_RCNN)for an explanation of the Mask R-CNN architecture and a general guide to notebook tutorials and notebooks for inspecting model inputs and outputs.

For an overview of the project in poster form, see this poster I presented at the Fall 2018 Meeting on [Center Pivot Crop Water Use](assets/cropmask_agu2018.pdf). 

Below are Preliminary results from test on 2004 Landsat SR scene over western Nebraska. Detections are in Red, Targets from the Nebraska Department of Agriculture are in Green. Metrics are (probability score)/(intersection over union)
![Center Pivot Detections](assets/cp_detection.png)

## Local Installation of matterport's maskrcnn, see `terraform/` folder for Azure instructions
1. Install dependencies
   ```bash
   conda env create -f geo-environment.yml
   ```
2. Clone this repository
3. Run setup from the repository root directory
    ```bash
    source activate cropmask
    python setup_cropmask.py install # for some reason this also installs mrcnn
    python setup_mrcnn.py install # but just in case
    ``` 
3. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).
4. (Optional) To train or test on MS COCO install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi.
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)

## For instructions on setting up the entire project on Azure, see the README in the terraform folder
