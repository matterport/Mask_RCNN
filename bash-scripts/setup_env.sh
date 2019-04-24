conda env create -f /home/rave/work/CropMaskRCNN/cropmask-env.yml
source activate cropmask
python -m ipykernel install --user --name cropmask
cd work/CropMask_RCNN
python setup.py develop
