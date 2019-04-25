source ~/.bashrc
/data/anaconda/envs/py35/bin/conda env create -f ~/work/CropMask_RCNN-2/cropmask-env.yml
source activate cropmask
python -m ipykernel install --user --name cropmask
cd ~/work/Cropmask_RCNN-2/
python setup.py develop
