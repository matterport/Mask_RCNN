source ~/.bashrc
/data/anaconda/envs/py35/bin/conda env create -f ~/work/$1/cropmask-env.yml
/data/anaconda/envs/cropmask/bin/python -m ipykernel install --user --name cropmask
cd ~/work/$1/
/data/anaconda/envs/cropmask/bin/python setup.py develop
