source ~/.bashrc
conda update -n base -c defaults conda -y
/data/anaconda/envs/py36/bin/conda env update -f ~/work/$1/cropmask-env.yml -y
cd ~/work/$1/
/data/anaconda/envs/py36/bin/python setup.py develop
/data/anaconda/envs/py36/bin/pip uninstall tensorflow
/data/anaconda/envs/py36/bin/conda install tensorflow-gpu
/data/anaconda/envs/py36/bin/conda install tensorflow
