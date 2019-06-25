source ~/.bashrc
conda update -n base -c defaults conda -y
conda env update -n py36 --file ~/work/$1/cropmask-env.yml
cd ~/work/$1/
/data/anaconda/envs/py36/bin/python setup.py develop
