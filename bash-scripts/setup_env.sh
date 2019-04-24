source ~/.bashrc
conda env create -f ../cropmask-env.yml
source activate cropmask
python -m ipykernel install --user --name cropmask
cd ../
python setup.py develop
