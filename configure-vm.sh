# installs cropmask dependencies, including Mask R-CNN and geospatial libraries
# you must sync this file to the cluster with make syncup and run from home directory

sudo add-apt-repository ppa:ubuntugis/ppa
sudo apt-get update
sudo apt-get install -y python-numpy gdal-bin libgdal-dev
source activate py27
pip install --upgrade python=2.7.15
pip install -y rasterio shapely
pip install -y opencv-python imgaug lsru planet porder geojson
conda env update -y -n py27 -f requirements.txt

source deactivate
source activate py36
pip install -y rasterio
pip install -y opencv-python imgaug lsru planet porder geojson
conda env update -y -n py36 -f requirements.txt
cd work/cropmask
python setup.y install
