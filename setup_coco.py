"""
Downloads COCO 2017 Dataset
and builds directory structure

written by: www.github.com/GustavZ
"""

import os
from mmrcnn import utils


ROOT_DIR = os.getcwd()
COCO_DIR = os.path.join(ROOT_DIR,'data/coco')

if not os.path.isdir(COCO_DIR):
    os.makedirs(COCO_DIR)

dataset_list = [['train2017','val2017','test2017'],['annotations_trainval2017','image_info_test2017']]
url_list = ['http://images.cocodataset.org/zips/','http://images.cocodataset.org/annotations/']

print ("> Downloading COCO 2017 Datasets (~27GB)")
for datasets,url in zip(dataset_list,url_list):
    for dataset in datasets:
        utils.download_zipfile(dataset,url,COCO_DIR)
print ("> Finished Downloading COCO 2017")
