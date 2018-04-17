"""
Downloads COCO 2017 Dataset
and builds directory structure

written by: www.github.com/GustavZ
"""

import os
import sys
from zipfile import ZipFile
import six.moves.urllib as urllib


def download_dataset(datasets,url,dest_dir):
    for dataset in datasets:
        dataset_file = dataset + '.zip'
        dataset_dir = os.path.join(dest_dir, dataset)
        if not os.path.isdir(dataset_dir):
            print ("> Dataset {} not found. downloading it".format(dataset))
            opener = urllib.request.URLopener()
            opener.retrieve(url + dataset_file, dataset_file)
            zipfile = ZipFile(dataset_file)
            zipfile.extractall(dest_dir)
            zipfile.close()
            os.remove(dataset_file)
        else:
            print('> Dataset {} Found. Proceed.'.format(dataset)


if __name__ == '__main__':
    ROOT_DIR = os.getcwd()
    COCO_DIR = os.path.join(ROOT_DIR,'data/coco')

    if not os.path.isdir(COCO_DIR):
        os.makedirs(COCO_DIR)

    dataset_list = [['train2017','val2017','test2017'],['annotations_trainval2017','image_info_test2017']]
    url_list = ['http://images.cocodataset.org/zips/','http://images.cocodataset.org/annotations/']

    print ("> Downloading COCO 2017 Datasets (~27GB)")
    for dataset,url in zip(dataset_list,url_list):
        download_dataset(dataset,url,COCO_DIR)
    print ("> Finished Downloading COCO 2017")
