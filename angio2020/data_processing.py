import cv2
import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from pycocotools import mask as maskUtils
from pathlib import Path
import json

'''
converts a given mask image to run length encoding format (rle)
rle itself is a dictionary containing:
    size: size of original binary map
    counts: the encoded binary map (bytes)

rle is shorter than raw binary map array

parameter threshold determines the minimum pixel value (ranges from 0 to 255) to be converted to 1 in bin map
if no img is given image will be loaded from given path
'''

def toBinMaskRle(img=None, path=None, threshold=10):
    if path:
        # read mask in grayscale format
        img = cv2.imread(path, 0)

    # create binary map by thresholding
    ret, binMap = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    #convert bin map to rle
    binMap = np.asfortranarray(binMap)
    rle = maskUtils.encode(binMap)
    rle['counts'] = str(rle['counts'], 'utf-8')
    return rle

def createJsonAnnotations(data_path, subset):
    # list of categories for the dataset
    categories = [{'id' : 1, 'name' : 'lad'}, {'id' : 2, 'name' : 'diagonal'}, {'id' : 3, 'name' : 'lcx1'}, {'id' : 4, 'name' : 'lcx2'}, {'id' : 5, 'name' : 'distal'}]
    category_to_id = {
        'lad' : 1,
        'diagonal' : 2,
        'lcx1' : 3,
        'lcx2' : 4,
        'distal' : 5
    }
    # list of annotations
    annotations = []
    #list of images
    images = []

    p = Path(data_path)
    id_cnt = 0
    # navigates the items found in data folder
    for item in p.iterdir():
        # if item is a folder containing masks
        if item.is_dir():
            # image_id is the folder name
            image_id = item.name
            # iterate through all the masks
            for f in item.iterdir():
                # get rle and add to list of annotations
                print(f.name)
                category =  f.name.split('_')[-1][:-4]
                if category in ['mask1', 'mask2', 'mask3', 'mask']:
                    continue
                category_id = category_to_id[category]
                id_cnt += 1
                rle = toBinMaskRle(path=str(f))
                area = int(maskUtils.area(rle))
                bbox = maskUtils.toBbox(rle).tolist()
                annotations.append({
                    'id': id_cnt,
                    'iscrowd': 0,
                    'segmentation': rle,
                    'area': area,
                    'bbox': bbox,
                    'category_id' : category_id,
                    'image_id' : image_id,
                })
            
        # item is a image
        else:
            image_id = item.name.split('.')[0]
            img = cv2.imread(str(item), 0)
            height = img.shape[0]
            width = img.shape[1]
            # add image info to json
            images.append({
                'filename': image_id + '.jpeg',
                'height':height,
                'width':width,
                'id': image_id
            })
            

    #create json
    jsonFile = {
        'categories' : categories,
        'annotations' : annotations,
        'images' : images
    }

    with open(data_path + '/data_{}.json'.format(subset), 'w', encoding='utf-8') as f:
        json.dump(jsonFile, f, ensure_ascii=False, indent=4)



createJsonAnnotations('A:/train', 'train')
createJsonAnnotations('A:/val', 'val')
createJsonAnnotations('A:/test', 'test')

# rle = jpgToBinMaskRle(image_path)
# print(rle)
