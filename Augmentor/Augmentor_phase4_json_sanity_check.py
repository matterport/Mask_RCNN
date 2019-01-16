from pycocotools.coco import COCO
import os


dataDir='/Users/orshemesh/Desktop/Project/augmented_cucumbers/origin/output_phase2/'
annFile=dataDir+'new_annotations.json'

files_in_dir = os.listdir(dataDir)
files_in_dir = [file.split('/')[-1] for file in files_in_dir if file.find('.PNG') != -1 and file.find("ground") == -1]

# initialize COCO api for instance annotations
coco = COCO(annFile)

# display COCO categories
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['fruit'])
imgIds = coco.getImgIds(catIds=catIds )

imgs = coco.loadImgs(imgIds)

error_found = False

for img in imgs:
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    found = False
    for ann in anns:
        for sig in ann['segmentation']:
            if len(sig) > 0:
                list = [f for f in files_in_dir if f != img['file_name']]
                if len(list) < len(files_in_dir):
                    files_in_dir = list
                    found = True
        if found:
            break
    if found:
        continue
    else:
        print('no annotation for {}'.format(img['file_name']))
        error_found = True
        continue

if len(files_in_dir) > 0:
    print("images in the directory without reference in the json")
    print(files_in_dir)
    error_found = True

files_in_dir = os.listdir(dataDir)
files_in_dir = [file.split('/')[-1] for file in files_in_dir if file.find('.PNG') != -1 and file.find("ground") == -1]

for img in imgs:
    files_in_dir = [f for f in files_in_dir if f != img['file_name']]

if len(files_in_dir) > 0:
    print("images in the json without reference in the directory")
    print(files_in_dir)
    error_found = True

if not error_found:
    print('{} passed sanity check'.format(annFile))

