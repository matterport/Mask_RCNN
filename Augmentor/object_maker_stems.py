

import numpy
from PIL import Image, ImageDraw
from pycocotools.coco import COCO


dataDir = '/Users/orshemesh/Desktop/Project/Photos/cucumber_stems/images/'
annFile = dataDir + 'coco_format_annotations.json'
coco = COCO(annFile)

cucumbers = coco.loadCats(coco.getCatIds())
categories_names = [cucumber['name'] for cucumber in cucumbers]
print('COCO categories: \n{}\n'.format(' '.join(categories_names)))

cucumbers_Ids = coco.getCatIds(catNms=categories_names[0])
images_Ids = coco.getImgIds(catIds=cucumbers_Ids )

imgs = coco.loadImgs(images_Ids)

for img in imgs:
    annotation_Ids = coco.getAnnIds(imgIds=img['id'], catIds=cucumbers_Ids, iscrowd=None)
    for i in range(len(annotation_Ids)):
        annotation = coco.loadAnns(annotation_Ids[i])
        polygon = []

        for j in range(len(annotation[0]['segmentation'][0]) // 2):
            polygon.append((annotation[0]['segmentation'][0][2*j], annotation[0]['segmentation'][0][2*j+1]))

        # read image as RGB and add alpha (transparency)
        images_dir_path = dataDir
        image_name = img['file_name']
        im = Image.open(images_dir_path+image_name).convert("RGBA")

        # convert to numpy (for convenience)
        imArray = numpy.asarray(im)

        # create mask
        maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
        ImageDraw.Draw(maskIm).polygon(polygon, fill=1, outline=0)
        mask = numpy.array(maskIm)

        dx = annotation[0]['bbox'][2]
        dy = annotation[0]['bbox'][3]
        x1 = annotation[0]['bbox'][0]
        y1 = annotation[0]['bbox'][1]
        x2 = x1 + dx
        y2 = y1 + dy

        # assemble new image (uint8: 0-255)
        shape = imArray.shape
        # newImArray = numpy.empty(imArray.shape, dtype='uint8')
        newImArray = numpy.empty((dy, dx, 4), dtype='uint8')

        # colors (three first columns, RGB)
        # newImArray[:, :, :3] = imArray[:, :, :3]
        newImArray[:, :, :3] = imArray[y1:y2, x1:x2, :3]

        # transparency (4th column)
        # newImArray[:, :, 3] = mask * 255
        newImArray[:, :, 3] = mask[y1:y2, x1:x2] * 255


        # objectImArray = numpy.empty((dy, dx, 4), dtype='uint8')
        # objectImArray[:, :, :] = newImArray[y1:y2, x1:x2, :]
        # back to Image from numpy
        # newIm = Image.fromarray(objectImArray, "RGBA")
        try:
            newIm = Image.fromarray(newImArray, "RGBA")
            if annotation[0]['area'] >= 10000 and (annotation[0]['bbox'][2]/annotation[0]['bbox'][3] < 0.25 or annotation[0]['bbox'][2]/annotation[0]['bbox'][3] > 4):
                newIm.save('/Users/orshemesh/Desktop/Project/Photos/cucumber_stems/objects/' + image_name.split('.')[0]+"_"+str(i)+".png")
            elif (annotation[0]['area'] < 10000):
                newIm.save('/Users/orshemesh/Desktop/Project/Photos/cucumber_stems/objects/small_' + image_name.split('.')[0]+"_"+str(i)+".png")
            else:
                newIm.save('/Users/orshemesh/Desktop/Project/Photos/cucumber_stems/objects/badRatio_' + image_name.split('.')[0]+"_"+str(i)+".png")
        except:
            print("Unexpected error: image {} annotation id {}".format(image_name.split('.')[0]+"_"+str(i)+".png", annotation[0]['id']))



