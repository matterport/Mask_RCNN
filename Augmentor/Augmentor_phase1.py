import numpy
from PIL import Image, ImageDraw
from pycocotools.coco import COCO


dataDir = '/Users/orshemesh/Desktop/Project/Photos/2018_05_31_09_02_segmentation_task_8_leaf_cucumber_KIM/'
annFile = '/Users/orshemesh/Desktop/Project/Photos/2018_05_31_09_02_segmentation_task_8_leaf_cucumber_KIM/segmentation_results.json'
coco = COCO(annFile)

cucumbers = coco.loadCats(coco.getCatIds())
categories_names = [cucumber['name'] for cucumber in cucumbers]
print('COCO categories: \n{}\n'.format(' '.join(categories_names)))

supercategories_names = set([cucumber['supercategory'] for cucumber in cucumbers])
print('COCO supercategories: \n{}'.format(' '.join(supercategories_names)))

cucumbers_Ids = coco.getCatIds(catNms=categories_names[0]);
images_Ids = coco.getImgIds(catIds=cucumbers_Ids );

imgs = coco.loadImgs(images_Ids)

for img in imgs:
    annotation_Ids = coco.getAnnIds(imgIds=img['id'], catIds=cucumbers_Ids, iscrowd=None)
    annotations = coco.loadAnns(annotation_Ids)

    # read image as RGB and add alpha (transparency)
    images_dir_path = dataDir
    image_name = img['file_name']
    im = Image.open(images_dir_path+image_name).convert("RGBA")
    # im.show()

    # convert to numpy (for convenience)
    imArray = numpy.asarray(im)

    # assemble new image (uint8: 0-255)
    shape = imArray.shape
    newImArray = numpy.zeros(imArray.shape, dtype='uint8')
    # newImArray = numpy.empty((dy, dx, 4), dtype='uint8')

    # colors (three first columns, RGB)
    newImArray[:, :,0:3] = imArray[:, :, 0:3]

    # for x in range(shape[0]):
    #     for y in range(shape[1]):
    #         newImArray[x][y][3] = 0
    # newImArray[:, :, :3] = imArray[y1:y2, x1:x2, :3]
    mask_colors = []
    for annotation in annotations:
        color = numpy.random.choice(range(256), size=3)
        color = numpy.append(color, [255])
        # while color in mask_colors:
        #     color = numpy.random.choice(range(256), size=3)
        #     color = numpy.append(color, [255])
        mask_colors.append(color)
        polygon = []
        for j in range(len(annotation['segmentation'][0]) // 2):
            polygon.append((annotation['segmentation'][0][2*j], annotation['segmentation'][0][2*j+1]))

        # create mask
        maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
        ImageDraw.Draw(maskIm).polygon(polygon, fill=1, outline=0)
        mask = numpy.array(maskIm)
        # transparency (4th column)
        # newImArray[:, :, 3] = mask * 255
        # numpy.arange()

        # for (x, y), _ in numpy.ndenumerate(mask):
        #     if mask[x][y] != 0:
        #         new_mask[x][y] = color

        rows = numpy.where(mask[:,:] != 0)
        for x, y in zip(rows[0], rows[1]):
            newImArray[x, y] = color


    # objectImArray = numpy.empty((dy, dx, 4), dtype='uint8')
    # objectImArray[:, :, :] = newImArray[y1:y2, x1:x2, :]
    # back to Image from numpy
    # newIm = Image.fromarray(objectImArray, "RGBA")
    try:
        newIm = Image.fromarray(newImArray, "RGBA")
        # newIm.show()
        newIm.save("/Users/orshemesh/Desktop/Project/leaves_after_phase1/"+image_name.split('.')[0]+".png")
    except:
        print("Unexpected error: image {}".format(image_name.split('.')[0]+".png"))


