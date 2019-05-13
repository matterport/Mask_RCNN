import numpy
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import os.path
import json

# dataDir = '/Users/orshemesh/Desktop/Project/augmented_leaves/origin/'
# annFile = dataDir + 'leaves.json'
# output_dir = dataDir + 'output_phase1/'

def create_categories_map(annFile):
    with open(annFile) as f:
        orig_json = json.load(f)
    categories_num = len(orig_json['categories'])
    colors_list = numpy.array_split(numpy.array(range(256)), categories_num)
    categories_map = {}
    for i,category in enumerate(orig_json['categories']):
        categories_map[category['id']] = {
            'orig_json': category,
            'colors': list(colors_list[i])
        }
    return  categories_map


# create a folder with all the masks as multicolor pngs
def augment_create_mask_files(dataset_dir_path, dataset_annotation_file_path, output_dir):
    dataDir = dataset_dir_path # e.g: '/Users/orshemesh/Desktop/Project/augmented_leaves/origin/'
    annFile = dataset_annotation_file_path # e.g: '/Users/orshemesh/Desktop/Project/augmented_leaves/origin/leaves.json'
    output_dir = output_dir # e.g: '/Users/orshemesh/Desktop/Project/augmented_leaves/origin/output_phase1/'

    categories_map = create_categories_map(annFile)

    coco = COCO(annFile)

    categories = coco.loadCats(coco.getCatIds())
    categories_names = [category['name'] for category in categories]
    print('COCO categories: \n{}\n'.format(' '.join(categories_names)))

    for category in categories:
        cucumbers_Ids = coco.getCatIds(catNms=category['name'])
        images_Ids = coco.getImgIds(catIds=cucumbers_Ids )

        imgs = coco.loadImgs(images_Ids)
        image_num = 0

        for img in imgs:
            annotation_Ids = coco.getAnnIds(imgIds=img['id'], catIds=cucumbers_Ids, iscrowd=None)
            annotations = coco.loadAnns(annotation_Ids)

            # read image as RGB and add alpha (transparency)
            images_dir_path = dataDir
            image_name = img['file_name']
            im = Image.open(os.path.join(images_dir_path, image_name)).convert("RGBA")
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

                # choose a color
                color = numpy.random.choice(categories_map[category['id']]['colors'], size=3)
                color = numpy.append(color, [255])

                # while color exist choose another one
                while len([c for c in mask_colors if c[0]==color[0] and c[1]==color[1] and c[2]==color[2]]) != 0:
                    color = numpy.random.choice(categories_map[category['id']]['colors'], size=3)
                    color = numpy.append(color, [255])

                # add to colors that already used
                mask_colors.append(color)

                for segmentation in annotation['segmentation']:

                    polygon = []
                    for j in range(len(segmentation) // 2):
                        polygon.append((segmentation[2*j], segmentation[2*j+1]))

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


            try:
                newIm = Image.fromarray(newImArray, "RGBA")
                # newIm.show()
                newIm.save(os.path.join(output_dir, image_name).split('.')[0]+".png")
                image_num = image_num + 1
                print('{} out of {}\n path:{}'.format(image_num, len(imgs), os.path.join(output_dir, image_name).split('.')[0]+".png"))
            except Exception as e:
                print(e)
                print("Unexpected error: image {}".format(image_name.split('.')[0]+".png"))
    return categories_map

