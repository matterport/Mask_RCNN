from PIL import Image
import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)
import json
import os
import datetime
import os
import glob

def create_sub_masks(mask_image):
    width, height = mask_image.size
    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            # pixel = mask_image.getpixel((x,y))

            pixel = mask_image.getpixel((x,y))[:3]
            A = mask_image.getpixel((x,y))[3]

            # If the pixel is not black...
            if A == 255 :
            # if pixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks

def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0)
        polygons.append(poly)
        try:
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            segmentations.append(segmentation)
        except:
            pass

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)

    if multi_poly.area < 150 or len(segmentations) == 0:
        return []

    print(multi_poly.area)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation

def get_image_json_doc(orig_json, gt_image_path, new_id):
    image_file_path = os.path.dirname(gt_image_path)
    gt_image_filename = os.path.split(gt_image_path)[1]

    # e.g: ground_truth_IMG_2055_22.PNG ==> IMG_2055_22
    core_name = gt_image_filename.split("ground_truth_")[-1].split(".")[0]

    # find image filename and discover its filetype
    image_file_path = glob.glob(os.path.join(image_file_path, core_name + "*"))[0]
    image_name = os.path.split(image_file_path)[-1]

    # original name in original json
    extension = image_name.split(".")[-1]
    core_name_parts = core_name.split('_')
    image_orig_name = "_".join(core_name_parts[:-1]) + "." + extension

    # get new width and height of image
    tmp = Image.open(os.path.join(image_file_path))
    width, height = tmp.size
    tmp.close()
    
    for img in orig_json['images']:
        if img['file_name'] == image_orig_name:

            if 'annotated' in img:
                annotated = img['annotated']
            else:
                annotated = False

            if 'metadata' in img:
                metadata = img['metadata']
            else:
                metadata = False

            if 'url' in img:
                url = img['url']
            else:
                url = ''

            result = {
                'date': str(datetime.datetime.now()),
                'file_name': image_name,
                'id': new_id,
                'path': image_file_path,
                'url': url,
                'width': width,
                'height': height,
                # 'dataset_id': dataset_id,
                'annotated': annotated,
                'metadata': metadata
            }
            return result
    return []

def get_category_id(categories_map, color):
    for category in categories_map:
        if color[0] in category['colors']:
            return category['id']
    print('Error: can not find appropriate category id')
    return -1

def augment_reconstruct_json(dir_path, orig_json_path, categories_map):
    # dir_path = '/Users/orshemesh/Desktop/Project/augmented_leaves/origin/output_phase2/'
    # dir_path: directory containing IMG... and groundtruth...
    # orig_json_path = '/Users/orshemesh/Desktop/Project/augmented_leaves/origin/leaves.json'

    files_in_dir = os.listdir(dir_path)
    ground_truth_images = [file for file in files_in_dir if file.find('ground_truth') != -1]
    # augmented_image_names = [file for file in files_in_dir if file.find('ground_truth') == -1]


    with open(orig_json_path) as f:
        orig_json = json.load(f)

    # new_image_jason = get_image_json_doc(orig_json, img1.filename.split('/')[-1].split('.')[0], 17)
    # new_image_jason2 = get_image_json_doc(orig_json, img2.filename.split('/')[-1], 18)

    # Define which colors match which categories in the images
    # houseplant_id, book_id, bottle_id, lamp_id = [1, 2, 3, 4]
    # category_ids = {
    #     1: {
    #         '(0, 255, 0)': houseplant_id,
    #         '(0, 0, 255)': book_id,
    #     },
    #     2: {
    #         '(255, 255, 0)': bottle_id,
    #         '(255, 0, 128)': book_id,
    #         '(255, 100, 0)': lamp_id,
    #     }
    # }

    is_crowd = 0

    # These ids will be automatically increased as we go
    annotation_id = 0
    image_id = 0
    # dataset_id = 2

    if 'info' in orig_json:
        info = orig_json['info']
    else:
        info = 'GeverNet project 2019!'
    categories = orig_json['categories']

    output = {
        'categories': categories,
        'images': [],
        'annotations': [],
        'info': info
    }

    # Create the annotations
    annotations = []
    images = []
    errors = []
    for file in ground_truth_images:
        mask_image = Image.open(os.path.join(dir_path, file))
        new_image_json_doc = get_image_json_doc(orig_json, os.path.join(dir_path, file), image_id)
        if new_image_json_doc == []:
            errors.append("{} file not found in original json".format(mask_image.filename.split('/')[-1]))
            print("{} : can't find augmented image or relevant image field in origin json!".format(mask_image.filename.split('/')[-1]))
            continue
        sub_masks = create_sub_masks(mask_image)
        mask_image.close()
        has_annotation = False
        for color, sub_mask in sub_masks.items():
            # category_id = category_ids[image_id][color]
            category_id = get_category_id(categories_map, color)
            annotation = create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd)
            if annotation != []:
                annotations.append(annotation)
                annotation_id += 1
                has_annotation = True

        if has_annotation:
            # change images size in json
            images.append(new_image_json_doc)
            print('{} Done! {} out of {}'.format(mask_image.filename, image_id + 1, len(ground_truth_images)))
        else:
            print('no annotation for {} {}'.format(mask_image.filename, image_id))
            errors.append('no annotation for {} {}'.format(mask_image.filename, image_id))
            continue

        image_id += 1
        output['images'] = images
        output['annotations'] = annotations

        if image_id % 100 == 0:
            with open(os.path.join(dir_path, 'new_annotations_' + str(image_id) + '.json'), 'w') as outfile:
                json.dump(output, outfile, indent=4, sort_keys=True)

    # print(json.dumps(output))
    with open(os.path.join(dir_path, 'annotations.json'), 'w') as outfile:
        json.dump(output, outfile, indent=4, sort_keys=True)

    print(errors)

# augment_reconstruct_json("/home/simon/Documents/cucu_dataset/real/1024/fruit/train/augmented", "/home/simon/Documents/cucu_dataset/real/1024/fruit/train/original/annotations.json")