from PIL import Image
import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)
import json
import cv2

def create_sub_masks(mask_image):
    width, height = mask_image.size
    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    pixel = mask_image.getpixel((87, 58))
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
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        try:
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            segmentations.append(segmentation)
        except:
            print(poly)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
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

path = "/Users/orshemesh/Desktop/Project/Leaves_augmentor/leaves_after_phase2/leaves_after_phase1/"
# img1 = Image.open(path+'leaves_after_phase1_original_IMG_9205.png').convert("RGBA")
# img2 = Image.open(path+'leaves_after_phase1_original_IMG_9166.png').convert("RGBA")
img1 = Image.open('/Users/orshemesh/Desktop/Project/leaves_after_phase1/IMG_9097.png')
img2 = Image.open('/Users/orshemesh/Desktop/Project/leaves_after_phase1/IMG_9159.png')


imgs = [img1, img2]

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
annotation_id = 1
image_id = 1

# Create the annotations
annotations = []
for mask_image in imgs:
    sub_masks = create_sub_masks(mask_image)
    for color, sub_mask in sub_masks.items():
        # category_id = category_ids[image_id][color]
        category_id = 2
        annotation = create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd)
        annotations.append(annotation)
        annotation_id += 1
    image_id += 1

print(json.dumps(annotations))
