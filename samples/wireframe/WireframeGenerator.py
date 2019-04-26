import os
from PIL import Image
import random


BACKGROUND_DIR = os.getcwd() + "/Backgrounds"
ICONS = os.listdir(os.getcwd() + "/Icons")
BACKGROUNDS = os.listdir(BACKGROUND_DIR)

# Remove annoying Mac files
if ".DS_Store" in ICONS:
    ICONS.remove(".DS_Store")
if ".DS_Store" in BACKGROUNDS:
    BACKGROUNDS.remove(".DS_Store")

# Define Dirs
MASK_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../"))
DATA_DIR = os.path.abspath(os.path.join(MASK_DIR, "datasets/wireframe"))

# Wire-frame dimensions (iPhone 8)
DIMENSIONS = (900, 1200)
bg_w, bg_h = DIMENSIONS
COLOR = (255, 255, 255, 255)

# Icon Dimensions
ICON_W = ICON_H = 50

#Specify training data or validation data
TYPES = ["/train", "/val"]


file_content = "{"


def write_string_to_json(string, type):
    """
    :param string: Content to write to file
    :param type: Train or validation dir
    """
    with open(DATA_DIR + type + '/via_region_data.json', 'w+') as f:
        f.write(string)


def add_box_json_polygon(filename, size, icons):
    """
    Helper function for the generate data function

    :param filename: Filename or image file
    :param size: Size of the image
    :param icons: List of icons in image
    :return:
    """
    s = " '{}{}':{{ 'filename': '{}', 'size': {}, 'regions': [".format(filename, size, filename, size)
    for icon in icons:
        x1, x2 = (icon[1], icon[1] + ICON_W)
        y1, y2 = (icon[2], icon[2] + ICON_H)
        s += "{{'shape_attributes': {{'name': 'polygon', " \
             "'all_points_x': [{},{},{},{}], 'all_points_y': [{},{},{},{}] }}," \
             "'region_attributes': {{'name': '{}'}} }} ,".format(x1, x1, x2, x2, y2, y1, y1, y2, icon[0])
    s = s[0:-1] + "]" + " },"
    s = s.replace("'", '"')
    return s


def generate_data(NUM_IMAGES, ICONS_PER_IMAGE):
    """
    Generates wireframes by inserting icons into backgrounds, and writing a "regions data" file
    that lets Mask R-CNN know where the BBoxes are on the image

    :param NUM_IMAGES: Number of images you want to generate
    :param ICONS_PER_IMAGE: MAX icons per image
    """
    global file_content
    for type in TYPES:
        file_content = "{"
        if type == "/val":
            NUM_IMAGES = int(NUM_IMAGES / 5)
        for j in range(NUM_IMAGES):
            icon_list = []
            NUM_ICONS = random.randint(1, ICONS_PER_IMAGE)
            background = Image.open(BACKGROUND_DIR + "/" + BACKGROUNDS[random.randint(0, len(BACKGROUNDS) - 1)]).convert("L")
            for i in range(NUM_ICONS):
                cur_icon = ICONS[random.randint(0, len(ICONS) - 1)]
                img = Image.open('Icons/' + cur_icon, 'r').resize((ICON_W, ICON_H))
                offset = random.randint(1, bg_w - ICON_W), random.randint(1, bg_h - ICON_H)
                background.paste(img, offset, img)
                icon_list.append((cur_icon[0:-4], offset[0], offset[1]))
            img_dir_name = DATA_DIR + type + '/' + str(j) + ".png"
            background.save(img_dir_name)
            filename = str(j) + ".png"
            img_size = os.stat(img_dir_name).st_size
            file_content += add_box_json_polygon(filename, img_size, icon_list)

        file_content = file_content[0:-1] + "}"
        write_string_to_json(file_content, type)
