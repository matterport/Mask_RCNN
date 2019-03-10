import os
from PIL import Image
import random

ICONS = os.listdir(os.getcwd() + "/Icons")
MASK_DIR = os.path.abspath(os.path.join(os.getcwd(),"../../"))
DATA_DIR = os.path.abspath(os.path.join(MASK_DIR, "datasets/wireframe"))

# Wire-frame dimensions (iPhone 8)
DIMENSIONS = (750, 1334)
bg_w, bg_h = DIMENSIONS
COLOR = (255, 255, 255, 255)

# Icon Dimensions
ICON_W = ICON_H = 50

#Specify training data or validation data
DATA_TYPE = "/train"
# DATA_TYPE = "/val"

file_content = "{"


def add_box_json(filename, size, icons):
    s = " '{}{}':{{ 'filename': '{}', 'size': {}, 'regions': [".format(filename, size, filename, size)
    for icon in icons:
        s += "{{'shape_attributes': {{'name': 'rect', 'x': {}, 'y': {}, 'width': 50, 'height': 50}}, " \
                "'region_attributes': {{'name': '{}'}} }} ,".format(icon[1], icon[2], icon[0])
    s = s[0:-1] + "]" + " },"
    s = s.replace("'", '"')
    return s

def add_box_json_polygon(filename, size, icons):
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


def write_string_to_json(string):
    with open(DATA_DIR + DATA_TYPE + '/via_region_data.json', 'w+') as f:
        f.write(string)


def generate_data(NUM_IMAGES=10, ICONS_PER_IMAGE=3):
    global file_content

    for j in range(NUM_IMAGES):
        icon_list = []
        NUM_ICONS = random.randint(1, ICONS_PER_IMAGE)

        background = Image.new('RGBA', DIMENSIONS, COLOR)
        for i in range(NUM_ICONS):
            cur_icon = ICONS[random.randint(0, len(ICONS) - 1)]
            img = Image.open('Icons/' + cur_icon, 'r').resize((ICON_W, ICON_H))
            offset = random.randint(1, bg_w - ICON_W), random.randint(1, bg_h - ICON_H)
            background.paste(img, offset)
            icon_list.append((cur_icon[0:-4], offset[0], offset[1]))

        img_dir_name = DATA_DIR + DATA_TYPE + '/' + str(j) + ".png"
        background.save(img_dir_name)
        filename = str(j) + ".png"
        img_size = os.stat(img_dir_name).st_size
        file_content += add_box_json_polygon(filename, img_size, icon_list)

    file_content = file_content[0:-1] + "}"
    write_string_to_json(file_content)


generate_data(10)