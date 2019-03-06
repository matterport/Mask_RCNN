import os
from PIL import Image
import random

ICON_DIR = os.getcwd() + "/Icons"
ICONS = os.listdir(ICON_DIR)

# Wire-frame dimensions (iPhone 8)
DIMENSIONS = (750, 1334)
bg_w, bg_h = DIMENSIONS
COLOR = (255, 255, 255, 255)

# Icon Dimensions
ICON_W = ICON_H = 50


def add_box_json(filename, size, x, y, width, height, label):
    s = " '{}{}':{{ 'filename': '{}', 'size': {}, 'regions': " \
        "[{{ 'shape_attributes': {{'name':'rect','x':{},'y':{},'width':{},'height':{}}}, " \
        "'region_attributes':{{'name':{} }} }}] }}"

    s = s.format(filename, size, filename, size, x, y, width, height, label)
    s = s.replace("'", '"')
    return s


for j in range(10):
    NUM_ICONS = random.randint(1, 3)
    background = Image.new('RGBA', DIMENSIONS, COLOR)
    for i in range(NUM_ICONS):
        cur_icon = ICONS[random.randint(0, len(ICONS) - 1)]
        img = Image.open('Icons/' + cur_icon, 'r').resize((50, 50))
        offset = random.randint(1, bg_w - ICON_W), random.randint(1, bg_h - ICON_H)
        background.paste(img, offset)
    img_dir_name = 'TrainingData/' + str(j) + ".png"
    background.save(img_dir_name)

    filename = str(j) + ".png"
    img_size = os.stat(img_dir_name).st_size
    x, y = offset
    x -= 10
    y -= 10
    height = ICON_H + 20
    width = ICON_W + 20
    cur_icon
    print(add_box_json(filename, img_size, x, y, width, height, cur_icon))
    break


