import cv2
import numpy as np

# change this to make each class were interested in a specific color
# ie cows near track is red, cow not near track is orange etc

def random_colors(N):
    # initialise the random number with a seed
    # get the same color for each frame
    np.random.seed(1)
    # generate tuple of values to represent color for mask
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors

# apply mask to image passed in

def apply_mask(image, mask, color, aplha=0.5):
    # apply the mask to the image, loop over the RGB colors
    for n, c in enumerate(color):
        # if the x,y location is in a mask apply the color and alpha channels
        # else use existing pixel colors 
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image



def display_instances(image, boxes, masks, ids, names, scores):
    # how many instances do we have
    n_instances = boxes.shape[0]
    # if there is no instances print it out
    if not n_instances:
        print('No Instances to Display')
    else:
        # assert some things about the boxes mask and ids
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    # get the color we are going to use for the mask, n_instance = number of colors required
    colors = random_colors(n_instances)
    # define the height and width take the first two items in the shape list
    height, width = image.shape[:2]
    # start the for loop, enumerate over th ecolors array
    for i, color in enumerate(colors):
        # skip if we get bad data on this iteration of colors for boxes array contine loop from top
        if not np.any(boxes[i]):
            continue
        # get locations of boxes co-ordinates
        y1, x1, y2, x2 = boxes[i]
        # get teh mask from the masks list
        mask = masks[:, :, i]
        # return an image with the mask on it
        image = apply_mask(image, mask, color)
        # add the bounding boxes using cv2 lib
        # image, top left, bottom right co-ordinates, color of box, line thickness
        image = cv2.rectangle(image, (x1, y1), (x2, y2) , color, 2)
        # create a label from the class name
        label = names[ids[i]]
        # create a score from the confidence number or none if empty
        score = scores[i] if scores is not None else None
        # create a label for the bbox, label and confidence
        caption = '{} {:.2f}'.format(label, score) if score else label
        # add a label to the image in the top left corner of the bounding box
        # image, the text, starting loc for text, the font used alpha, color of box, stroke
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHET_COMPEX, 0.7, color, 2
        )
    # return the image to the caller
    return image

# testing file if run directly wont run if imported
# this code allows for a stand alone run using python kiwi_rail.py
# it is intended for testing camera input and to see frame rate achived
# this code allows us an easy place to accesss the frame and mask data for
# passing to other python modules

if __name__ == '__main__':
    import os
    import sys
    import random
    import math
    import time
    from samples.coco import coco
    from mrcnn import utils
    from mrcnn import model as modellib

# get location of weights file
    ROOT_DIR =os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # download if not found
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

# create an inference class
class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
# instatiate the config
config = InferenceConfig()
# print the config
config.display()

# initialise the model
model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)

# load the weights file and use the supplied name
model.load_weights(COCO_MODEL_PATH, by_name=True)

# class name list, stuff we can detect
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# initialise capture device
capture = cv2.VideoCapture(0)

# set the result ouput window dimentions
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# while loop

while True:
    ret, frame = capture.read()
    results = model.detect([frame], verbose=0)
    # get the first thing in the results array
    r = results[0]

    # get the image overlayed with the mask
    frame = display_instances(
        frame, r['rois'], r['maks'], r['class_ids'], class_names, r['scores']
    )

    # display the image to screen
    cv2.imshow('frame', frame)

    # exit on q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# clean up after quit
capture.release()
cv2.destroyAllWindows()






