# Custom code to visualize real time video using OpenCV

import cv2 as cv
import numpy as np 

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    # Apply mask to the image
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1-alpha) + alpha * c,
            image[:, :, n]
        )
    
    return image

def display_instances(image, boxes, masks, ids, names, scores):
    num_instances = boxes.shape[0]

    if not num_instances:
        print("No instances to display!\n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    colors = random_colors(num_instances)
    height, width = image.shape[:2]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue
        
        y1, x1, y2, x2 = boxes[i]
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = "{} {:.2f}".format(label, score) if score else label
        image = cv.putText(
            image, caption, (x1, y1), cv.FONT_HERSHEY_COMPLEX, 0.7, color, 2

        )

    return image

if __name__ == '__main__':
    import os
    import sys

    sys.path.append('./mrcnn')
    sys.path.append('./samples/coco')
    import random
    import math
    import time
    import coco
    import utils
    import model as modellib

    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    class InferenceConfig(coco.CocoConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    
    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(
        mode='inference', model_dir=MODEL_DIR, config=config
    )

    model.load_weights(COCO_MODEL_PATH, by_name=True)


    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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
                'teddy bear', 'hair drier', 'toothbrush']

    cap = cv.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        results = model.detect([frame], verbose=0)
        r = results[0]

        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
        )

        cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv.destroyAllWindows()