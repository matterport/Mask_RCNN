import visualize
import pickle
import os
import sys
import traceback
import shutil
import string
from skimage import io
import numpy as np

path = os.path.join(os.getcwd(), 'data', 'images')
print('dat file path %s' % path)
files = os.listdir(path)
for file in files:
    if file.endswith(".dat"):
        index = file[0: -4]
        index = int(index)
        f = open(os.path.join(path, '%s.dat' % index), 'rb')
        image, r, class_names = pickle.load(f)
        print(image.shape)
        # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        #                             class_names, r['scores'])
        # save sample file
        dir_path = os.path.join(os.getcwd(), "data", "samples", "%s" % index)

        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
        else:
            os.mkdir(dir_path)

        # io.imsave(os.path.join(dir_path, "ori.jpg"), image)
        src = os.path.join(path, "%d.bmp" % index)
        dst = os.path.join(dir_path, "ori.bmp")
        shutil.copy(src, dst)

        boxes = np.asarray(r['rois'])
        for i in range(boxes.shape[0]):
            y1, x1, y2, x2 = boxes[i]
            sub_image = image[y1:y2, x1:x2, :]
            io.imsave(os.path.join(dir_path, "%s.jpg" % i), sub_image)
