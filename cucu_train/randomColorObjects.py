import os
import random
import cv2
import numpy as np




def randomColorObject(objectPath, destPath):
    im = cv2.imread(objectPath,cv2.IMREAD_UNCHANGED)
    b = random.randint(0,255)
    g = random.randint(0,10)
    r = random.randint(0,255)
    print(im.shape)
    im[np.where((im != [0,0,0,0]).all(axis = 2))] = im[np.where((im != [0,0,0,0]).all(axis = 2))] +  [b,g,r,0]
    cv2.imwrite(destPath, im)