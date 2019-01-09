import os
import random
import cv2
import numpy as np




def randomColorObject(objectPath, destPath):
    im = cv2.imread(objectPath,cv2.IMREAD_UNCHANGED)
    b = random.randint(0,255)
    g = random.randint(0,10)
    r = random.randint(0,255)
    im[np.where((im != [0,0,0,0]).all(axis = 2))] = im[np.where((im != [0,0,0,0]).all(axis = 2))] +  [b,g,r,0]
    cv2.imwrite(destPath, im)


def toGray(objectPath, destPath):
    image = cv2.imread(objectPath,cv2.IMREAD_UNCHANGED)
    # extract the alpha channel:
    alpha= image[:,:,3]
    image = image[:,:,:3]
    #use that as a mask for bitwise compositing:
    image = cv2.bitwise_and(image,image,mask = alpha)

    #convert *that* to grayscale
    image = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGRA)
    image[:,:,3] = alpha
    cv2.imwrite(destPath, image)
