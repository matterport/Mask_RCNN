
import math
import numpy as np
import cv2


from PIL import Image


# In[18]:




def rotate_bound(image, angle):
    '''
    input:
        image - image with attribute shape contating height and width
        angle - rotation angle in degrees
    
    returns rotated image.
    '''
    # gggrab the dimensions of the image and then determine the
    # center
    # first 2 avlues of shape are height,width
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def add_image(img1, cucumberObj, x_center, y_center, x_scale, y_scale, angle):
    """
    name I gave: pasteImageOnOther ruined the overriding of base class function
    pasting re-scaled image on other image (collage effect)
    """
    cucumberObj = cucumberObj.resize((int(x_scale*cucumberObj.size[0]),int(y_scale*cucumberObj.size[1])), Image.ANTIALIAS)

    cucumberObj = cucumberObj.rotate(angle, resample=Image.BICUBIC, expand=True)
    

    rows,cols,channels = np.asarray(cucumberObj).shape
    x_from = x_center - math.floor(cols/2.)
    y_from = y_center - math.floor(rows/2.)

    img1.paste(cucumberObj, (x_from, y_from), cucumberObj)

    return img1

def add_imageWithoutTransparency(img1, cucumberObj, x_center, y_center, x_scale, y_scale, angle):
    """
    pasting re-scaled image on other image (collage effect) without transparency
    cucumberObj - an object on transparent background from ./objects folder
    """
    # apply all transformation-data saved on the object - so we have it's exact appearance in a certain Collage
    # it belongs to.
    objectInCollage = cv2.resize(cucumberObj,None,fx=x_scale, fy=y_scale, interpolation = cv2.INTER_CUBIC)

    #Rotate
    objectInCollage = rotate_bound(objectInCollage, 360-angle)
    
    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = objectInCollage.shape
    x_from = x_center - math.floor(cols/2.)
    x_to = x_center + math.ceil(cols/2.)
    y_from = y_center - math.floor(rows/2.)
    y_to = y_center + math.ceil(rows/2.)
    
    y_max, x_max, _ = img1.shape
    
    #cases image out of bg image
    if x_from < 0:
        objectInCollage = objectInCollage[:,-x_from:]
        x_from = 0
    if x_to >= x_max:
        objectInCollage = objectInCollage[:,:-(x_to-x_max+1)]
        x_to = x_max-1
    if y_from < 0:
        objectInCollage = objectInCollage[-y_from:,:]
        y_from = 0
    if y_to >= y_max:
        objectInCollage = objectInCollage[:-(y_to-y_max+1),:]
        y_to = y_max-1
    
    roi = img1[y_from:y_to, x_from:x_to]

    # Now create a mask of logo and create its inverse mask also
    alpha_image = mask_from_RGBA(objectInCollage)
    
    # dilate the mask
    # kernel = np.ones((5,5),np.uint8)
    # alpha_image = cv2.dilate(alpha_image, kernel, iterations=1)
    
    # since alpha values are in range [0,1] and we know that transparent bg has alpha < 0.4 approx.    
    _ , mask = cv2.threshold(alpha_image, 0.4 , 255, cv2.THRESH_BINARY)
    
    mask_inv = cv2.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(objectInCollage,objectInCollage,mask = mask)
    # Put logo in ROI and modify the main image
    #print(img1.shape)
    #print(cucumberObj.shape)
    dst = cv2.add(img1_bg,img2_fg[:,:,0:3])
    img1[y_from:y_to, x_from:x_to] = dst
    return img1

def mask_from_RGBA(image):
    return image[:,:,-1]
    
def image_to_mask(image):
    mask = np.zeros_like(image)
    while not np.any(mask):
        color = (0, 0, 0)
        distance = np.sum(image, axis=-1)
        mask = distance > 0
    return mask

def mask_to_image(mask):
    x,y,z = mask.shape
    image = np.zeros((x,y))
    for i in range(0,z):
        image += mask[:,:,i]*(i+1)
    print(z)
    return image


