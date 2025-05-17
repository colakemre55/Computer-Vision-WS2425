import cv2 as cv
import numpy as np
import random
import sys
from numpy.random import randint


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows() 


if __name__ == '__main__':

    # set image path
    img_path = 'bonn.png' 
    
    #-----
    # 2a: read and display the image 
    img = cv.imread(img_path)
    display_image('2 - a - Original Image', img)

    #-----
    # 2b: display the intensity image
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    display_image('2 - b - Intensity Image', img_gray)
    

    #-----
    # 2c: for loop to perform the operation
    img_half = img_gray * 0.5 #multiplying the intensity image

    result = np.zeros_like(img) #subtracting the gray from the original one
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b, g, r = img[i, j]
            I = img_half[i, j]
            maxB = max(b - I, 0)
            maxG = max(g - I, 0)
            maxR = max(r- I, 0)

            result[i, j] = [maxB, maxG, maxR]

    img_cpy = result
    display_image('2 - c - Reduced Intensity Image', img_cpy)

    #-----
    # 2d: one-line statement to perfom the operation above
    img_cpy = np.maximum(img - (np.expand_dims(img_gray * 0.5, axis=2)) , 0 ).astype(np.uint8)
    display_image('2 - d - Reduced Intensity Image One-Liner', img_cpy)    

    #-----
    # 2e: Extract the center patch and place randomly in the image
    centerx = img.shape[1]//2
    centery = img.shape[0]//2
    img_patch = img[centery-8:centery+8, centerx-8:centerx+8, :]
    display_image('2 - e - Center Patch', img_patch)  
    
    rand_coord=randint(0,img.shape[0]-16),randint(0,img.shape[1]-16)
    img_cpy[centery-8:centery+8, centerx-8:centerx+8, :] = 0
    img_cpy[rand_coord[0]:rand_coord[0]+16,rand_coord[1]:rand_coord[1]+16,:]=img_patch

    # Random location of the patch for placement
    display_image('2 - e - Center Patch Placed Random %d, %d' % (rand_coord[0], rand_coord[1]), img_cpy)  

    #-----
    # 2f: Draw random rectangles and ellipses
    img_width, img_height = img.shape[1], img.shape[0]

    def random_color():
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # Drawing 10 rects
    for _ in range(10):
        x1, y1 = random.randint(0, img_width), random.randint(0, img_height)
        x2, y2 = random.randint(x1, min(x1 + 100, img_width)), random.randint(y1, min(y1 + 100, img_height))
        cv.rectangle(img, (x1, y1), (x2, y2), random_color(), -1)

    # Drawing 10 ellipses
    for _ in range(10):
        center = (random.randint(0, img_width), random.randint(0, img_height))
        axes = (random.randint(20, 100), random.randint(20, 100))
        angle = random.randint(0, 360)
        cv.ellipse(img, center, axes, angle, 0, 360, random_color(), -1)
    
    display_image('2 - f - Rectangles and Ellipses', img)
    
   
    # destroy all windows
    cv.destroyAllWindows()
