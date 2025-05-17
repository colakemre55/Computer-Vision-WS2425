import cv2
import numpy as np
import time
import random
import sys
from numpy.random import randint


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    

def addSaltPepper(img):
    row , col = img.shape


    for i in range(0, row-1):
        for j in range(0, col-1):
            num = random.randint(1,10)
            if (num<=3):
                randBlackWhite = random.randint(1,2)
                if (randBlackWhite == 1):
                    img[i][j] = 0
                else:
                    img[i][j] = 255
        
    return img

def main():
    #retrieve gray image
    img_path = 'bonn.png'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    display_image("Gray scale image", img)

    saltPepperImage = addSaltPepper(img)
    
    display_image("Salt Pepper Image",saltPepperImage)

    filteredImage = cv2.GaussianBlur(saltPepperImage, (5,5), 2)

    display_image("Filtered image" , filteredImage)

    stackBlurImage = cv2.stackBlur(saltPepperImage, ksize=(5,5))
    display_image("stackblur image" , stackBlurImage)

    bilateralImg = cv2.bilateralFilter(saltPepperImage,10,50,50)
    display_image("bilateral filter image" , bilateralImg)
if __name__ == "__main__":
    main()
