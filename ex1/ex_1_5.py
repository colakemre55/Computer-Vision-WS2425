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

def getKernel(size, sigma):
    kernel_1d = cv2.getGaussianKernel(size, sigma)
    kernel = np.outer(kernel_1d, kernel_1d)

    return kernel


def convolution(img, kernel):

    sizeOfMatrix = 5 #5x5


    xAxis = img.shape[0]
    yAxis = img.shape[1]

    
    finalImageSize = ( xAxis - sizeOfMatrix + 1 ) * ( yAxis - sizeOfMatrix + 1)
    img_final = np.zeros((xAxis - sizeOfMatrix + 1, yAxis - sizeOfMatrix + 1), dtype=np.float32)
    print(xAxis   , yAxis)
    result = 0 

    for i in range(xAxis - sizeOfMatrix + 1):
        for j in range(yAxis - sizeOfMatrix + 1):
            result = 0
            for ki in range(sizeOfMatrix):
                for kj in range(sizeOfMatrix):
                    result += kernel[ki][kj] * img[i + ki][j + kj]
            img_final[i, j] = result

        
    return img_final
    

def main():
   #print(cv2.getGaussianKernel(ksize=5,sigma=2))


    #retrieve gray image
    img_path = 'bonn.png'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    kernelOne = getKernel(5,2)
    kernelTwo = getKernel(5,2*np.sqrt(2))

    img_fin = convolution(img,kernelOne)
    img_fin = convolution(img_fin, kernelOne)
    img_fin = convolution(img_fin, kernelTwo)

    display_image("Gray img",img)
    display_image("Final img",img_fin)



if __name__ == "__main__":
    main()
