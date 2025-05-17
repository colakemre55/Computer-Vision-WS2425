import cv2 as cv
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
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows() 

def build_gaussian_pyramid(img):
    imgDown = cv.pyrDown(img)

    return imgDown

def build_laplacian_pyramid(img):
    img_down = cv.pyrDown(img) #use gaus
    img_up = cv.pyrUp(img_down, dstsize=(img.shape[1], img.shape[0]))
    laplace = cv.subtract(img, img_up)
    
    return laplace

def sliceAndCombine(img1, img2):
    # Resize img2 to match img1's dimensions
    img2_resized = cv.resize(img2, (img1.shape[1], img1.shape[0]))

    # Find the midpoint of the width
    mid = img1.shape[1] // 2

    # Slice each image at the midpoint
    sliced1 = img1[:, :mid]
    sliced2 = img2_resized[:, mid:]

    # Combine the two halves horizontally
    combined = np.hstack((sliced1, sliced2))

    # Display and return the combined image
    display_image("Combined", combined)
    return combined


def main():
    ronaldoImg = cv.imread("data/ronaldo.jpeg")
    #display_image("img",ronaldoImg)

    messiImg = cv.imread("data/messi.jpg")
    #display_image("img",messiImg)

    '''ronaldoGray = cv.cvtColor(ronaldoImg, cv.COLOR_BGR2GRAY).astype(np.float32)
    display_image("ronaldo gray" , ronaldoGray)

    messiGray = cv.cvtColor(messiImg, cv.COLOR_BGR2GRAY).astype(np.float32)
    display_image("ronaldo gray" , messiGray)
'''

    laplacianMessi = build_laplacian_pyramid(messiImg)
    laplacianRonaldo = build_laplacian_pyramid(ronaldoImg)

    print(laplacianMessi.shape)
    print(laplacianRonaldo.shape)
    display_image(" laplacianMessi", laplacianMessi)
    display_image(" laplacianRonaldo", laplacianRonaldo)

    sliceAndCombine(laplacianMessi, laplacianRonaldo)
    
    

if __name__ == "__main__":
    main()