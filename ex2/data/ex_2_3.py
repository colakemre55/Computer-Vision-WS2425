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




def main():
    ronaldoImg = cv.imread("data/ronaldo.jpeg")
    display_image("Ronaldo",ronaldoImg)

    messiImg = cv.imread("data/messi.jpg")
    display_image("Messi",messiImg)

    ronaldoGray = cv.cvtColor(ronaldoImg, cv.COLOR_BGR2GRAY).astype(np.float32)
    display_image("gray" , ronaldoGray)

    messiGray = cv.cvtColor(messiImg, cv.COLOR_BGR2GRAY).astype(np.float32)
    display_image("gray" , messiGray)



if __name__ == "__main__":
    main() 