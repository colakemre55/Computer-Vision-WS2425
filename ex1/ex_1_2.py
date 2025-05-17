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



def hist_own(img):
    M, N = img.shape
    histogram = np.zeros(shape=256, dtype=int)

    # Fill histogram
    for x in range(M):
        for y in range(N):
            intensity = img[x, y]
            histogram[intensity] += 1

    # (CDF)
    cdf = np.zeros(shape=256, dtype=int)
    cdf[0] = histogram[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + histogram[i]

    #
    minCdf = np.min(cdf[cdf > 0])
    totalPixels = M * N
    normalizedCdf = np.zeros(shape=256, dtype=int)

    # Normalize
    for i in range(256):
        normalizedCdf[i] = round((cdf[i] - minCdf) / (totalPixels - minCdf) * 255)

    # Remap the pixel values to equalized values
    eqImage = np.zeros_like(img)
    for x in range(M):
        for y in range(N):
            intensity = img[x, y]
            eqImage[x, y] = normalizedCdf[intensity]

    return eqImage


def main():
    #retrieve gray image
    img_path = 'bonn.png'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    display_image("Gray img",img)
    #task 2.1
    img_eq_cv = cv2.equalizeHist(img)
    display_image("2.1 - using opencv", img_eq_cv)

    #task 2.2
    img_eq_own = hist_own(img)
    display_image("2.2 - own implementation", img_eq_own)
    
    # Calculate abs pixel difference
    diff = cv2.absdiff(img_eq_cv, img_eq_own)
    print(f"Maximum pixel error: {np.max(diff)}")


if __name__ == "__main__":
    main()
