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

def filterSpatial(img):
    # Define the 7x7 Sobel filter
    sobelFilter = np.array([
        [-1, -4, -5,  0,  5,  4,  1],
        [-6, -24, -30, 0, 30, 24,  6],
        [-15, -60, -75, 0, 75, 60, 15],
        [-20, -80, -100, 0, 100, 80, 20],
        [-15, -60, -75, 0, 75, 60, 15],
        [-6, -24, -30, 0, 30, 24,  6],
        [-1, -4, -5,  0,  5,  4,  1]])


    img_x, img_y = img.shape[:2]
    pad_size = sobelFilter.shape[0] // 2
    img_padded = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')

    img_filtered = np.zeros_like(img)


    for i in range(img_x):
        for j in range(img_y):

            wantedRegion = img_padded[i:i + 7, j:j + 7]

            img_filtered[i, j] = np.sum(wantedRegion * sobelFilter)

    # Normalize to 0-255 range
    img_filtered = np.clip(img_filtered, 0, 255)
    img_filtered = img_filtered.astype(np.uint8)

    return img_filtered


def filterSpatial2(img, sobel):

    sizeOfMatrix = 7  # 7x7 filter
    bound = (sizeOfMatrix - 1) // 2

    # Convert the image to float32 to prevent overflow issues
    img = img.astype(np.float32)

    xAxis, yAxis = img.shape[:2]
    img_final = np.zeros((xAxis - sizeOfMatrix + 1, yAxis - sizeOfMatrix + 1), dtype=np.float32)

    for i in range(bound, xAxis - bound - 1):
        for j in range(bound, yAxis - bound - 1):
            result = 0
            for ki in range(sizeOfMatrix):
                for kj in range(sizeOfMatrix):
                    result += sobel[ki][kj] * img[i + ki - bound][j + kj - bound]
            img_final[i - bound, j - bound] = result

    # Normalize the result to 0-255 and convert back to uint8
    img_final = np.clip(img_final, 0, 255)
    img_final = img_final.astype(np.uint8)

    return img_final

    
def frequencyFilter(img, sobel):
    imgFft = np.fft.fft2(img)
    
    sobelPadded = np.zeros_like(img)
    filterShape = sobel.shape
    sobelPadded[:filterShape[0], :filterShape[1]] = sobel

    sobelFft = np.fft.fft2(sobelPadded)

    result = sobelFft * imgFft

    imgFiltered = np.fft.ifft2(result)

    resultFiltered = np.real(imgFiltered) #to prevent imaginary parts
    
    return resultFiltered


def main():
    img = cv.imread("data/messi.jpg")
    display_image("img",img)

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float32)
    display_image("gray" , img_gray)

    imgFiltered = filterSpatial(img_gray)
    display_image("Filtered" , imgFiltered)

    sobel7x7 = np.array([[-3, -2, -1, 0, 1, 2, 3],
                         [-3, -2, -1, 0, 1, 2, 3],
                         [-3, -2, -1, 0, 1, 2, 3],
                         [-3, -2, -1, 0, 1, 2, 3],
                         [-3, -2, -1, 0, 1, 2, 3],
                         [-3, -2, -1, 0, 1, 2, 3],
                         [-3, -2, -1, 0, 1, 2, 3]])
    
    sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
    
    spatialFiltered = cv.filter2D(img_gray, -1 , sobel7x7)
    display_image("Spatial Filtered" , spatialFiltered)

    imgFreqFiltered = frequencyFilter(img_gray, sobel7x7)
    display_image("Freq Filtered" , imgFreqFiltered)

if __name__ == "__main__":
    main()