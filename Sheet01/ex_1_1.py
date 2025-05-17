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


def recursive_integral(img, current_index):
    global img_integral
    global flag

    if current_index == (0, 0): # Left upper corner case
        img_integral[0, 0] = img[0, 0]
        flag = True
    elif current_index[0] == 0: # Upper edge case
        next_index = (0, current_index[1] - 1)
        img_integral[0, current_index[1]] = img[-1, -1] + recursive_integral(img[:, :-1], next_index)
    elif current_index[1] == 0: # Left edge case
        next_index = (current_index[0] - 1, 0)
        img_integral[current_index[0], 0] = img[-1, -1] + recursive_integral(img[:-1, :], next_index)
    else: # Usual case
        if flag:
            img_integral[current_index[0], current_index[1]] = \
                img[-1, -1] + \
                img_integral[current_index[0], current_index[1] - 1] + \
                recursive_integral(img[:-1, :], (current_index[0] - 1, current_index[1])) - \
                img_integral[current_index[0] - 1, current_index[1] - 1]
        else:
            img_integral[current_index[0], current_index[1]] = \
                img[-1, -1] + \
                recursive_integral(img[:, :-1], (current_index[0], current_index[1] - 1)) + \
                recursive_integral(img[:-1, :], (current_index[0] - 1, current_index[1])) - \
                img_integral[current_index[0] - 1, current_index[1] - 1]

    return img_integral[current_index[0], current_index[1]]


def get_integral_img(img):
    # Initialize image with 0s
    global img_integral
    global flag
    img_integral = np.zeros(shape = img.shape, dtype = int)
    flag = False

    # Start recursive calculation
    current_index = (img_integral.shape[0] - 1, img_integral.shape[1] - 1)
    img_integral[-1, -1] = recursive_integral(img, current_index)

    return img_integral


def get_mean_val_naive(img, rand_coords = [[0, 0]], size = None, verbose = True):
    if size is None:
        size = (img.shape[0], img.shape[1])

    # Run through randomized coords
    for (x, y) in rand_coords:
        # Sum up values
        sum = 0
        for row in img[x : x + size[0], y : y + size[1]]:
            for val in row:
                sum += val
        mean = sum / (size[0] * size[1])
        if verbose:
            print(f"Mean using simple summation ({x}, {y}): {mean}")


def get_mean_val_own(img, rand_coords = [[0, 0]], size = None, verbose = True):
    if size is None:
        size = (img.shape[0], img.shape[1])

    # Get integral image
    img_integral = get_integral_img(img)
    img_integral = np.pad(img_integral, ((1, 0), (1, 0)), 'constant')

    # Run through randomized coords
    for (x, y) in rand_coords:
        sum = img_integral[x + size[0], y + size[1]] - img_integral[x, y]
        mean = sum / (size[0] * size[1])
        if verbose:
            print(f"Mean using own integral implementation ({x}, {y}): {mean}")


def get_mean_val_cv(img, rand_coords = [[0, 0]], size = None, verbose = True):
    if size is None:
        size = (img.shape[0], img.shape[1])

    # Get integral image
    img_integral = cv.integral(img)

    # Run through randomized coords
    for (x, y) in rand_coords:
        sum = img_integral[x + size[0], y + size[1]] - img_integral[x, y]
        mean = sum / (size[0] * size[1])
        if verbose:
            print(f"Mean using cv integral implementation ({x}, {y}): {mean}")


def time_function(func, img):
    # Get randomized patches
    np.random.seed(19)
    patch_size = (100, 100)
    n = 10
    rand_coords = np.dstack(
        [randint(low = 0, high = img.shape[0] - patch_size[0], size = n), 
         randint(low = 0, high = img.shape[1] - patch_size[1], size = n)]
         )[0]
    
    # Time function
    start = time.time()
    func(img, rand_coords, patch_size, verbose = False)
    end = time.time()
    print(f"Elapsed time for {func.__name__}: {end - start}")


def main():
    # Load image and convert to gray
    img_path = 'bonn.png'
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    '''# Task 3.1a
    img_integral_own = get_integral_img(img)
    img_integral_display = cv.normalize(img_integral, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    display_image("3.1 - Rectangles and Integral Images", img_integral_display)
'''
    '''# Task 3.1b
    get_mean_val_naive(img, verbose = True)
    get_mean_val_cv(img, verbose = True)
    get_mean_val_own(img, verbose = True)

    # Task 3.1c
    time_function(get_mean_val_naive, img)
    time_function(get_mean_val_cv, img)
    time_function(get_mean_val_own, img)'''


if __name__ == "__main__":
    main()
