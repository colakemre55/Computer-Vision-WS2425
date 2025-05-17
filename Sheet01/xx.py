import cv2
import numpy as np
import random


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


def add_salt_pepper(img, noise_percent):
    """
    Add salt and pepper noise to the image.
    :param img: input image
    :param noise_percent: percentage of image to be corrupted with noise
    :return: noisy image
    """
    row, col = img.shape
    total_pixels = row * col
    num_noisy_pixels = int(noise_percent * total_pixels)
    
    # Randomly choose pixel locations to apply noise
    for _ in range(num_noisy_pixels):
        i = random.randint(0, row - 1)
        j = random.randint(0, col - 1)
        img[i][j] = random.choice([0, 255])  # randomly choose black or white
    
    return img


def mean_gray_value_distance(img1, img2):
    """
    Calculate mean gray value distance between two images.
    :param img1: first image
    :param img2: second image
    :return: mean gray value distance
    """
    return np.mean(np.abs(img1 - img2))


def main():
    # Load and convert to gray image
    img_path = 'bonn.png'
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    display_image("Gray scale image", img_gray)

    # Add salt and pepper noise (30% noise)
    salt_pepper_img = add_salt_pepper(img_gray.copy(), 0.30)
    display_image("Salt Pepper Image", salt_pepper_img)

    # Filter the image with different methods
    filtered_gaussian = cv2.GaussianBlur(salt_pepper_img, (5, 5), 2)
    display_image("Gaussian Filtered image", filtered_gaussian)

    # Bilateral filter
    bilateral_img = cv2.bilateralFilter(salt_pepper_img, 9, 75, 75)
    display_image("Bilateral Filter Image", bilateral_img)

    # Calculate mean gray value distance for different filter sizes and choose the best one
    original_img = img_gray.copy()
    filter_sizes = [1, 3, 5, 7, 9]
    best_filter_size = None
    min_distance = sys.maxsize

    for size in filter_sizes:
        gaussian_filtered = cv2.GaussianBlur(salt_pepper_img, (size, size), 2)
        bilateral_filtered = cv2.bilateralFilter(salt_pepper_img, size, 75, 75)

        # Calculate the distance for Gaussian and Bilateral
        gaussian_distance = mean_gray_value_distance(original_img, gaussian_filtered)
        bilateral_distance = mean_gray_value_distance(original_img, bilateral_filtered)

        # Select the best filter size
        if gaussian_distance < min_distance:
            min_distance = gaussian_distance
            best_filter_size = (size, 'Gaussian')

        if bilateral_distance < min_distance:
            min_distance = bilateral_distance
            best_filter_size = (size, 'Bilateral')

    print(f"Best filter: {best_filter_size}")

if __name__ == "__main__":
    main()
