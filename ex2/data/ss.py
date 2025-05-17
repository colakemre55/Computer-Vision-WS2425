import cv2 as cv
import numpy as np

def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def build_gaussian_pyramid(img, levels=5):
    gaussian_pyramid = [img]
    for i in range(levels - 1):
        img = cv.pyrDown(img)
        gaussian_pyramid.append(img)
        display_image(f"Gaussian Level {i+1}", img)  #displaying all gaussians
    return gaussian_pyramid

def build_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    levels = len(gaussian_pyramid)
    for i in range(levels - 1):
        img_down = gaussian_pyramid[i + 1]
        img_up = cv.pyrUp(img_down, dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
        laplace = cv.subtract(gaussian_pyramid[i], img_up)
        laplacian_pyramid.append(laplace)
        display_image(f"Laplacian Level {i+1}", laplace)  # Display each Laplacian level
    laplacian_pyramid.append(gaussian_pyramid[-1])  # Add the smallest level as it is
    display_image(f"Laplacian Level {levels}", gaussian_pyramid[-1])  # Display the smallest level
    return laplacian_pyramid

def sliceAndCombine(laplacian_messi, laplacian_ronaldo):
    combined_pyramid = []
    for i in range(len(laplacian_messi)):
        laplacian1 = laplacian_messi[i]
        laplacian2 = laplacian_ronaldo[i]
        mid = laplacian1.shape[1] // 2
        combined = np.hstack((laplacian1[:, :mid], laplacian2[:, mid:]))
        combined_pyramid.append(combined)
        display_image(f"Combined Level {i+1}", combined)  # Display each combined level
    return combined_pyramid

def collapse_pyramid(pyramid):
    collapsed_img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        collapsed_img = cv.pyrUp(collapsed_img, dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
        collapsed_img = cv.add(collapsed_img, pyramid[i])
    return collapsed_img

def main():
    # Load the images
    ronaldoImg = cv.imread("data/ronaldo.jpeg")
    messiImg = cv.imread("data/messi.jpg")

    # Resize Ronaldo's image to match Messi's size
    ronaldoImg = cv.resize(ronaldoImg, (messiImg.shape[1], messiImg.shape[0]))

    # Build Gaussian pyramids
    gaussian_messi = build_gaussian_pyramid(messiImg)
    gaussian_ronaldo = build_gaussian_pyramid(ronaldoImg)

    # Build Laplacian pyramids
    laplacian_messi = build_laplacian_pyramid(gaussian_messi)
    laplacian_ronaldo = build_laplacian_pyramid(gaussian_ronaldo)

    # Combine the Laplacian pyramids
    combined_pyramid = sliceAndCombine(laplacian_messi, laplacian_ronaldo)

    # Collapse the pyramid to get the final blended image
    blended_image = collapse_pyramid(combined_pyramid)

    # Display the final blended image
    display_image("Blended Image", blended_image)

if __name__ == "__main__":
    main()
