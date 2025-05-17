import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

###########################################################
#                                                         #
#                        TASK 2                           #
#                                                         #  
###########################################################
def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

def get_convolution_using_fourier_transform(img, kernel):
    imgFft = np.fft.fft2(img)
    
    sobelPadded = np.zeros_like(img)
    filterShape = kernel.shape
    sobelPadded[:filterShape[0], :filterShape[1]] = kernel

    sobelFft = np.fft.fft2(sobelPadded)

    result = sobelFft * imgFft

    imgFiltered = np.fft.ifft2(result)

    resultFiltered = np.real(imgFiltered) #to prevent imaginary parts
    
    return resultFiltered

def get_convolution(image, kernel):

    img_x, img_y = image.shape[:2]
    pad_size = kernel.shape[0] // 2
    img_padded = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')

    img_filtered = np.zeros_like(image)

    for i in range(img_x):
        for j in range(img_y):

            wantedRegion = img_padded[i:i + 7, j:j + 7]

            img_filtered[i, j] = np.sum(wantedRegion * kernel)

    # Normalize
    img_filtered = np.clip(img_filtered, 0, 255)
    img_filtered = img_filtered.astype(np.uint8)

    return img_filtered

def mean_absolute_difference(img1, img2):

    diff = np.abs(img1.astype("float32") - img2.astype("float32"))
    meanAbsDif = np.mean(diff)
    return meanAbsDif

def task2():
    # Load image
    # note: get_convolution takes about 5mins on my pc with oldtown.jpg, to test fast messi/ronaldo.jpg could be used
    start_time = time.time()
    img = cv2.imread("./data/oldtown.jpg")
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    kernel_x = cv2.getDerivKernels(1, 0, 7)[0]
    kernel_y = cv2.getDerivKernels(0, 1, 7)[1]
    kernel = np.outer(kernel_y, kernel_x)
    fft_result = get_convolution_using_fourier_transform(image, kernel)
    display_image("fft result", fft_result)
    cv_result = cv2.filter2D(image, -1 , kernel)
    display_image("cv2.read result", cv_result)
    conv_result = get_convolution(image, kernel)
    display_image("convolution result", conv_result)
    
    end_time = time.time()
    print("cv_result - conv_result" ,mean_absolute_difference( cv_result,conv_result))
    print("cv_result - fft_result" ,mean_absolute_difference(cv_result,fft_result))
    print("fft_result - conv_result" ,mean_absolute_difference(fft_result,conv_result))
    
    print("Time taken for this process is : ", end_time -  start_time )

###########################################################
#                                                         #
#                        TASK 3                           #
#                                                         #  
###########################################################

def normalized_cross_correlation(image, template):
    img_h, img_w = image.shape
    temp_h, temp_w = template.shape
    result = np.zeros((img_h - temp_h + 1, img_w - temp_w + 1))
    template_mean = np.mean(template)
    
    template_centered = template - template_mean
    template_denominator = np.sqrt(np.sum(template_centered ** 2))

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            image_patch = image[i:i+temp_h, j:j+temp_w]

            image_patch_mean = np.mean(image_patch)
            image_patch_centered = image_patch - image_patch_mean
            numerator = np.sum((template - template_mean) * (image_patch - image_patch_mean))
            patch_denominator = np.sqrt(np.sum(image_patch_centered ** 2))

            if patch_denominator!=0:
                result[i, j] = numerator / (template_denominator * patch_denominator)
            else:
                result[i, j] = 0
    return result

def ssd(image, template):
    img_h, img_w = image.shape
    temp_h, temp_w = template.shape

    #result will be a 2D array with size (img_h - temp_h + 1, img_w - temp_w + 1) 
    result = np.zeros((img_h - temp_h + 1, img_w - temp_w + 1))

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            image_patch = image[i:i+temp_h, j:j+temp_w]
            ssd_value = np.sum((template - image_patch) ** 2)
            
            result[i, j] = ssd_value

    return result

def draw_rectangle_at_matches(image, template_h, template_w, matches):
    for (y, x) in matches:
        cv2.rectangle(image, (x, y), (x + template_w, y + template_h), (0, 0, 0), 1)
    return image

def task3():
    # Load images
    image = cv2.imread("./data/einstein.jpeg", cv2.IMREAD_GRAYSCALE)
    template = cv2.imread("./data/einstein_eye.jpeg", cv2.IMREAD_GRAYSCALE)
    
    template_h, template_w = template.shape

    # convert to float and apply intensity transformation to image
    image = image.astype(np.float32) / 255.0
    template = template.astype(np.float32) / 255.0
    
    result_ncc = normalized_cross_correlation(image, template)
    result_ssd = ssd(image, template)
    
    #Note: SSD couldnt match with 0.1 so I had to change to 0.2
    ssd_matches = np.where(result_ssd <= 0.2) 
    ncc_matches = np.where(result_ncc >= 0.7)

    image_ssd = image.copy()
    image_ncc = image.copy()

    image_ssd = draw_rectangle_at_matches(image_ssd, template_h, template_w, zip(*ssd_matches))
    image_ncc = draw_rectangle_at_matches(image_ncc, template_h, template_w, zip(*ncc_matches))

    cv2.imshow("SSD Matches", image_ssd)
    cv2.imshow("NCC Matches", image_ncc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Now, try to subtract 0.5 to the image (make sure that the values do not become negative)
    image2=np.maximum(image-0.5,0).astype(np.float32)
    result_ncc2 = normalized_cross_correlation(image2, template)
    result_ssd2 = ssd(image2, template)
    
    ssd_matches2 = np.where(result_ssd2 <= 0.1) 
    ncc_matches2 = np.where(result_ncc2 >= 0.7)
    
    image_ssd2 = image2.copy()
    image_ncc2 = image2.copy()

    image_ssd2 = draw_rectangle_at_matches(image_ssd2, template_h, template_w, zip(*ssd_matches2))
    image_ncc2 = draw_rectangle_at_matches(image_ncc2, template_h, template_w, zip(*ncc_matches2))

    cv2.imshow("SSD Matches", image_ssd2)
    cv2.imshow("NCC Matches", image_ncc2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Are there any differences between using SSD and NCC?
    # ans: Yes. As it can be seen when the image gets darkened, 
    # the SDD cannot find the eye. That is because the SSD is sensitive to the intensity of the image.
    # On the other hand, NCC is not sensitive to the intensity of the image because
    # it normalizes the image and the template. So, it can find the eye even when the image is darkened.
    


###########################################################
#                                                         #
#                        TASK 4                           #
#                                                         #  
###########################################################


def build_gaussian_pyramid_opencv(image, num_levels):
    pyramid = [image]
    
    for i in range(num_levels-1):
        image = cv2.pyrDown(pyramid[-1])  # Downsample the last image in the pyramid
        pyramid.append(image)
    
    return pyramid


def build_gaussian_pyramid(image, num_levels):
    pyramid = [image]
    
    for i in range(num_levels-1):
        blurred_image = cv2.GaussianBlur(pyramid[-1], (5, 5), 0)
        
        # Downsample the image by taking every second pixel in both dimensions
        downsampled_image = blurred_image[::2, ::2]
        pyramid.append(downsampled_image)
    
    return pyramid

def template_matching_multiple_scales(pyramid_image, pyramid_template):
    best_match = None
    best_scale = None
    best_value = -1
    match_location = None
    
    for level in range(len(pyramid_image)-1, -1, -1):
        image_level = pyramid_image[level]
        template_level = pyramid_template[level]
        
        # If it's the coarsest level, search the whole image
        if level == len(pyramid_image) - 1:
            result = normalized_cross_correlation(image_level, template_level)
        else:
            # Define a small search window around the match location from the coarser level
            x, y = match_location
            t_height, t_width = template_level.shape
            search_window_size = max(t_height, t_width)  # Define window size based on template size
            
            # Extract a search region (window) around the scaled-up match location
            x_start = max(0, x*2 - search_window_size)
            y_start = max(0, y*2 - search_window_size)
            x_end = min(image_level.shape[1], x*2 + search_window_size + t_width)
            y_end = min(image_level.shape[0], y*2 + search_window_size + t_height)
            
            # Extract the search region from the finer level
            search_region = image_level[y_start:y_end, x_start:x_end]
            result = normalized_cross_correlation(search_region, template_level)
        
        # Find the best match in this level
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if level < len(pyramid_image) - 1:  # Adjust location to image coordinates
            max_loc = (max_loc[0] + x_start, max_loc[1] + y_start)
        
        if max_val > best_value:
            best_value = max_val
            match_location = max_loc  # Update the best match location
            best_scale = level

    return match_location, best_scale, best_value


def task4():
    # Load images
    image = cv2.imread("./data/traffic.jpg", cv2.IMREAD_GRAYSCALE)
    template = cv2.imread("./data/traffic-template.png", cv2.IMREAD_GRAYSCALE)

    cv_pyramid = build_gaussian_pyramid_opencv(image, 4)

    my_pyramid = build_gaussian_pyramid(image, 4)
    my_pyramid_template = build_gaussian_pyramid(template, 4)

    # TODO: compare and print mean absolute difference at each level
    for i in range(4):
        mad = np.mean(np.abs(cv_pyramid[i], my_pyramid[i]))
        print(f"Level {i} - Mean Absolute Difference: {mad:.5f}")
    # TODO: calculate the time needed for template matching without the pyramid
    
    start_time = time.time()
    result_no_pyramid = normalized_cross_correlation(image, template)
    end_time = time.time()
    print(f"Time for template matching without pyramid: {end_time - start_time:.5f} seconds")
    #result = template_matching_multiple_scales(my_pyramid, my_pyramid_template)
    
    # TODO: calculate the time needed for template matching with the pyramid
    start_time = time.time()
    result_with_pyramid = template_matching_multiple_scales(my_pyramid, my_pyramid_template)
    end_time = time.time()
    print(f"Time for template matching with pyramid: {end_time - start_time:.5f} seconds")

    # TODO: show the template matching results using the pyramid
    h, w = template.shape
    best_match, best_scale, best_value = result_with_pyramid
    # Scale the match location back to the original image size
    scale_factor = 2 ** best_scale
    best_match_scaled = (best_match[0] * scale_factor, best_match[1] * scale_factor)

    # Draw rectangle around the match in the original image
    result_image_with_pyramid = image.copy()
    cv2.rectangle(result_image_with_pyramid, best_match_scaled,
                  (best_match_scaled[0] + w, best_match_scaled[1] + h), 255, 2)
    cv2.imshow('Best Match With Pyramid', result_image_with_pyramid)
    cv2.waitKey(0)

###########################################################
#                                                         #
#                        TASK 5                           #
#                                                         #  
###########################################################

def build_gaussian_pyramid(img, levels=5):
    gaussian_pyramid = [img]
    for i in range(levels - 1):
        img = cv2.pyrDown(img)
        gaussian_pyramid.append(img)
        cv2.imshow("Gaussian level",img)
        cv2.waitKey(0)
    return gaussian_pyramid

def build_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    levels = len(gaussian_pyramid)
    for i in range(levels - 1):
        img_down = gaussian_pyramid[i + 1]
        img_up = cv2.pyrUp(img_down, dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
        laplace = cv2.subtract(gaussian_pyramid[i], img_up)
        laplacian_pyramid.append(laplace)
        cv2.imshow("Gaussian level",laplace)
        cv2.waitKey(0)
    laplacian_pyramid.append(gaussian_pyramid[-1])  # Add the smallest level as it is

    return laplacian_pyramid




def blendPyramids(laplacianMessi, laplacianRonaldo):
    combinedPyramid = []
    for i in range (len(laplacianMessi)):
        laplace1 = laplacianMessi[i]
        laplace2 = laplacianRonaldo[i]
        middle = laplace1.shape[1] // 2
        sliced1 = laplace1[:, :middle]
        sliced2 = laplace2[:, middle:]
        combined = np.hstack((sliced1, sliced2))
        combinedPyramid.append(combined)
        cv2.imshow("Gaussian level",combined)
        cv2.waitKey(0)

    return combinedPyramid

def collapse(pyramid):
    collapsedImage = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        collapsedImage = cv2.pyrUp(collapsedImage, dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
        collapsedImage = cv2.add(collapsedImage, pyramid[i])
    return collapsedImage

def task5():
    # Load images
    messi = cv2.imread('data/messi.jpg')
    ronaldo = cv2.imread('data/ronaldo.jpeg')
    ronaldo= cv2.resize(ronaldo, (messi.shape[1], messi.shape[0]))
    gaussMessi = build_gaussian_pyramid(messi)
    gaussRonaldo = build_gaussian_pyramid(ronaldo)

    laplaceMessi = build_laplacian_pyramid(gaussMessi)
    laplaceRonaldo = build_laplacian_pyramid(gaussRonaldo)

    combinedPyramid = blendPyramids(laplaceMessi, laplaceRonaldo)

    collapsedImg = collapse(combinedPyramid)
    cv2.imshow("Result",collapsedImg)
    cv2.waitKey(0)
    


if __name__ == "__main__":
    task2()
    #task3()
    #task4()
    #task5()
