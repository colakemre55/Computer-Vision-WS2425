import cv2 as cv
import numpy as np

img= cv.imread('bonn.png')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#1.Filter the images using the two 2D filter kernels given blow
filter1= np.array([[0.0113, 0.0838, 0.0113], [0.0838, 0.6193, 0.0838], [0.0113, 0.0838, 0.0113]])
img_filtered_1 = cv.filter2D(img_gray, -1, filter1)

filter2=np.array([[-1.7497,0.3426,1.1530,-0.2524,0.9813],[0.5142 ,0.2211 ,-1.0700,-0.1894,0.2550],[-0.4580, 0.4351 ,-0.5835, 0.8168, 0.6727],[0.1044,-0.5312, 1.0297, -0.4381,-1.1183],[1.6189,1.5416,-0.2518,-0.8424,0.1845]])
img_filtered_2 = cv.filter2D(img_gray, -1, filter2)

#2.Use the class SVD of OpenCV to separate each kernel.
w1, u1, vt1 = cv.SVDecomp(filter1)

# Rank-1 approximation for the first kernel
filter1_approx = w1[0, 0] * (u1[:, 0].reshape(-1, 1) @ vt1[0, :].reshape(1, -1))

img_filtered_1_approx = cv.filter2D(img_gray, -1, filter1_approx)

cv.imshow('Filtered Image 1', img_filtered_1)
cv.waitKey(0)
cv.imshow('Filtered Image 1 - Approx', img_filtered_1_approx)
cv.waitKey(0)

w2, u2, vt2 = cv.SVDecomp(filter2)

# Rank-2 approximation for the second kernel (we add two 2D matrices here)
filter2_approx = (w2[0, 0] * (u2[:, 0].reshape(-1, 1) @ vt2[0, :].reshape(1, -1)) +
                  w2[1, 0] * (u2[:, 1].reshape(-1, 1) @ vt2[1, :].reshape(1, -1)))
img_filtered_2_approx = cv.filter2D(img_gray, -1, filter2_approx)

cv.imshow('Filtered Image 2', img_filtered_2)   
cv.waitKey(0)
cv.imshow('Filtered Image 2 - Approx', img_filtered_2_approx)
cv.waitKey(0)

#3.Compute the absolute pixel-wise difference between the results of (a) and (b), and print the maximum pixel error
abs_diff_1 = cv.absdiff(img_filtered_1, img_filtered_1_approx)
max_pixel_error_1 = np.max(abs_diff_1)
print(f'Maximum pixel error for Filter 1: {max_pixel_error_1}')
abs_diff_2 = cv.absdiff(img_filtered_2, img_filtered_2_approx)
max_pixel_error_2 = np.max(abs_diff_2)
print(f'Maximum pixel error for Filter 2: {max_pixel_error_2}')