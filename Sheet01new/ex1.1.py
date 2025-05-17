import cv2 as cv
import numpy as np
import random as rand
import time as t

img=cv.imread("bonn.png")
img_gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow("ex", img)
#cv.waitKey(0)
height, width= img_gray.shape
#1.1Compute and display the integral image without using the function integral.

integrals = np.zeros((height, width), dtype=np.int32)

for i in range(height):
    for j in range(width):
        pixel_value=img_gray[i,j]
        if i==0 and j==0:
            integrals[i][j]=pixel_value
        elif i==0 and j!=0:
            integrals[i][j]=pixel_value+integrals[i][j-1]
        elif j==0 and i!=0:
            integrals[i][j]=pixel_value+integrals[i-1][j]
        else:
            integrals[i][j]=pixel_value+integrals[i-1][j]+integrals[i][j-1]-integrals[i-1][j-1]
        
cv.imshow("Integral Image", integrals.astype(np.uint8))
cv.waitKey(0)

int= cv.integral(img_gray)

# OpenCV's integral image has an extra row and column
int_opencv_sliced = int[1:, 1:]

# Compare the two integral images
equal = np.array_equal(integrals, int_opencv_sliced)
print("Integral images are equal:", equal)

#1.2 Computation of the mean gray value 
#1.2.1 Summing up each pixel
sum= np.int32(0)
for i in range(height):
    for j in range(width):
        sum+=img_gray[i,j]

mean_val=sum/(height*width)
print("Mean value of the image is:", mean_val)

#1.2.2 Using the integral image with integral funtion
int = cv.integral(img_gray)
mean_val_int = int[height, width] / (height * width)

#check whether they are same
print("Mean values are equal:", np.array_equal(mean_val, mean_val_int))

#1.2.3 Using the integral image with my own function
def compute_integral_image(image):
    height, width= image.shape
    integrals = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            pixel_value=image[i,j]
            if i==0 and j==0:
                integrals[i][j]=pixel_value
            elif i==0 and j!=0:
                integrals[i][j]=pixel_value+integrals[i][j-1]
            elif j==0 and i!=0:
                integrals[i][j]=pixel_value+integrals[i-1][j]
            else:
                integrals[i][j]=pixel_value+integrals[i-1][j]+integrals[i][j-1]-integrals[i-1][j-1]
            
    return integrals

integrals = compute_integral_image(img_gray)
mean_val_integral = integrals[height - 1, width - 1] / (height * width)

# Check whether they are same
print("Mean values are equal:", np.array_equal(mean_val, mean_val_integral))

#1.3 Select 7 random squares of size 100 × 100 within the image and compute the mean gray value using the three versions.
squares=[]

for _ in range(7):
    # Randomly choose the top-left corner of the square
    x = rand.randint(0, height - 100)
    y = rand.randint(0, width - 100)
    
    # Extract the 100x100 square and store it in the list
    square = img_gray[x:x+100, y:y+100]
    squares.append(square)

# Compute the mean gray value for each square using the three versions and output the time for each
for i, square in enumerate(squares):
    start_time=t.time()
    mean_val_sum = np.sum(square) / (100 * 100)
    end_time=t.time()
    print(f"Square {i+1} - Mean value (summing up each pixel): {mean_val_sum}, Time: {end_time-start_time} seconds")
    start_time=t.time()
    mean_val_int = cv.integral(square)[100, 100] / (100 * 100)
    end_time=t.time()
    print(f"Square {i+1} - Mean value (using OpenCV's integral function): {mean_val_int}, Time: {end_time-start_time} seconds")
    start_time=t.time()
    mean_val_own = compute_integral_image(square)[99, 99] / (100 * 100)
    end_time=t.time()
    print(f"Square {i+1} - Mean value (using my own integral function): {mean_val_own}, Time: {end_time-start_time} seconds")
    break

