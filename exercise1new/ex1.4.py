import cv2 as cv
import numpy as np

img=cv.imread("bonn.png")
cv.imshow("bonn",img)
cv.waitKey(0)

img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("bonn gray",img_gray)
cv.waitKey(0)

#filter the image using GaussianBlur with sigma=2sqrt2
img_blur=cv.GaussianBlur(img_gray,(3,3),2*np.sqrt(2))

#using filter2d without getgaussiankernel
kernel=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
img_blur2=cv.filter2D(img_gray,-1,kernel)

#using sepfilter2d without getgaussiankernel
kernel_x=np.array([[1,2,1]])/4
kernel_y=np.array([[1],[2],[1]])/4
img_blur3=cv.sepFilter2D(img_gray,-1,kernel_x,kernel_y)

#display the results
cv.imshow("bonn blur 1",img_blur)
cv.waitKey(0)
cv.imshow("bonn blur 2",img_blur2)      
cv.waitKey(0)
cv.imshow("bonn blur 3",img_blur3) 
cv.waitKey(0)


#Compute the absolute pixelwise difference between all pairs 
#and print the maximum pixel error for each pair.
abs_error1=cv.absdiff(img_blur,img_blur2)
max_error1=np.max(abs_error1)
print("Maximum pixel error between img_blur and img_blur2: ",max_error1)
abs_error2=cv.absdiff(img_blur,img_blur3)
max_error2=np.max(abs_error2)
print("Maximum pixel error between img_blur and img_blur3: ",max_error2)
abs_error3=cv.absdiff(img_blur2,img_blur3)
max_error3=np.max(abs_error3)
print("Maximum pixel error between img_blur2 and img_blur3: ",max_error3)


