import cv2
import numpy as np

# Load your image in grayscale
img = cv2.imread('./data/oldtown.jpg')
cv2.imshow("sdds",img)
cv2.waitKey(0)
# Step 1: Generate the Sobel kernels
# For x direction (horizontal edges)
sobel_x = cv2.getDerivKernels(dx=1, dy=0, ksize=7)
sobel_x_kernel = sobel_x[0] @ sobel_x[1].T  # Outer product to get 2D kernel

# For y direction (vertical edges)
sobel_y = cv2.getDerivKernels(dx=0, dy=1, ksize=7)
sobel_y_kernel = sobel_y[0] @ sobel_y[1].T

# Step 2: Define a function to apply a kernel manually
def apply_convolution(image, kernel):
    # Padding the image to keep the output size the same as input
    pad_size = kernel.shape[0] // 2
    padded_img = np.pad(image, pad_size, mode='constant', constant_values=0)
    output = np.zeros_like(image)
    
    # Perform convolution
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_img[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            output[i, j] = np.sum(region * kernel)
    
    # Clip values to fit in uint8 range
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

# Step 3: Apply the Sobel kernels
sobel_x_output = apply_convolution(img, sobel_x_kernel)
sobel_y_output = apply_convolution(img, sobel_y_kernel)

# Step 4: Combine the x and y direction results
sobel_output = cv2.magnitude(sobel_x_output.astype(float), sobel_y_output.astype(float))
sobel_output = np.clip(sobel_output, 0, 255).astype(np.uint8)

# Step 5: Display or save the result
cv2.imshow('Sobel Filtered Image', sobel_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
