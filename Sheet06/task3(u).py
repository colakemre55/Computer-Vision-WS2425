import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import multivariate_normal

class MOG():
    def __init__(self, height=360, width=640, number_of_gaussians=3, background_thresh=0.6, lr=0.01):
        self.number_of_gaussians = number_of_gaussians
        self.background_thresh = background_thresh
        self.lr = lr
        self.height = height
        self.width = width
        self.mus = np.zeros((self.height, self.width, self.number_of_gaussians, 3))  # means for each gaussian
        self.sigmaSQs = np.zeros((self.height, self.width, self.number_of_gaussians))  # variances for each gaussian
        self.omegas = np.zeros((self.height, self.width, self.number_of_gaussians))  # weights for each gaussian
        self.BG_pivot = np.ones((self.height, self.width))  # pivot for background (all 1 initially)
        
        # Initialize Gaussian parameters with some reasonable values
        for i in range(self.height):
            for j in range(self.width):
                self.mus[i, j] = np.array([[122, 122, 122]] * self.number_of_gaussians)  # initial mean values
                self.sigmaSQs[i, j] = [36.0] * self.number_of_gaussians  # initial variance values
                self.omegas[i, j] = [1.0 / self.number_of_gaussians] * self.number_of_gaussians  # equal weights

    def updateParam(self, img, BG_pivot):
        labels = np.zeros((self.height, self.width))  # Initialize labels (0 for background, 1 for foreground)

        # Iterate over each pixel in the image
        for i in range(self.height):
            for j in range(self.width):
                pixel = img[i, j]
                
                # Compute likelihood for each Gaussian
                likelihoods = np.zeros(self.number_of_gaussians)
                for k in range(self.number_of_gaussians):
                    dist = multivariate_normal(self.mus[i, j, k], np.diag([self.sigmaSQs[i, j, k]]*3))
                    likelihoods[k] = dist.pdf(pixel)
                
                # Sort Gaussians by omega / sigma
                sorted_indices = np.argsort(self.omegas[i, j] / np.sqrt(self.sigmaSQs[i, j]))
                bg_model = sorted_indices[0]  # Background model (first Gaussian in sorted order)

                # Update the background model if the pixel matches it closely enough
                if likelihoods[bg_model] > self.background_thresh:
                    labels[i, j] = 0  # It's considered background
                    M_k = 1  # Assign pixel to background
                else:
                    labels[i, j] = 1  # It's considered foreground
                    M_k = 0  # Assign pixel to foreground

                # Update the parameters of the Gaussians
                for k in range(self.number_of_gaussians):
                    if M_k == 1:  # Background update
                        self.omegas[i, j, k] = (1 - self.lr) * self.omegas[i, j, k] + self.lr
                        self.mus[i, j, k] = (1 - self.lr) * self.mus[i, j, k] + self.lr * pixel
                        self.sigmaSQs[i, j, k] = (1 - self.lr) * self.sigmaSQs[i, j, k] + self.lr * np.sum((pixel - self.mus[i, j, k])**2)
                    else:  # Foreground update (if the pixel doesn't match background)
                        self.omegas[i, j, k] = (1 - self.lr) * self.omegas[i, j, k]

        return labels

# Set the parameters
subtractor = MOG(height=360, width=640, number_of_gaussians=3, background_thresh=0.6, lr=0.01)

# Read the frame
frame = cv2.imread('person.jpg')  

# Get the labels (background/foreground)
labels = subtractor.updateParam(frame, subtractor.BG_pivot)

# Create the background subtracted image (foreground only)
foreground = cv2.bitwise_and(frame, frame, mask=labels.astype(np.uint8))

# Display the result
plt.imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
plt.title("Background Subtracted (Foreground Visible Only)")
plt.show()
