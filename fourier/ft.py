import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
image = cv2.imread('vertwood.jpg', cv2.IMREAD_GRAYSCALE)

# Compute the 2D Fourier Transform
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)  # Shift zero freq to center
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # Avoid log(0)

# Display original image and its magnitude spectrum
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Fourier Transform (Magnitude Spectrum)')
plt.axis('off')

plt.tight_layout()
plt.show()
