import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Input binary image
binary_image = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [1, 1, 1, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 1, 1, 1],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0]
])

# Replace 1s (foreground) with 0 and 0s (background) with infinity
distance_image = np.where(binary_image == 1, 0, np.inf)

# Dimensions of the image
rows, cols = distance_image.shape

# Forward pass
for i in range(rows):
    for j in range(cols):
        if distance_image[i, j] != 0:  # Skip foreground pixels
            min_dist = distance_image[i, j]
            # Check top neighbor
            if i > 0:
                min_dist = min(min_dist, distance_image[i-1, j] + 1)
            # Check top-left neighbor
            if i > 0 and j > 0:
                min_dist = min(min_dist, distance_image[i-1, j-1] + 1)
            # Check left neighbor
            if j > 0:
                min_dist = min(min_dist, distance_image[i, j-1] + 1)
            # Check top-right neighbor
            if i > 0 and j < cols - 1:
                min_dist = min(min_dist, distance_image[i-1, j+1] + 1)
            distance_image[i, j] = min_dist

# Backward pass
for i in range(rows - 1, -1, -1):
    for j in range(cols - 1, -1, -1):
        if distance_image[i, j] != 0:  # Skip foreground pixels
            min_dist = distance_image[i, j]
            # Check bottom neighbor
            if i < rows - 1:
                min_dist = min(min_dist, distance_image[i+1, j] + 1)
            # Check bottom-right neighbor
            if i < rows - 1 and j < cols - 1:
                min_dist = min(min_dist, distance_image[i+1, j+1] + 1)
            # Check right neighbor
            if j < cols - 1:
                min_dist = min(min_dist, distance_image[i, j+1] + 1)
            # Check bottom-left neighbor
            if i < rows - 1 and j > 0:
                min_dist = min(min_dist, distance_image[i+1, j-1] + 1)
            distance_image[i, j] = min_dist

# Replace infinities with a high value for display purposes
max_distance = np.nanmax(distance_image[np.isfinite(distance_image)])
distance_image[np.isinf(distance_image)] = max_distance + 1

# Display the distance transform
plt.imshow(distance_image, cmap='gray', interpolation='nearest')
plt.title('2D Distance Transform')
plt.colorbar(label='Distance')
plt.show()
