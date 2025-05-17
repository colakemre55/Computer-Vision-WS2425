import numpy as np
import matplotlib.pyplot as plt

def distance_transform(binary_image):
    """
    Compute the distance transform of a binary image using two-pass algorithm.
    
    Args:
        binary_image (np.ndarray): Binary input image (0: background, 1: foreground)
    
    Returns:
        np.ndarray: Distance transform map
    """
    if not isinstance(binary_image, np.ndarray) or binary_image.dtype != bool:
        binary_image = binary_image.astype(bool)
    
    height, width = binary_image.shape
    dist_map = np.zeros_like(binary_image, dtype=float)
    
    # TODO: Initialize distance map (set to inf for foreground, 0 for background)
    dist_map[binary_image == 1] = np.inf  
    
    
    # TODO: Implement forward pass (top-left to bottom-right)
    for x in range(height):
        for y in range(width):
            if binary_image[x, y] == 1:  # Foreground pixel
                # Check neighboring pixels in forward direction
                for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    # Ensure the neighbor is within bounds
                    if 0 <= nx < height and 0 <= ny < width:
                        dist_map[x, y] = min(dist_map[x, y], dist_map[nx, ny] + np.sqrt(dx**2 + dy**2))

    # TODO: Implement backward pass (bottom-right to top-left)
    for x in range(height - 1, -1, -1):
        for y in range(width - 1, -1, -1):
            if binary_image[x, y] == 1:  # Foreground pixel
                # Check neighboring pixels in backward direction
                for dx, dy in [(1, -1), (1, 0), (1, 1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    # Ensure the neighbor is within bounds
                    if 0 <= nx < height and 0 <= ny < width:
                        dist_map[x, y] = min(dist_map[x, y], dist_map[nx, ny] + np.sqrt(dx**2 + dy**2))

    return dist_map

def evaluate_distance_transform(dist_map, ground_truth):
    """
    Evaluate the accuracy of the distance transform.
    
    Args:
        dist_map (np.ndarray): Computed distance transform
        ground_truth (np.ndarray): Ground truth distance transform
    
    Returns:
        float: Mean absolute error between computed and ground truth
    """
    # TODO: Implement evaluation metric
    return np.mean(np.abs(dist_map - ground_truth))

def visualize(img, ground_truth, calculated, name, error):
    
    plt.figure(figsize=(15, 5))
    plt.suptitle(name)
    
    # plot original
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='binary_r')
    plt.title("Binary Image")
    plt.axis('off')
    
    # plot segmented
    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title("Ground Truth Heatmap")
    plt.axis('off')
    
    # plot segmented
    plt.subplot(1, 3, 3)
    plt.imshow(calculated, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(f"Calculated Heatmap: Error = {error}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(name)

if __name__ == "__main__":
    for i, name in enumerate(['square', 'circle', 'triangle']):
        img = np.load(f"test_images/task1/{name}.npy")
        ground_truth = np.load(f"test_images/task1/{name}_ground_truth_dist_map.npy")
        calculated = distance_transform(img)
        error = evaluate_distance_transform(ground_truth, calculated)
        visualize(img, ground_truth, calculated, name, error)