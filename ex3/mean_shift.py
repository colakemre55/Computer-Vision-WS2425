import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(distance, bandwidth):
    """
    Compute Gaussian kernel weight.
    
    Args:
        distance (float): Distance between points
        bandwidth (float): Kernel bandwidth parameter
    
    Returns:
        float: Kernel weight
    """
    # TODO: Implement Gaussian kernel
    return np.exp(-0.5 * (distance / bandwidth) ** 2)

def mean_shift_step(point, points, bandwidth):
    """
    Perform one step of mean shift.
    
    Args:
        point (np.ndarray): Current point
        points (np.ndarray): All points in the feature space
        bandwidth (float): Kernel bandwidth
    
    Returns:
        np.ndarray: New position after one step
    """
    distances = np.linalg.norm(points - point, axis=1)
    weights = gaussian_kernel(distances, bandwidth)
    
    # Weighted mean shift
    numerator = np.sum(weights[:, np.newaxis] * points, axis=0)
    denominator = np.sum(weights)
    return numerator / denominator
    
    

def mean_shift_segmentation(image, bandwidth, max_iter=50):
    """
    Perform mean shift segmentation on an RGB image.
    
    Args:
        image (np.ndarray): RGB image normalized to [0,1]
        bandwidth (float): Kernel bandwidth
        max_iter (int): Maximum number of iterations
    
    Returns:
        np.ndarray: Segmented image
    """
    # TODO: Implement mean shift segmentation
    # TODO: Implement convergence check
    # TODO: Create final segmentation
    
    # Reshape image to (N, 3) where N = height * width, and 3 for RGB channels
    h, w, c = image.shape
    flat_image = image.reshape(-1, c)
    # Initialize shifted points to the original points
    shifted_points = np.copy(flat_image)
    
    # Perform the mean shift
    for _ in range(max_iter):
        new_shifted_points = np.array([
            mean_shift_step(point, flat_image, bandwidth) for point in shifted_points
        ])
        
        # Convergence check
        if np.allclose(new_shifted_points, shifted_points):
            break
        shifted_points = new_shifted_points
    
    # Reshape the result back to image format
    segmented_image = shifted_points.reshape(h, w, c)
    return segmented_image


def normalize_image(image):
    # Convert the image to float to avoid integer division issues
    image = image.astype(np.float32)
    # Subtract the minimum and divide by the range
    normalized_image = (image - image.min()) / (image.max() - image.min())
    return normalized_image

def visualize(original, segmented, name):
    
    plt.figure(figsize=(10, 5))
    plt.suptitle(name)
    
    # plot original
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis('off')
    
    # plot segmented
    plt.subplot(1, 2, 2)
    plt.imshow(segmented)
    plt.title("Segmented")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(name)

if __name__ == "__main__":
    bandwidth = 0.1 # SET PARAMETER
    
    for i, name in enumerate(['simple', 'gradient', 'concentric']):
        original = np.load(f"test_images/task3/{name}.npy")
        segmented = mean_shift_segmentation(original, bandwidth=bandwidth, max_iter=50)
        segmented_normalized = normalize_image(segmented)
        visualize(original, segmented_normalized, name)