import numpy as np
import matplotlib.pyplot as plt

def create_accumulator(image_shape, theta_res=1):
    """
    Create an accumulator array for Hough transform.

    Args:
        image_shape (tuple): Shape of the input image
        theta_res (int): Resolution of theta in degrees

    Returns:
        tuple: (accumulator array, rho values, theta values)
    """
    height, width = image_shape
    diagonal = int(np.ceil(np.sqrt(height**2 + width**2)))
    rho_range = np.arange(-diagonal, diagonal + 1, 1)
    theta_range = np.deg2rad(np.arange(-90, 90, theta_res))
    accumulator = np.zeros((len(rho_range), len(theta_range)), dtype=np.uint64)
    return accumulator, rho_range, theta_range

def hough_transform(edge_image):
    """
    Perform Hough transform for line detection.

    Args:
        edge_image (np.ndarray): Binary edge image

    Returns:
        tuple: (accumulator array, rho values, theta values)
    """
    accumulator, rho_range, theta_range = create_accumulator(edge_image.shape)
    y_idxs, x_idxs = np.nonzero(edge_image)  # Note: y coordinates come first in image arrays
    cos_t = np.cos(theta_range)
    sin_t = np.sin(theta_range)
    num_thetas = len(theta_range)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for theta_idx in range(num_thetas):
            rho = int(round(x * cos_t[theta_idx] + y * sin_t[theta_idx])) + accumulator.shape[0] // 2
            accumulator[rho, theta_idx] += 1

    return accumulator, rho_range, theta_range

def find_peaks(accumulator, n_peaks, threshold=0.5):
    """
    Find peaks in the accumulator array.

    Args:
        accumulator (np.ndarray): Hough transform accumulator array
        n_peaks (int): Number of peaks to find
        threshold (float): Detection threshold

    Returns:
        list: List of (rho_idx, theta_idx) tuples for detected lines
    """
    accumulator_copy = accumulator.copy()
    peaks = []
    threshold_value = threshold * np.max(accumulator)
    neighborhood_size = 10  # Suppression neighborhood size

    for _ in range(n_peaks):
        idx = np.argmax(accumulator_copy)
        rho_idx, theta_idx = np.unravel_index(idx, accumulator_copy.shape)
        if accumulator_copy[rho_idx, theta_idx] < threshold_value:
            break
        peaks.append((rho_idx, theta_idx))

        # Suppress the neighborhood
        rho_min = max(rho_idx - neighborhood_size // 2, 0)
        rho_max = min(rho_idx + neighborhood_size // 2, accumulator.shape[0])
        theta_min = max(theta_idx - neighborhood_size // 2, 0)
        theta_max = min(theta_idx + neighborhood_size // 2, accumulator.shape[1])
        accumulator_copy[rho_min:rho_max, theta_min:theta_max] = 0

    return peaks

def visualize_hough_results(image, accumulator, rho_range, theta_range, peaks, name):
    """
    Visualize the original image, Hough space, and detected lines.
    
    Args:
        image (np.ndarray): Input binary image
        accumulator (np.ndarray): Hough transform accumulator array
        rho_range (np.ndarray): Range of rho values
        theta_range (np.ndarray): Range of theta values
        peaks (list): List of peak coordinates (rho_idx, theta_idx)
        name (str): Path to save the visualization
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot Hough space
    ax2.imshow(accumulator, extent=[np.rad2deg(theta_range[0]), np.rad2deg(theta_range[-1]), 
                                  rho_range[0], rho_range[-1]], 
               aspect='auto', cmap='hot')
    ax2.set_title('Hough Space')
    ax2.set_xlabel('Theta (degrees)')
    ax2.set_ylabel('Rho (pixels)')
    
    # Plot detected lines
    ax3.imshow(image, cmap='gray')
    ax3.set_title('Detected Lines')
    
    height, width = image.shape
    for peak in peaks:
        rho = rho_range[peak[0]]
        theta = theta_range[peak[1]]
        
        # Convert from rho-theta to endpoints
        if np.sin(theta) != 0:
            # y = (-cos(theta)/sin(theta))x + rho/sin(theta)
            x0, x1 = 0, width
            y0 = int(rho/np.sin(theta) - x0*np.cos(theta)/np.sin(theta))
            y1 = int(rho/np.sin(theta) - x1*np.cos(theta)/np.sin(theta))
            ax3.plot([x0, x1], [y0, y1], 'r-')
        else:
            # Vertical line
            ax3.axvline(x=rho, color='r')
    
    ax3.axis('off')
    plt.tight_layout()

    plt.savefig(f"{name}.png")
    plt.close()

if __name__ == "__main__":
    n_peaks = 200  # Set number of peaks to find

    for name in ['parallel', 'box', 'cross', 'noisy']:
        img = np.load(f"test_images/task2/{name}.npy")
        # Ensure the image is binary
        img = (img > 0).astype(np.uint8)
        accumulator, rho_range, theta_range = hough_transform(img)
        peaks = find_peaks(accumulator, n_peaks=n_peaks, threshold=0.5)
        visualize_hough_results(img, accumulator, rho_range, theta_range, peaks, name)
