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
    diagonal = int(np.sqrt(image_shape[0]**2 + image_shape[1]**2))
    rho_range = np.arange(-diagonal, diagonal + 1, 1)
    theta_range = np.deg2rad(np.arange(-90, 90, theta_res))
    accumulator = np.zeros((len(rho_range), len(theta_range)), dtype=np.int32)
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
    y_idxs, x_idxs = np.nonzero(edge_image)
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for theta_idx, theta in enumerate(theta_range):
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            rho_idx = np.argwhere(rho_range == rho)[0][0]
            accumulator[rho_idx, theta_idx] += 1
    return accumulator, rho_range, theta_range

def find_peaks(accumulator, n_peaks, threshold=0.5):
    """
    Find peaks in the accumulator array.
    
    Args:
        accumulator (np.ndarray): Hough transform accumulator array
        n_peaks (int): Number of peaks to find
        threshold (float): Detection threshold
    
    Returns:
        list: List of (rho, theta) pairs for detected lines
    """
    peaks = []
    threshold_value = threshold * np.max(accumulator)
    for _ in range(n_peaks):
        max_idx = np.argmax(accumulator)
        rho_idx, theta_idx = np.unravel_index(max_idx, accumulator.shape)
        if accumulator[rho_idx, theta_idx] >= threshold_value:
            peaks.append((rho_idx, theta_idx))
            accumulator[rho_idx-5:rho_idx+5, theta_idx-5:theta_idx+5] = 0  # Suppress surrounding peaks
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
    n_peaks = 5  # Set number of peaks to find

    for name in ['parallel', 'box', 'cross', 'noisy']:
        img = np.load(f"test_images/task2/{name}.npy")        
        accumulator, rho_range, theta_range = hough_transform(img)
        peaks = find_peaks(accumulator, n_peaks=n_peaks, threshold=0.5)
        visualize_hough_results(img, accumulator, rho_range, theta_range, peaks, name)
