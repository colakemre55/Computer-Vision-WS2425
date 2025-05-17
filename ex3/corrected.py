import numpy as np
import matplotlib.pyplot as plt
import cv2

def create_accumulator(image_shape, theta_res=1):
    """
    Create an accumulator array for Hough transform.
    
    Args:
        image_shape (tuple): Shape of the input image (height, width)
        theta_res (int): Resolution of theta in degrees
    
    Returns:
        tuple: (accumulator array, rho values, theta values)
    """
    diagonal = int(np.sqrt(image_shape[0]**2 + image_shape[1]**2))
    rhoValues = np.linspace(-diagonal, diagonal, 2 * diagonal)
    thetaValues = np.deg2rad(np.arange(-90, 90, theta_res))
    accumulatorArray = np.zeros((len(rhoValues), len(thetaValues)), dtype=np.int32)
    return accumulatorArray, rhoValues, thetaValues

def hough_transform(edge_image):
    """
    Perform Hough transform for line detection.
    
    Args:
        edge_image (np.ndarray): Binary edge image
    
    Returns:
        tuple: (accumulator array, rho values, theta values)
    """
    accumulator, rhoValues, thetaValues = create_accumulator(edge_image.shape)
    y_idxs, x_idxs = np.nonzero(edge_image)  # Get the coordinates of edge points
    
    diagonal = len(rhoValues) // 2  # Center index in rhoValues

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for theta_idx, theta in enumerate(thetaValues):
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            # Correct the rho_index calculation
            rho_idx = np.clip(int(rho + diagonal), 0, len(rhoValues) - 1)
            if 0 <= rho_idx < len(rhoValues):
                accumulator[rho_idx, theta_idx] += 1

    return accumulator, rhoValues, thetaValues

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
    threshold_value = threshold * accumulator.max()
    peaksList = []
    for i in range(accumulator.shape[0]):
        for j in range(accumulator.shape[1]):
            if accumulator[i, j] >= threshold_value:
                peaksList.append((i, j))  # Append as a tuple

    return peaksList[:n_peaks]

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
    n_peaks = 5 # SET PARAMETER
    
    for i, name in enumerate(['parallel', 'box', 'cross', 'noisy']):
        img = np.load(f"test_images/task2/{name}.npy")        
        #plt.imshow(img, cmap='gray')
        #plt.show()
        #hough_transform(img)
        accumulator, rho_range, theta_range = hough_transform(img)
        peaks = find_peaks(accumulator, n_peaks=n_peaks, threshold=0.5)
        visualize_hough_results(img, accumulator, rho_range, theta_range, peaks, name)
