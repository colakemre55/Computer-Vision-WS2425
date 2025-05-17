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
    #calculating diagonal of the image for the array sizes
    diagonal = int(np.sqrt(image_shape[0]**2 + image_shape[1]**2))

    rhoValues = np.arange(-diagonal, diagonal + 1, 1) 
    thetaValues = np.deg2rad(np.arange(-90, 90, theta_res)) # theta angle from -90 to 90 ( could be 0 to 180 too)      
    accumulatorArray = np.zeros((len(rhoValues), len(thetaValues)), dtype=np.uint64) # 2d array for (p,theta)

    return accumulatorArray , rhoValues , thetaValues


def hough_transform(edge_image):
    """
    Perform Hough transform for line detection.
    
    Args:
        edge_image (np.ndarray): Binary edge image
    
    Returns:
        tuple: (accumulator array, rho values, theta values)
    """
    yShape, xShape = np.nonzero(edge_image) 
    accArray , rhoValues , thetaValues = create_accumulator(edge_image.shape)
    #diagonal = int(np.sqrt(edge_image.shape[0]**2 + edge_image.shape[1]**2))

    #creating the list for cos/sin values in advance
    cosValues = np.cos(thetaValues) 
    sinValues = np.sin(thetaValues)

    #the voting algorithm
    for i in range(len(xShape)):
        x = xShape[i]
        y = yShape[i]
        for j in range(len(thetaValues)):
            rho = int(round(x * cosValues[j] + y * sinValues[j])) + accArray.shape[0] // 2 #adding this to shift the range 
            accArray[rho, j] += 1 #increment for the voting

    return accArray , rhoValues , thetaValues

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
    accArray = accumulator.copy()
    peaks = []
    threshold_value = threshold * accumulator.max()
    neighborhood_size = 10 #to avoid multiple detections at same line

    for _ in range(n_peaks):
        idx = np.argmax(accArray)
        rho_idx, theta_idx = np.unravel_index(idx, accArray.shape)
        if accArray[rho_idx, theta_idx] < threshold_value:
            break
        peaks.append((rho_idx, theta_idx))

        rhoMin = max(rho_idx - neighborhood_size // 2, 0)
        rhoMax = min(rho_idx + neighborhood_size // 2, accumulator.shape[0])
        thetaMin = max(theta_idx - neighborhood_size // 2, 0)
        thetaMax = min(theta_idx + neighborhood_size // 2, accumulator.shape[1])
        accArray[rhoMin:rhoMax, thetaMin:thetaMax] = 0

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
    n_peaks = 5 # SET PARAMETER
    
    for i, name in enumerate(['parallel', 'box', 'cross', 'noisy']):
        img = np.load(f"test_images/task2/{name}.npy")        
        accumulator, rho_range, theta_range = hough_transform(img)
        peaks = find_peaks(accumulator, n_peaks=n_peaks, threshold=0.5)
        visualize_hough_results(img, accumulator, rho_range, theta_range, peaks, name)