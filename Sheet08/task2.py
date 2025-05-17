import numpy as np
import utils

def ppca(covariance, preservation_ratio=0.9):
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    
    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Calculate cumulative variance ratio
    total_variance = np.sum(eigenvalues)
    cumulative_variance_ratio = np.cumsum(eigenvalues) / total_variance
    
    # Find number of components needed to preserve desired ratio
    num_components = np.argmax(cumulative_variance_ratio >= preservation_ratio) + 1
    
    return eigenvalues[:num_components], eigenvectors[:, :num_components], num_components

def create_covariance_matrix(kpts, mean_shape):
    num_samples = kpts.shape[0]
    # Flatten keypoints to 2*num_points vectors
    flat_kpts = kpts.reshape(num_samples, -1)
    flat_mean = mean_shape.reshape(-1)
    
    # Center the data
    centered_data = flat_kpts - flat_mean
    
    # Calculate covariance matrix
    covariance = np.dot(centered_data.T, centered_data) / (num_samples - 1)
    return covariance

def visualize_impact_of_pcs(mean, pcs, pc_weights):
    num_pcs = pcs.shape[1]
    variations = []
    
    # Generate variations for each PC
    for i in range(num_pcs):
        pc = pcs[:, i].reshape(-1)
        weight = pc_weights[i]
        
        # Create positive and negative variations
        pos_variation = (mean.reshape(-1) + 2 * weight * pc).reshape(1, -1, 2)
        neg_variation = (mean.reshape(-1) - 2 * weight * pc).reshape(1, -1, 2)
        mean_shape = mean.reshape(1, -1, 2)
        
        # Stack variations for visualization
        variations.append(np.vstack([pos_variation, mean_shape, neg_variation]))
        
        # Visualize
        utils.visualize_hands(variations[-1], f"PC {i+1} Variation")

def train_statistical_shape_model(kpts):
    # Calculate mean shape
    mean_shape = np.mean(kpts, axis=0)
    
    # Create covariance matrix
    covariance = create_covariance_matrix(kpts, mean_shape)
    
    # Perform PPCA
    eigenvalues, eigenvectors, num_components = ppca(covariance, 0.9)
    
    # Calculate weights (standard deviations)
    weights = np.sqrt(eigenvalues)
    
    print(f"Number of components preserving 90% variance: {num_components}")
    
    return mean_shape, eigenvectors, weights

def reconstruct_test_shape(kpts, mean, pcs, pc_weights):
    # Flatten test shape and mean
    flat_test = kpts.reshape(-1)
    flat_mean = mean.reshape(-1)
    
    # Center the test shape
    centered_test = flat_test - flat_mean
    
    # Calculate coefficients (h_k values)
    coefficients = np.dot(pcs.T, centered_test)
    
    # Normalize coefficients by weights
    h_values = coefficients / pc_weights
    
    # Reconstruct the shape
    reconstruction = flat_mean + np.dot(pcs, coefficients)
    reconstruction = reconstruction.reshape(-1, 2)
    
    # Calculate RMS error
    rms_error = np.sqrt(np.mean((kpts - reconstruction) ** 2))
    
    print("h_k values:", h_values)
    print("RMS Error:", rms_error)
    
    # Visualize original and reconstructed shapes
    comparison = np.stack([kpts, reconstruction])
    utils.visualize_hands(comparison, "Original (blue) vs Reconstructed (orange)")
    
    return reconstruction, h_values