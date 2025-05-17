import numpy as np
import utils

def calculate_mean_shape(kpts):
    return np.mean(kpts, axis=0)

def normalize_shape(shape):
    # Center
    centroid = np.mean(shape, axis=0)
    centered_shape = shape - centroid
    
    # Scale to unit size
    scale = np.sqrt(np.sum(centered_shape ** 2))
    if scale > 0:
        normalized_shape = centered_shape / scale
    else:
        normalized_shape = centered_shape
        
    return normalized_shape

def get_rotation_matrix(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]])

def align_shapes(shape, reference):
    # Normalize both shapes
    shape_norm = normalize_shape(shape)
    ref_norm = normalize_shape(reference)
    
    # Calculate optimal rotation
    H = np.dot(shape_norm.T, ref_norm)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    
    # Check for reflection
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = np.dot(Vt.T, U.T)
    
    # Calculate scale
    scale = np.sqrt(np.sum(ref_norm ** 2) / np.sum(shape_norm ** 2))
    
    # Transform shape
    aligned = scale * np.dot(shape_norm, R)
    
    # Translate to reference centroid
    ref_centroid = np.mean(reference, axis=0)
    aligned += ref_centroid
    
    return aligned

def procrustres_analysis_step(kpts, reference_mean):
    num_samples = kpts.shape[0]
    aligned_kpts = np.zeros_like(kpts)
    
    for i in range(num_samples):
        aligned_kpts[i] = align_shapes(kpts[i], reference_mean)
    
    return aligned_kpts

def compute_avg_error(kpts, mean_shape):
    errors = np.sqrt(np.mean((kpts - mean_shape[np.newaxis, :, :]) ** 2, axis=(1, 2)))
    return np.mean(errors)

def procrustres_analysis(kpts, max_iter=1000, min_error=1e-5):
    aligned_kpts = kpts.copy()
    prev_error = float('inf')
    
    for iter in range(max_iter):
        # Calculate mean shape
        mean_shape = calculate_mean_shape(aligned_kpts)
        
        # Align shapes to mean shape
        aligned_kpts = procrustres_analysis_step(aligned_kpts, mean_shape)
        
        # Calculate error
        current_error = compute_avg_error(aligned_kpts, mean_shape)
        
        # Check convergence
        if abs(prev_error - current_error) < min_error:
            print(f"Converged after {iter + 1} iterations")
            break
            
        prev_error = current_error
        
    mean_shape = calculate_mean_shape(aligned_kpts)
    return aligned_kpts, mean_shape