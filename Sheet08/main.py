import numpy as np
import time

import utils
import task1
import task2

hands_orig_train = 'data/hands_orig_train.txt.new'
hands_aligned_test = 'data/hands_aligned_test.txt.new'
hands_aligned_train = 'data/hands_aligned_train.txt.new'

def get_keypoints(path):
    data_info = utils.load_data(path)
    kpts = utils.convert_samples_to_xy(data_info['samples'])
    return kpts

def task_1():
    # Load training data
    kpts = get_keypoints(hands_orig_train)
    
    # Visualize original shapes
    print("Visualizing original shapes...")
    utils.visualize_hands(kpts, "Original Shapes")
    
    # Calculate and visualize initial mean
    initial_mean = task1.calculate_mean_shape(kpts)
    utils.visualize_hands(np.expand_dims(initial_mean, axis=0), "Initial Mean Shape")
    
    # Perform Procrustes analysis
    print("\nPerforming Procrustes analysis...")
    aligned_kpts, mean_shape = task1.procrustres_analysis(kpts)
    
    # Visualize results
    print("\nVisualizing aligned shapes...")
    utils.visualize_hands(aligned_kpts, "Aligned Shapes")
    utils.visualize_hands(np.expand_dims(mean_shape, axis=0), "Final Mean Shape")
    
    return aligned_kpts, mean_shape


def task_2_1():
     # Load aligned training data
    kpts = get_keypoints(hands_aligned_train)
    
    # Train statistical shape model
    mean, pcs, pc_weights = task2.train_statistical_shape_model(kpts)
    
    # Visualize the impact of principal components
    task2.visualize_impact_of_pcs(mean, pcs, pc_weights)
    
    return mean, pcs, pc_weights

def task_2_2(mean, pcs, pc_weights):
    # Load test data
    test_kpts = get_keypoints(hands_aligned_test)
    
    # Take first test shape 
    test_shape = test_kpts[0]
    
    # Reconstruct test shape
    reconstructed_shape, h_values = task2.reconstruct_test_shape(
        test_shape, mean, pcs, pc_weights)

if __name__ == '__main__':
    
    print("Running Task 1")
    task_1()

    print("Running Task 2.1")
    mean, pcs, pc_weights = task_2_1()

    print("Running Task 2.2")
    task_2_2(mean, pcs, pc_weights)
