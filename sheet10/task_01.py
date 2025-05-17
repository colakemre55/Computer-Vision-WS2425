import numpy as np
import cv2
import matplotlib.pyplot as plt

# Suppress scientific notation in numpy array printing
np.set_printoptions(suppress=True)


def read_pointcloud(file_name):
    try:
        # Read the file line by line
        points = []
        colors = []
        with open(file_name, 'r') as f:
            # Skip the header line
            next(f)
            
            for line in f:
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Split the line by commas
                values = line.strip().split(',')
                # Convert first 3 values to float (XYZ coordinates)
                point = [float(x) for x in values[:3]]
                # Convert last 3 values to int (RGB colors)
                color = [int(x) for x in values[3:]]
                points.append(point)
                colors.append(color)
        
        return np.array(points), np.array(colors)
    except Exception as e:
        print(f"Error reading point cloud file: {e}")
        return None, None

def decompose_p(P):
    # Convert P to 3x4 matrix if not already
    P = np.array(P).reshape(3, 4)
    
    # Extract the 3x3 matrix M from P
    M = P[:, :3]
    
    # RQ decomposition to get K and R
    K, R = np.linalg.qr(M)
    
    # Ensure K has positive diagonal elements
    D = np.diag(np.sign(np.diag(K)))
    K = K @ D
    R = D @ R
    
    # Extract translation
    t = np.linalg.inv(K) @ P[:, 3]
    
    return K, R, t

def project_to_rgb_depth(points, P, img_width=960, img_height=527, colors=None):
    # Convert points to homogeneous coordinates
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # Project points
    proj_points = (P @ points_h.T).T
    
    # Convert to image coordinates
    proj_points = proj_points[:, :2] / proj_points[:, 2:]
    
    # Create empty images
    rgb_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    depth_image = np.zeros((img_height, img_width), dtype=np.float32)
    
    # Filter valid points
    valid_mask = (proj_points[:, 0] >= 0) & (proj_points[:, 0] < img_width) & \
                (proj_points[:, 1] >= 0) & (proj_points[:, 1] < img_height)
    
    valid_points = proj_points[valid_mask].astype(int)
    valid_depths = points[valid_mask, 2]  # Z coordinate for depth
    valid_colors = colors[valid_mask] if colors is not None else None
    
    # Fill images
    for i in range(len(valid_points)):
        x, y = valid_points[i]
        depth = valid_depths[i]
        
        # Update depth image (keep closest point)
        if depth_image[y, x] == 0 or depth < depth_image[y, x]:
            depth_image[y, x] = depth
            if valid_colors is not None:
                rgb_image[y, x] = valid_colors[i]
            else:
                # Create color based on depth if no colors provided
                color = plt.cm.viridis(depth / np.max(valid_depths))[:3]
                rgb_image[y, x] = (color * 255).astype(np.uint8)
    
    return rgb_image, depth_image

def project_to_cloud(rgb_image, depth_image, K, R, t):
    # Get image coordinates
    height, width = depth_image.shape
    y, x = np.meshgrid(range(height), range(width), indexing='ij')
    
    # Filter valid depth points
    valid_mask = depth_image > 0
    x = x[valid_mask]
    y = y[valid_mask]
    z = depth_image[valid_mask]
    
    # Back-project to 3D
    K_inv = np.linalg.inv(K)
    points_2d = np.vstack((x, y, np.ones_like(x)))
    points_3d = z.reshape(1, -1) * (K_inv @ points_2d)
    
    # Transform to world coordinates
    R_inv = np.linalg.inv(R)
    points_3d = (R_inv @ points_3d).T
    points_3d = points_3d + (R_inv @ (-t)).reshape(1, 3)
    
    # Get colors
    colors = rgb_image[valid_mask]
    
    return points_3d, colors

def task_01():
    # Given projection matrix
    P = np.array([
        [-963.470267, 203.213333, -310.739668, -617520844.064071],
        [206.891280, 234.199183, -898.521276, -1072106164.603584],
        [-0.456213, -0.677763, -0.576634, 3064160.527511]
    ])
    
    # Read point cloud
    points, colors = read_pointcloud('data/point_cloud.txt')
    if points is None:
        return
    
    # Decompose projection matrix
    K, R, t = decompose_p(P)
    
    # Project to images
    rgb_image, depth_image = project_to_rgb_depth(points, P, colors=colors)  # Pass colors to the function
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    plt.imshow(rgb_image)
    plt.title('RGB Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(depth_image, cmap='viridis')
    plt.colorbar(label='Depth')
    plt.title('Depth Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Reconstruct point cloud
    reconstructed_points, reconstructed_colors = project_to_cloud(rgb_image, depth_image, K, R, t)
    
    # Visualize reconstructed point cloud
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reconstructed_points[:, 0], 
              reconstructed_points[:, 1], 
              reconstructed_points[:, 2], 
              c=reconstructed_colors/255, 
              s=1)
    plt.show()

if __name__ == '__main__':
    task_01()