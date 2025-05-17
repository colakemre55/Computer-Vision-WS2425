import numpy as np
import cv2
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the image
superPixel = cv2.imread("./superpixels.png")  # Update path as needed
if superPixel is None:
    raise FileNotFoundError("Image not found. Check the file path.")
x, y, _ = superPixel.shape

# Parameters
sigma_color_values = [10, 20, 30]
sigma_space_values = [5, 10, 15]
radius_values = [5, 10]
k = 5  # Number of clusters

# Function definitions
def spatial_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def color_distance(c1, c2):
    return np.sqrt(np.sum((c1 - c2) ** 2))

def evaluate_segmentation(segmentation):
    """Dummy evaluation function; replace with an actual metric."""
    return np.std(segmentation)

# Track the best segmentation and parameters
best_score = float('inf')
best_segmentation = None
best_params = None
results = []

for sigma_color in sigma_color_values:
    for sigma_space in sigma_space_values:
        for radius in radius_values:
            print(f"Testing parameters: sigma_color={sigma_color}, sigma_space={sigma_space}, radius={radius}")
            
            # Compute weights using sparse matrix
            weights = lil_matrix((x * y, x * y))
            for i in range(x):
                for j in range(y):
                    idx1 = i * y + j
                    for di in range(-radius, radius + 1):
                        for dj in range(-radius, radius + 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < x and 0 <= nj < y:
                                idx2 = ni * y + nj
                                dist_space = spatial_distance((i, j), (ni, nj))
                                if dist_space <= radius:
                                    color1 = superPixel[i, j]
                                    color2 = superPixel[ni, nj]
                                    dist_color = color_distance(color1, color2)
                                    weight = np.exp(-dist_color ** 2 / (2 * sigma_color ** 2)) * \
                                             np.exp(-dist_space ** 2 / (2 * sigma_space ** 2))
                                    weights[idx1, idx2] = weight

            weights = weights + weights.T  # Symmetrize weights

            # Compute Laplacian
            D = np.array(weights.sum(axis=1)).flatten()
            D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-10))
            L_sym = np.eye(x * y) - D_inv_sqrt @ weights.toarray() @ D_inv_sqrt

            # Compute eigenvectors
            L_sym_sparse = csr_matrix(L_sym)
            eigenvalues, eigenvectors = eigsh(L_sym_sparse, k=k, which="SM")

            # Normalize eigenvector rows
            U_normalized = eigenvectors / np.linalg.norm(eigenvectors, axis=1, keepdims=True)

            # Perform k-means clustering
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
            labels = kmeans.fit_predict(U_normalized)
            segmentation = labels.reshape((x, y))

            # Evaluate segmentation
            score = evaluate_segmentation(segmentation)
            results.append((sigma_color, sigma_space, radius, score))

            # Track the best result
            if score < best_score:
                best_score = score
                best_segmentation = segmentation
                best_params = (sigma_color, sigma_space, radius)

# Output the best parameters
print(f"Best parameters: sigma_color={best_params[0]}, sigma_space={best_params[1]}, radius={best_params[2]}")

# Display the best segmentation
segmented_image = (best_segmentation * (255 // best_segmentation.max())).astype(np.uint8)
cv2.imshow('Best Segmentation', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Analyze results
df = pd.DataFrame(results, columns=["sigma_color", "sigma_space", "radius", "score"])
for radius in radius_values:
    heatmap_data = df[df["radius"] == radius].pivot(index="sigma_color", columns="sigma_space", values="score")
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"Scores for Radius={radius}")
    plt.show()
