import numpy as np
import cv2
import maxflow
import matplotlib.pyplot as plt


def plot_result(img, denoised_img, rho, pairwise_cost_type, figsize=(15, 7)):
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    fig.suptitle(f"Result for rho={rho} and pairwise_cost_type={pairwise_cost_type} ", fontsize=16)
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Noisy Image")
    axes[0].tick_params(labelbottom=False, labelleft=False)
    
    axes[1].imshow(denoised_img, cmap='gray')
    axes[1].set_title("Denoised Image")
    axes[1].tick_params(labelbottom=False, labelleft=False)

    plt.savefig("result_question5.png")
    plt.show()


def compute_pairwise_cost(wm, wn, cost_type, k1=10, k2=1):
    if cost_type == "quadratic":
        return (wm - wn) ** 2
    elif cost_type == "truncated_quadratic":
        return min(k1, k2 * (wm - wn) ** 2)
    elif cost_type == "potts":
        return 1 if wm != wn else 0
    else:
        raise ValueError("Invalid pairwise cost type")


def get_neighbors(img, i, j):
    neighbors = []
    h, w = img.shape
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  #up, down, left, right
        ni, nj = i + di, j + dj
        if 0 <= ni < h and 0 <= nj < w:
            neighbors.append((ni, nj))
    return neighbors


def alpha_expansion(I, label, rho, pairwise_cost_type):
    """Perform alpha expansion for a given label."""
    h, w = I.shape
    graph = maxflow.Graph[int]()
    node_ids = graph.add_grid_nodes((h, w))
    
    
    for i in range(h): #unary costs
        for j in range(w):
            current_label = I[i, j]
            unary_cost_label = abs(current_label - label) * rho
            unary_cost_keep = abs(current_label - current_label) * (1 - rho)
            
            graph.add_tedge(node_ids[i, j], unary_cost_label, unary_cost_keep)
    

    for i in range(h): #pairwise costs
        for j in range(w):
            for ni, nj in get_neighbors(I, i, j):
                cost = compute_pairwise_cost(I[i, j], I[ni, nj], pairwise_cost_type)
                graph.add_edge(node_ids[i, j], node_ids[ni, nj], cost, cost)
    
    graph.maxflow()
    
    
    new_labels = np.zeros_like(I, dtype=np.uint8) #extract labels
    for i in range(h):
        for j in range(w):
            new_labels[i, j] = label if graph.get_segment(node_ids[i, j]) == 0 else I[i, j]
    
    return new_labels


def denoise_grayscale_image(img, rho, pairwise_cost_type):
    """Denoise grayscale image using Alpha Expansion."""
    labels = [0, 128, 255]  #gray levels
    denoised_img = img.copy()
    for label in labels:
        denoised_img = alpha_expansion(denoised_img, label, rho, pairwise_cost_type)
    return denoised_img


if __name__ == "__main__":
    image = cv2.imread('./images/noisy_grayscale.png', cv2.IMREAD_GRAYSCALE)
    pairwise_cost_type = "quadratic"  # choosing the formü

    rho = 1  # Unary cost weight

    # Denoise image
    result = denoise_grayscale_image(image, rho=rho, pairwise_cost_type=pairwise_cost_type)
    
    # Plot and save results
    plot_result(image, 
                result,
                rho=rho, 
                pairwise_cost_type=pairwise_cost_type,
                figsize=(10, 5))
