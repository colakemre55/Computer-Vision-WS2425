import numpy as np
import cv2
import maxflow
import matplotlib.pyplot as plt


def plot_result(img, denoised_img, rho, pairwise_same, pairwise_diff, figsize=(15, 7)):
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    fig.suptitle(f"Result for rho={rho} pairwise_cost_same={pairwise_same} and pairwise_cost_diff={pairwise_diff} ", fontsize=16)
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Noisy Image")
    axes[0].tick_params(labelbottom=False, labelleft=False)
    
    axes[1].imshow(denoised_img, cmap='gray')
    axes[1].set_title("Denoised Image")
    axes[1].tick_params(labelbottom=False, labelleft=False)

    plt.savefig(f"result_question4.png")
    plt.show()



def binary_img_denoiser(img, rho, pairwise_same, pairwise_diff):
    # TODO: Change to binary image
    # Convert to binary (0 and 1)
    img = (img > 127).astype(np.uint8)

    # Ensure the input image is binary
    assert np.array_equal(np.unique(img), [0, 1]), "Input image must be binary (0 or 1)."

    # TODO: Define Graph and add pixels as nodes
    # Get image dimensions
    height, width = img.shape

    graph = maxflow.Graph[int](height * width, height * width)
    nodes = graph.add_nodes(height * width)

    # TODO: Add unary costs to the graph
    # Helper function to convert 2D pixel (x, y) to 1D index
    def pixel_to_node(x, y):
        return x * width + y

    for x in range(height):
        for y in range(width):
            node_id = pixel_to_node(x, y)
            pixel_value = img[x, y]

            # Unary costs using Bernoulli probabilities
            cost_to_source = -np.log(rho if pixel_value == 0 else 1 - rho)
            cost_to_sink = -np.log(1 - rho if pixel_value == 0 else rho)

            graph.add_tedge(node_id, cost_to_source, cost_to_sink)


    # TODO: Add pairwise costs to the graph
    for x in range(height):
        for y in range(width):
            node_id = pixel_to_node(x, y)

            # Check 4 neighbors: right, down
            for dx, dy in [(0, 1), (1, 0)]:
                nx, ny = x + dx, y + dy
                if nx < height and ny < width:
                    neighbor_id = pixel_to_node(nx, ny)

                    # Add pairwise costs
                    graph.add_edge(node_id, neighbor_id, pairwise_same, pairwise_diff)


    # TODO: Perform Maxflow optimization
    flow = graph.maxflow()

    # TODO: Extract labels and construct the denoised image
    denoised_img = np.zeros_like(img, dtype=np.uint8)
    for x in range(height):
        for y in range(width):
            node_id = pixel_to_node(x, y)
            denoised_img[x, y] = graph.get_segment(node_id)

    return denoised_img
    
    


if __name__ == "__main__":
    # TODO: Read the noisy binary image
    image = cv2.imread('./images/noisy_binary.png', cv2.IMREAD_GRAYSCALE)

    rho = 0.3  # Parameter for unary cost
    pairwise_same = 1  # Cost for same labels
    pairwise_diff = 2  # Cost for different labels

    result = binary_img_denoiser(image, rho=rho, pairwise_same=pairwise_same, pairwise_diff=pairwise_diff)

    plot_result(image, 
                result,
                rho=rho, 
                pairwise_same=pairwise_same, 
                pairwise_diff=pairwise_diff)