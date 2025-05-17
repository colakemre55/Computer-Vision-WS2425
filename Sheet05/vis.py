import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Define nodes
nodes = ['a', 'b', 'c', 'd', 'e']

# Add nodes to the graph
G.add_nodes_from(nodes)

# Define edges and their weights (pairwise costs for adjacent nodes)
edges = [
    ('a', 'b', 1),  # cost between a and b
    ('b', 'c', 1),  # cost between b and c
    ('c', 'd', 1),  # cost between c and d
    ('d', 'e', 1)   # cost between d and e
]

# Add edges to the graph
for edge in edges:
    G.add_edge(edge[0], edge[1], weight=edge[2])

# Define labels before expansion
labels_before = {'a': 'β', 'b': 'γ', 'c': 'α', 'd': 'α', 'e': 'β'}

# Define positions for nodes
pos = {
    'a': (0, 1),
    'b': (1, 1),
    'c': (2, 1),
    'd': (3, 1),
    'e': (4, 1)
}

# Draw the graph
plt.figure(figsize=(10, 5))
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=2000, font_size=15, font_weight="bold")

# Add labels to nodes
nx.draw_networkx_labels(G, pos, labels=labels_before, font_size=15, font_color="black")

# Add edge labels
edge_labels = { (u, v): f"{d['weight']}" for u, v, d in G.edges(data=True) }
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)

plt.title("Graph Structure for Alpha Expansion - Initial State")
plt.show()
