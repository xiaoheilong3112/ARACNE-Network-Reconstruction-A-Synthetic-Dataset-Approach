#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import networkx as nx

# Set the seed for reproducibility
np.random.seed(0)

# Define the number of genes and interactions
num_genes = 20
num_interactions = 100

# Generate a random network
G = nx.gnm_random_graph(num_genes, num_interactions, seed=0)

# Generate a covariance matrix based on the network structure
cov_matrix = nx.adjacency_matrix(G).toarray() + np.eye(num_genes)

# Generate the expression data
expression_data = np.random.multivariate_normal(np.zeros(num_genes), cov_matrix, size=1000)

# Convert the expression data to a DataFrame
expression_df = pd.DataFrame(expression_data, columns=[f'Gene{i+1}' for i in range(num_genes)])
expression_df.head()

# In[ ]:


from sklearn.metrics import mutual_info_score
from itertools import combinations

def compute_mi(x, y, bins=30):
    # Compute mutual information between two variables x and y
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

def compute_mi_matrix(data, bins=30):
    # Compute mutual information matrix
    N = data.shape[1]
    mi_matrix = np.zeros((N, N))
    for i, j in combinations(range(N), 2):
        mi = compute_mi(data[:, i], data[:, j], bins)
        mi_matrix[i, j] = mi
        mi_matrix[j, i] = mi
    return mi_matrix

def apply_dpi(mi_matrix, dpi_threshold):
    # Apply Data Processing Inequality criterion
    N = mi_matrix.shape[0]
    for i, j, k in combinations(range(N), 3):
        if mi_matrix[i, j] < min(mi_matrix[i, k], mi_matrix[j, k]) - dpi_threshold:
            mi_matrix[i, j] = 0
            mi_matrix[j, i] = 0
    return mi_matrix

# Compute MI matrix
mi_matrix = compute_mi_matrix(expression_data)

# Apply DPI
dpi_threshold = 0.1
mi_matrix_dpi = apply_dpi(mi_matrix, dpi_threshold)

# In[ ]:


import networkx as nx

def construct_network(mi_matrix, threshold):
    # Construct network from MI matrix
    N = mi_matrix.shape[0]
    G = nx.Graph()
    for i, j in combinations(range(N), 2):
        if mi_matrix[i, j] > threshold:
            G.add_edge(i, j)
    return G

# Construct network
threshold = 0.1
G = construct_network(mi_matrix_dpi, threshold)

# Print number of nodes and edges
print('Number of nodes:', G.number_of_nodes())
print('Number of edges:', G.number_of_edges())

# In[ ]:


import matplotlib.pyplot as plt

def plot_degree_distribution(G):
    # Get degree sequence
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)

    # Plot degree distribution
    plt.figure(figsize=(8, 6))
    plt.hist(degree_sequence, bins=20)
    plt.title('Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.show()

# Plot degree distribution of the network
plot_degree_distribution(G)

# In[ ]:


def plot_network(G):
    # Plot network
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=False, node_size=20)
    plt.title('Network')
    plt.show()

# Plot the network
plot_network(G)

# In[ ]:
