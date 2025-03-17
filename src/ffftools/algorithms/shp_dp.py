import math
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

# TODO: both shp_dp and shp_sa night clean-up.
# E.g. creation of the distance matrix could be shared by both

def calculate_total_distance(df, path):
    """
    Calculate the total distance of the path based on the coordinates in the DataFrame.

    Parameters:
    - df: fff-compatible pd.DataFrame containing the coordinates with columns 'M_Selection_X' and 'M_Selection_Y'.
    - path: list of integers representing the indices of the nodes in the order they are visited.

    Returns:
    - total_distance: float representing the total distance of the path.
    """
    total_distance = 0.0
    
    for i in range(len(path) - 1):
        start_index = path[i]
        end_index = path[i + 1]
        
        start_pos = (df.iloc[start_index]['M_Selection_X'], df.iloc[start_index]['M_Selection_Y'])
        end_pos = (df.iloc[end_index]['M_Selection_X'], df.iloc[end_index]['M_Selection_Y'])
        
        total_distance += euclidean(start_pos, end_pos)
    
    return total_distance


# Create a fully connected distance matrix (graph) from the DataFrame
def create_distance_graph(df):
    n = len(df)
    graph = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                graph[i][j] = euclidean([df.iloc[i]['M_Selection_X'], df.iloc[i]['M_Selection_Y']], 
                                         [df.iloc[j]['M_Selection_X'], df.iloc[j]['M_Selection_Y']])
    return graph

# Implement the Dynamic Programming (Held-Karp) algorithm for SHP
def shortest_hamiltonian_path(df, start_node=None):
    n = len(df)
    graph = create_distance_graph(df)
    
    # Initialize DP table: dp[mask][v] = minimum cost to visit nodes in 'mask' ending at 'v'
    dp = np.full((1 << n, n), float('inf'))
    parent = np.full((1 << n, n), -1)  # To reconstruct the path
    
    if start_node is not None:
        # Fixed start node case: Only initialize the DP for the start_node
        dp[1 << start_node][start_node] = 0
    else:
        # Flexible start node case: Initialize the DP for all possible start nodes
        for i in range(n):
            dp[1 << i][i] = 0

    # DP state transitions
    for mask in range(1 << n):
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            for v in range(n):
                if mask & (1 << v) or graph[u][v] == 0:
                    continue
                new_cost = dp[mask][u] + graph[u][v]
                if new_cost < dp[mask | (1 << v)][v]:
                    dp[mask | (1 << v)][v] = new_cost
                    parent[mask | (1 << v)][v] = u

    # Find the minimum cost path:
    min_cost = float('inf')
    end_node = -1
    final_mask = (1 << n) - 1

    if start_node is not None:
        for v in range(n):
            if v != start_node and dp[final_mask][v] < min_cost:
                min_cost = dp[final_mask][v]
                end_node = v
    else:
        for v in range(n):
            if dp[final_mask][v] < min_cost:
                min_cost = dp[final_mask][v]
                end_node = v

    # Reconstruct the path
    path = []
    current_node = end_node
    current_mask = final_mask

    while current_node != -1:
        path.append(current_node)
        next_node = parent[current_mask][current_node]
        current_mask ^= (1 << current_node)  # Remove current node from mask
        current_node = next_node

    path.reverse()  # Reverse to get the correct order

    return min_cost, path