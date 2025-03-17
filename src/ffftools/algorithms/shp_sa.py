import pandas as pd
import numpy as np
import random
from scipy.spatial.distance import euclidean
from tqdm import tqdm

# TODO: both shp_dp and shp_sa night clean-up.
# E.g. creation of the distance matrix could be shared by both

# Create a distance matrix from the DataFrame
def create_distance_matrix(df):
    n = len(df)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                pos_i = (df.iloc[i]['M_Selection_X'], df.iloc[i]['M_Selection_Y'])
                pos_j = (df.iloc[j]['M_Selection_X'], df.iloc[j]['M_Selection_Y'])
                distance_matrix[i][j] = euclidean(pos_i, pos_j)
    
    return distance_matrix

# Calculate the total cost of a given path
def calculate_path_cost(path, distance_matrix):
    total_cost = 0
    for i in range(len(path) - 1):
        total_cost += distance_matrix[path[i]][path[i + 1]]
    return total_cost

# Simulated Annealing algorithm for SHP
def simulated_annealing_shp(df, start_node=None, initial_temp=1000000, cooling_rate=0.9999, max_iterations=3000000, use_tqdm=True):
    distance_matrix = create_distance_matrix(df)
    n = len(df)
    
    # Initialize the start node
    if start_node is None:
        start_node = 0

    # Generate an initial path (start_node -> random permutation of others)
    current_path = [start_node] + random.sample([i for i in range(n) if i != start_node], n - 1)
    current_cost = calculate_path_cost(current_path, distance_matrix)

    # Set the best path and best cost to the current path and cost
    best_path = list(current_path)
    best_cost = current_cost

    # Simulated annealing parameters
    temperature = initial_temp

    iterations = range(max_iterations)
    if use_tqdm:
        iterations = tqdm(iterations, desc="Simulated Annealing Progress")
    
    for iteration in iterations:
        # Create a new candidate path by swapping two random nodes (not including the start node)
        new_path = list(current_path)
        i, j = random.sample(range(1, n), 2)  # Do not swap the start node (index 0)
        new_path[i], new_path[j] = new_path[j], new_path[i]
        
        # Calculate the new path cost
        new_cost = calculate_path_cost(new_path, distance_matrix)
        
        # Decide whether to accept the new path based on the Metropolis criterion
        cost_diff = new_cost - current_cost
        if cost_diff < 0 or random.uniform(0, 1) < np.exp(-cost_diff / temperature):
            current_path = new_path
            current_cost = new_cost

        # Update the best path and cost if the new path is better
        if current_cost < best_cost:
            best_path = current_path
            best_cost = current_cost
        
        # Cool down the temperature
        temperature *= cooling_rate

        # Stop if the temperature is sufficiently low
        if temperature < 1e-10:
            break

    return best_path, best_cost