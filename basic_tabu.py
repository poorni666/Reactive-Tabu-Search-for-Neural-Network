
import random
import time
import math


random.seed(42)  # For reproducibility
n = 30           # Number of facilities and locations
max_iterations = 100
list_size = 10   # Size of tabu list


# current_perm: index is location, value is facility
current_perm = random.sample(range(n), n)
# Stores the last time a specific facility was assigned to a specific location
latest_occupation = [[float('-inf') for _ in range(n)] for _ in range(n)]

#input
flow = [[random.random() for _ in range(n)] for _ in range(n)]
distance = [[random.random() for _ in range(n)] for _ in range(n)]

def calculate_total_cost(perm, flow_mat, dist_mat, size):
    """
    Calculating the cost of a solution by summing the product of flow 
    and distance for all pairs of facilities
    """
    total_cost = 0
    for i in range(size):
        for j in range(size):
            total_cost += flow_mat[perm[i]][perm[j]] * dist_mat[i][j]
    return total_cost

def move_change(r, s, current_perm, flow, distance, n):
    facility_r = current_perm[r]
    facility_s = current_perm[s]
    delta = 0
    for k in range(n):
        if k != r and k != s:
            facility_k = current_perm[k]
            delta += (flow[facility_r][facility_k] - flow[facility_s][facility_k]) * \
                     (distance[s][k] - distance[r][k])
            delta += (flow[facility_k][facility_r] - flow[facility_k][facility_s]) * \
                     (distance[k][s] - distance[k][r])
    delta += (flow[facility_r][facility_s] - flow[facility_s][facility_r]) * \
             (distance[s][r] - distance[r][s])
    return delta

def make_tabu(r, s, perm, occupation_matrix, time_step):
    """
    Record the last occupation of the facilities of two units that are being swapped.
    Updates the tabu list to prevent both facilities from returning 
    to their original locations immediately
    """
    facility_r = perm[r]
    facility_s = perm[s]
    
    # Facility r is moving to location s
    occupation_matrix[facility_r][s] = time_step
    # Facility s is moving to location r
    occupation_matrix[facility_s][r] = time_step

def is_tabu(r, s, perm, occupation_matrix, tabu_tenure, time_step):
    """
    Forbids swap if it goes to previous assignments to avoid cycling.
    Returns True if the move is Tabu.
    """
    facility_r = perm[r]
    facility_s = perm[s]

    # to check assigning facility_r to location s is tabu
    tabu_r = occupation_matrix[facility_r][s] >= time_step - tabu_tenure
    # to check assigning facility_s to location r is tabu
    tabu_s = occupation_matrix[facility_s][r] >= time_step - tabu_tenure

    # If EITHER assignment is tabu, the move is tabu.
    return tabu_r or tabu_s

def aspiration(current_f, delta, best_so_far):
    """
    Allows forbidden moves if they result in a solution better than the global best.
    """
    return current_f+delta < best_so_far



if __name__ == "__main__":
    # Calculate initial cost 
    current_cost = calculate_total_cost(current_perm, flow, distance, n)
    
    # Initialize Best Solution so far
    best_cost = current_cost
    best_so_far = current_perm.copy()

    print(f"Starting Search...")
    print(f"Initial Cost: {current_cost:.4f}")
    print(f"Search Space Size: {math.factorial(n):.2e}")
    print(f"Neighborhood Size: {n * (n - 1) / 2}")
    print("-" * 30)

    for t in range(max_iterations):
        r, s = random.sample(range(n), 2)
        delta = move_change(r, s, current_perm, flow, distance, n)
    
        if not is_tabu(r, s, current_perm, latest_occupation, list_size, t) or \
            aspiration(r, s, current_cost, best_cost, delta):
        
        # make tabu and swap
            make_tabu(r, s, current_perm, latest_occupation, t)
            current_perm[r], current_perm[s] = current_perm[s], current_perm[r]
        
            # incremental cost update
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_so_far = current_perm.copy()
    
    print(f"Step {t}: Current cost={current_cost}, Best cost={best_cost}")
    print("Search Completed.")
    print(f"Best solution found: {best_so_far}")
    print(f"Best cost: {best_cost:.4f}")
    print(f"Final Tabu Status (0,1): {is_tabu(0, 1, current_perm, latest_occupation, list_size, max_iterations)}")



