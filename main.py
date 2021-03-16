import logging
import random
import time
from functools import partial

import numpy as np
from osmnx import graph_from_place, config

from constants import HILLS, USE_SAVED_DISTANCES, START_HILL, END_HILL
from route_optimization import Optimizer
from utils import calculate_distance_matrix, create_map, calculate_full_path, create_init_route


def exp_schedule(t, max_temperature, decay_constant):
    return max_temperature * np.exp(-decay_constant * t)


def random_swap(route, mutation_probability):
    for k in range(len(route)):
        if random.random() < mutation_probability:
            indices = [k + 1 for k in range(len(route) - 2)]  # leave first and last point out; those cannot be changed
            i, j = random.sample(indices, 2)
            value_i = route[i]
            value_j = route[j]
            route[i] = value_j
            route[j] = value_i
    return route


config(log_console=False, use_cache=True, cache_folder='./cache')
logging.basicConfig(level=logging.INFO)

print("Calculating graph")
graph = graph_from_place('Turku, Finland', network_type='walk')

hill_names, coordinates = [], []
for name, coord in HILLS.items():
    hill_names.append(name)
    coordinates.append(coord)

distances_matrix = None
if USE_SAVED_DISTANCES:
    print("Loading distances from file")
    try:
        distances_matrix = np.loadtxt("distances.txt")
    except OSError:
        USE_SAVED_DISTANCES = False
        print("Could not find distances.txt file.")

if not USE_SAVED_DISTANCES:
    print("Calculating distances between points")
    distances_matrix = calculate_distance_matrix(graph, coordinates)
    print("Saving distances to file")
    np.savetxt("distances.txt", distances_matrix)

optimizer = Optimizer(distances_matrix,
                      max_iter=500,
                      mutation_function=partial(random_swap, mutation_probability=0.2),
                      schedule_function=partial(exp_schedule, max_temperature=1.0, decay_constant=0.005))

print("Calculating optimal order of hills")
init_route = create_init_route(hill_names.index(START_HILL), hill_names.index(END_HILL), distances_matrix.shape[0])
start_time = time.time()
optimal_route, total_distance = optimizer.run(init_route)
end_time = time.time()
duration_ms = 1000 * (end_time - start_time)
print(f"Solution took {duration_ms:0.0f} ms")
optimizer.plot_solution()

print("Creating optimal path")
full_path, distances = calculate_full_path(graph, optimal_route, coordinates)

print("Creating map")
map = create_map(graph, full_path, optimal_route, hill_names, coordinates, distances)

print("Saving map to file")
map.save('results/optimal_route.html')
