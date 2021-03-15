import functools
import logging

import folium
import networkx as nx
import numpy as np
import osmnx as ox
from folium import DivIcon

from constants import HILLS, USE_SAVED_DISTANCES, START_HILL, END_HILL
from route_optimization import simulated_annealing

ox.config(log_console=False, use_cache=True)
logging.basicConfig(level=logging.INFO)

print("Calculating graph")
graph = ox.graph_from_place('Turku, Finland', network_type='walk')

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
    n_hills = len(hill_names)
    distances_matrix = np.empty((n_hills, n_hills))

    for i in range(n_hills):
        for j in range(n_hills):
            orig_node = ox.get_nearest_node(graph, coordinates[i])
            dest_node = ox.get_nearest_node(graph, coordinates[j])
            distance = nx.shortest_path_length(G=graph, source=orig_node, target=dest_node, weight='length') / 1000
            distances_matrix[i][j] = distance
            print(f"{hill_names[i]} -> {hill_names[j]}: {distance} km")

    print("Saving distances to file")
    np.savetxt("distances.txt", distances_matrix)

start_index = hill_names.index(START_HILL)
end_index = hill_names.index(END_HILL)
init_route = [k for k in range(distances_matrix.shape[0])]
if start_index in init_route:
    init_route.remove(start_index)
if end_index in init_route:
    init_route.remove(end_index)
init_route = [start_index] + init_route + [end_index]

state = simulated_annealing(distances_matrix,
                            init_route,
                            mutation_probability=0.2,
                            max_iter=500,
                            debug_plot=False)

print('Simulated annealing solution')
for i in range(0, len(state)):
    s = hill_names[state[i]]
    if i < len(state) - 1:
        s += " -> "
    print(s, end='')
print('\nTotal distance: {0} km'.format(state.cost))
print()

print("Plotting route")
route_map = None
full_route = []
distances = []
for i in range(0, len(state) - 1):
    from_index = state[i]
    end_index = state[i + 1]

    orig_node = ox.get_nearest_node(graph, coordinates[from_index])
    dest_node = ox.get_nearest_node(graph, coordinates[end_index])

    route = nx.shortest_path(graph, orig_node, dest_node, weight='length')

    distance = nx.shortest_path_length(G=graph, source=orig_node, target=dest_node, weight='length') / 1000
    distances.append(distance)

    print(f"{hill_names[from_index]} -> {hill_names[end_index]}: {distance} km ")

    full_route += route[1:]

route_map = ox.plot_route_folium(graph, full_route,
                                 tiles="OpenStreetMap",
                                 tooltip=f"Total distance {state.cost:0.2f} km")

for i in range(len(state) - 1):
    index = state[i]
    folium.Marker(coordinates[index],
                  tooltip=f"{i}: {hill_names[index]}; {distances[i]:0.2f} km to next hill").add_to(route_map)
    folium.map.Marker(
        coordinates[index],
        icon=DivIcon(
            icon_size=(250, 36),
            icon_anchor=(0, 0),
            html='<div style="font-size: 15pt">' + str(i) + '</div>',
        )
    ).add_to(route_map)

route_map.save('results/route.html')
