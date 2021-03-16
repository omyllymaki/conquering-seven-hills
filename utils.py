import folium
import networkx
import numpy as np
import osmnx
from folium import DivIcon


def calculate_distance_matrix(graph, coordinates):
    n = len(coordinates)
    distances_matrix = np.empty((n, n))

    for i in range(n):
        for j in range(n):
            source_node = osmnx.get_nearest_node(graph, coordinates[i])
            target_node = osmnx.get_nearest_node(graph, coordinates[j])
            distance = networkx.shortest_path_length(G=graph,
                                                     source=source_node,
                                                     target=target_node,
                                                     weight='length')
            distances_matrix[i][j] = distance / 1000

    return distances_matrix


def create_map(graph, path, route, hill_names, coordinates, distances):
    route_map = osmnx.plot_route_folium(graph, path,
                                        tiles="OpenStreetMap",
                                        tooltip=f"Total distance {np.sum(distances):0.2f} km")

    for i in range(len(route) - 1):
        index = route[i]
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
    return route_map


def calculate_full_path(graph, route, coordinates):
    full_path = []
    distances = []
    for i in range(0, len(route) - 1):
        from_index = route[i]
        end_index = route[i + 1]

        source_node = osmnx.get_nearest_node(graph, coordinates[from_index])
        target_node = osmnx.get_nearest_node(graph, coordinates[end_index])

        path = networkx.shortest_path(graph, source_node, target_node, weight='length')

        distance = networkx.shortest_path_length(G=graph, source=source_node, target=target_node,
                                                 weight='length') / 1000
        distances.append(distance)

        full_path += path[1:]

    return full_path, distances


def create_init_route(start, end, n):
    route = [k for k in range(n)]
    if start in route:
        route.remove(start)
    if end in route:
        route.remove(end)
    route = [start] + route + [end]
    return route