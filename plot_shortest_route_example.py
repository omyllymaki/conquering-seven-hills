from networkx import shortest_path
from osmnx import graph_from_place, get_nearest_node, plot_graph_route
import matplotlib.pyplot as plt

from constants import HILLS

graph = graph_from_place('Turku, Finland', network_type='walk')

orig_node = get_nearest_node(graph, HILLS["Vartiovuori"])
dest_node = get_nearest_node(graph, HILLS["Kakolanmäki"])

path = shortest_path(graph, orig_node, dest_node, weight='length')

fig, ax = plot_graph_route(graph, path)
plt.show()
