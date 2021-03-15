# Conquering Seven Hills

Finding optimal (shortest) walking path for seven hills in Turku.

# Task

The task is to find shortest walking path for seven hills in Turku. The walk should start and end to same place, and every hill should be visited once.

# Technologies

- Visualization: Folium
- Calculating walking distances: networkx, osmnx

# Results

Shortest path, as suggested by simulated annealing method, with Kerttulinmäki as start and end point:

Kerttulinmäki -> Vartiovuori -> Samppalinnanmäki -> Korppolaismäki -> Kakolanmäki -> Puolalanmäki -> Yliopistonmäki -> Kerttulinmäki

Total distance: 14.29 km

<p align="center">
<img src="results/route.jpg" width="800px" />
</p>

<p align="center">
<img src="results/simulated_annealing_solution.jpeg" width="800px" />
</p>
