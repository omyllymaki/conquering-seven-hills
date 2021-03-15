import folium
from folium.plugins import MeasureControl

from constants import HILLS, START_HILL

m = folium.Map(HILLS[START_HILL], zoom_start=14, control_scale=True)

for name, coordinates in HILLS.items():
    folium.Marker(coordinates, tooltip=name).add_to(m)
m.add_child(MeasureControl())
m.save("results/hills_on_map.html")