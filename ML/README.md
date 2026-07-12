






## FOLIUM

[Getting Started](https://python-visualization.github.io/folium/latest/getting_started.html)
- Installation
> $ pip install folium

- Basic Map with starting location and tile theme
folium.Map((45.5236, -122.6750), tiles="cartodb positron")

- Adding Marker
```
m = folium.Map([45.35, -121.6972], zoom_start=12)

folium.Marker(
    location=[45.3288, -121.6625],
    tooltip="Click me!",
    popup="Mt. Hood Meadows",
    icon=folium.Icon(icon="cloud"),
).add_to(m)

folium.Marker(
    location=[45.3311, -121.7113],
    tooltip="Click me!",
    popup="Timberline Lodge",
    icon=folium.Icon(color="green"),
).add_to(m)

m
```

### Vectors such as lines
```
m = folium.Map(location=[-71.38, -73.9], zoom_start=11)

trail_coordinates = [
    (-71.351871840295871, -73.655963711222626),
    (-71.374144382613707, -73.719861619751498),
    (-71.391042575973145, -73.784922248007007),
    (-71.400964450973134, -73.851042243124397),
    (-71.402411391077322, -74.050048183880477),
]

folium.PolyLine(trail_coordinates, tooltip="Coast").add_to(m)

m
```


### Grouping and controlling
```
m = folium.Map((0, 0), zoom_start=7)

group_1 = folium.FeatureGroup("first group").add_to(m)
folium.Marker((0, 0), icon=folium.Icon("red")).add_to(group_1)
folium.Marker((1, 0), icon=folium.Icon("red")).add_to(group_1)

group_2 = folium.FeatureGroup("second group").add_to(m)
folium.Marker((0, 1), icon=folium.Icon("green")).add_to(group_2)

folium.LayerControl().add_to(m)

m
```

Alternatively 
m = folium.Map((0, 0), zoom_start=7)
group_1 = folium.FeatureGroup("first group").add_to(m)
for coordinate in [[0, 0], [1, 0]]:
    group_1.add_child(folium.Marker(location=coordinate, icon=folium.Icon("red")))




## ROUTING
[Youtube](https://www.youtube.com/watch?v=OOCvhc0k1R4&t=302s&ab_channel=SyntaxByte)
[Tutorial Page](https://syntaxbytetutorials.com/vehicle-route-optimization-in-python-with-openrouteservice/)

```
client = ors.Client(key='YOUR_KEY_HERE')
coords = [
    [-87.7898356, 41.8879452],
    [-87.7808524, 41.8906422],
    [-87.7895149, 41.8933762],
    [-87.7552925, 41.8809087],
    [-87.7728134, 41.8804058],
    [-87.7702890, 41.8802231],
    [-87.7787924, 41.8944518],
    [-87.7732345, 41.8770663],
]
vehicle_start = [-87.800701, 41.876214]
m = folium.Map(location=list(reversed([-87.787984, 41.8871616])), tiles="cartodbpositron", zoom_start=14)
for coord in coords:
    folium.Marker(location=list(reversed(coord))).add_to(m)
    
folium.Marker(location=list(reversed(vehicle_start)), icon=folium.Icon(color="red")).add_to(m)
m
vehicles = [
    ors.optimization.Vehicle(id=0, profile='driving-car', start=vehicle_start, end=vehicle_start, capacity=[5]),
    ors.optimization.Vehicle(id=1, profile='driving-car', start=vehicle_start, end=vehicle_start, capacity=[5])
]
jobs = [ors.optimization.Job(id=index, location=coords, amount=[1]) for index, coords in enumerate(coords)]
optimized = client.optimization(jobs=jobs, vehicles=vehicles, geometry=True)
line_colors = ['green', 'orange', 'blue', 'yellow']
for route in optimized['routes']:
    folium.PolyLine(locations=[list(reversed(coords)) for coords in ors.convert.decode_polyline(route['geometry'])['coordinates']], color=line_colors[route['vehicle']]).add_to(m)
m
```