How to read the .graphml file.

import osmnx as ox
input_dir = '../workspace/gps/geolife'
G = ox.load_graphml(os.path.join(input_dir, 'Geolife.graphml'))
