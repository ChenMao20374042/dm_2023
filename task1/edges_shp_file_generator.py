import json
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from tqdm import tqdm
import re
import csv


def load_road_graph(file_path):

    nodes = {} # {(x,y):node_id}
    edges = {}  # {id: scopes}
    
    node_id=0

    data = pd.read_csv(file_path, header=0)
    bar = tqdm(data.iterrows(), desc='load roads', total=len(data))

    for step, row in bar:
        road_id = row['id']
        pattern = re.compile('\d+\.\d+')
        coordinate = pattern.findall(row['coordinates'])
        coordinate = [float(x) for x in coordinate]
        scopes = []
        for i in range(len(coordinate)//2):
            scopes.append((coordinate[2*i], coordinate[2*i+1]))
        
        if scopes[0] not in nodes.keys():
            nodes[scopes[0]] = node_id
            node_id += 1
        
        if scopes[-1] not in nodes.keys():
            nodes[scopes[-1]] = node_id
            node_id += 1
    
        edges[road_id] = scopes
    
    return nodes, edges
    
def dump_node_csv(nodes, save_path):

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['node_id','coordinate'])

        for coordinate in nodes.keys():
            node_id = nodes[coordinate]
            writer.writerow([node_id, list(coordinate)])
    

def dump_edges_shp(nodes, edges, save_path):

    columns = ["id", "source", "target", "x1", "y1", "x2", "y2", "geometry"]
    gdf_shapefile = gpd.GeoDataFrame(columns=columns)

    bar = tqdm(total=len(edges), desc='dump edge shape file')

    for id in edges.keys():
        scopes = edges[id]
        source_coordinate = scopes[0]
        target_coordinate = scopes[-1]

        source_geoid = nodes[source_coordinate]

        target_geoid = nodes[target_coordinate]
        
        gdf_shapefile = pd.concat([gdf_shapefile, gpd.GeoDataFrame({
        'id': [id],
        'source': [source_geoid],
        'target': [target_geoid],
        'x1': [source_coordinate[0]],
        'y1': [source_coordinate[1]],
        'x2': [target_coordinate[0]],
        'y2': [target_coordinate[1]],
        'geometry': [LineString(scopes)]
        })], ignore_index=True)
        bar.update(1)
    
    bar.close()
    gdf_shapefile.to_file(save_path)
    