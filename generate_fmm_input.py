from task1.edges_shp_file_generator import load_road_graph, dump_edges_shp, dump_node_csv
from task1.trail_shp_file_generator import generate_traj_shp


if __name__ == '__main__':
    
    nodes, edges = load_road_graph('./data/road.csv')
    dump_node_csv(nodes, './data/node.csv')
    dump_edges_shp(nodes, edges, './data/edges.shp')
    generate_traj_shp('./data/new_traj.csv', './data/trips.shp')
    