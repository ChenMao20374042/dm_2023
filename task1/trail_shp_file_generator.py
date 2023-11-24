import json
from datetime import datetime

import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from tqdm import tqdm


def generate_traj_shp(traj_file_path, save_path):
    
    trails = pd.read_csv(traj_file_path)
    columns = ["id", "geometry"]
    trails_organized_by_id = gpd.GeoDataFrame(columns=columns)
    single_trail = []
    single_timestamp = []
    tid = 0
    bar = tqdm(total=len(trails), desc="reading trails")
    for _, trail in trails.iterrows():
        if trail["traj_id"] != tid:
            trails_organized_by_id = pd.concat([trails_organized_by_id, gpd.GeoDataFrame({
                "id": [tid],
                "geometry": [LineString(single_trail)],
                "timestamp": [str(single_timestamp)],
            })], ignore_index=True)
            single_trail = []
            single_timestamp = []
            tid = trail["traj_id"]
        coordinates = json.loads(trail["coordinates"])
        single_trail.append((coordinates[0], coordinates[1]))
        parsed_time = datetime.strptime(trail["time"], "%Y-%m-%dT%H:%M:%SZ")
        single_timestamp.append(int(parsed_time.timestamp()))
        bar.update(1)
    bar.close()

    trails_organized_by_id.to_file(save_path)
