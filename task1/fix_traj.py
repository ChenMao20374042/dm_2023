import csv
from math import radians, sin, cos, sqrt, atan2
import pandas as pd
import tqdm
from datetime import datetime
import re


def distance(point1, point2):

    point1 = point1[0:2]
    point2 = point2[0:2]

    lat1, lon1 = map(radians, point1)
    lat2, lon2 = map(radians, point2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    radius = 6371.0

    distance = radius * c
    return distance

def fix_drift_points(traj, drift_thresh):
    """
        traj: [point1, point2, ...] where each point in (x,y,timestamp,speed)
        thresh: the threshold to detect drift points
    """
    drift_points_index = []
    for i in range(1, len(traj)-1):
        dist_prev = distance(traj[i-1], traj[i])
        dist_next = distance(traj[i], traj[i+1])
        dist_between = distance(traj[i-1], traj[i+1])
        if dist_prev > drift_thresh and dist_next > drift_thresh and dist_between < drift_thresh:
            drift_points_index.append(i)

    for i in drift_points_index:
        x = (traj[i-1][0]+traj[i+1][0])/2
        y = (traj[i-1][1]+traj[i+1][1])/2
        time = traj[i][2]
        speed = traj[i][3]
        traj[i] = (x,y,time,speed)
    
    return traj, len(drift_points_index)

def fill_missing_points(traj, miss_thresh):
    """
    在轨迹中插值填充缺失点
    """
    filled_traj = [traj[0]]
    for i in range(1, len(traj)):
        dist_between = distance(traj[i-1], traj[i])
        if dist_between > miss_thresh:
            num_points_to_fill = int(dist_between / miss_thresh)
            for j in range(1, num_points_to_fill + 1):
                ratio = j / (num_points_to_fill + 1)
                filled_point = (
                    traj[i-1][0] + ratio * (traj[i][0] - traj[i-1][0]),
                    traj[i-1][1] + ratio * (traj[i][1] - traj[i-1][1]),
                    traj[i-1][2] + ratio * (traj[i][2] - traj[i-1][2]),
                    (traj[i-1][3] + traj[i][3])/2
                )
                filled_traj.append(filled_point)
        filled_traj.append(traj[i])

    return filled_traj, len(filled_traj)-len(traj)

def load_original_trajs(file_path):

    data = pd.read_csv(file_path, header=0)
    iterator = tqdm.tqdm(data.iterrows(), desc='loading gps traj file', total = len(data))
    trajs = {} # {traj_id: [point, point, ...]}
    cur_traj = []
    cur_traj_id = 0

    for step, row in iterator:
        if row['traj_id'] != cur_traj_id:
            trajs[cur_traj_id] = cur_traj
            cur_traj_id = row['traj_id']
            cur_traj = []
        time = row['time']
        dt_object = datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ")
        timestamp = dt_object.timestamp()
        pattern = re.compile('\d+\.\d+')
        coordinate = pattern.findall(row['coordinates'])
        coordinate = [float(x) for x in coordinate]
        speed = float(row['speeds'])
        point = (coordinate[0], coordinate[1], timestamp, speed)
        cur_traj.append(point)
    
    trajs[cur_traj_id] = cur_traj
    return trajs

def dump_new_trajs(file_path, new_trajs):
    
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id','traj_id','coordinates','time','speed'])

        bar = tqdm.tqdm(total=len(new_trajs), desc='dump new trajs')
        id = 0

        for traj_id in new_trajs.keys():
            traj = new_trajs[traj_id]
            for point in traj:
                coordinates = [point[0], point[1]]
                time = point[2]
                dt_object = datetime.fromtimestamp(time)
                formatted_time = dt_object.strftime("%Y-%m-%dT%H:%M:%SZ")
                speed = point[3]
                writer.writerow([id, traj_id, coordinates, formatted_time, speed])
                id += 1
            bar.update(1)
    
        bar.close()

def fix_trajs(trajs, drift_thresh, missing_thresh):

    new_trajs = {}
    bar = tqdm.tqdm(total=len(trajs), desc='fix trajs')

    drift_sum = 0
    missing_sum = 0

    for traj_id in trajs.keys():
        traj = trajs[traj_id]
        traj, drift_num = fix_drift_points(traj, drift_thresh)
        traj, missing_num = fill_missing_points(traj, missing_thresh)
        new_trajs[traj_id] = traj
        drift_sum += drift_num
        missing_sum += missing_num
        bar.update(1)
    
    print('fix %d drift points, fill %d missing points' %(drift_sum, missing_sum))

    return new_trajs
