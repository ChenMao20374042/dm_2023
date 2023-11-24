import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import re

def load_fmm_cpath(file_path):
    
    result = {} # {traj_id: [road_id, road_id, ...]} which is cpath in fmm result.

    with open(file_path, 'r') as f:
        f.readline()
        lines = f.readlines()
        bar = tqdm(desc='loading fmm result', total=len(lines))

        for line in lines:
            
            cpath = line.split(';')[1].split(',')
            traj_id = int(line.split(';')[0])
            cpath = [int(x) for x in cpath if len(x) > 0]
            if len(cpath) > 0:
                result[traj_id] = cpath
            bar.update(1)
    
        bar.close()
    
    return result

def load_new_trajs(file_path):

    data = pd.read_csv(file_path, header=0)
    iterator = tqdm(data.iterrows(), desc='loading new traj file', total = len(data))
    trajs = {} # {traj_id: [point, point, ...]} point=(x,y,timestamp,speed)
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
        speed = float(row['speed'])
        point = (coordinate[0], coordinate[1], timestamp, speed)
        cur_traj.append(point)
    
    trajs[cur_traj_id] = cur_traj
    return trajs

def generate_features(fmm_result, trajs, num_roads=38027):

    flow_day = np.zeros([num_roads, 7])
    flow_hour = np.zeros([num_roads, 24])

    speed_day = [[[] for j in range(7)] for i in range(num_roads)]
    speed_hour = [[[] for j in range(24)] for i in range(num_roads)]

    bar = tqdm(total=len(fmm_result), desc='generating features')

    for traj_id in fmm_result.keys():
        cpath = fmm_result[traj_id]
        traj = trajs[traj_id]
        for i in range(len(traj)):
            (x,y,time,speed) = traj[i]
            date = datetime.fromtimestamp(time)
            day = date.weekday()
            hour = date.hour
            for road_id in cpath:
                flow_day[road_id][day] += 1
                flow_hour[road_id][hour] += 1
                speed_day[road_id][day].append(speed)
                speed_hour[road_id][hour].append(speed)
        bar.update(1)
    
    bar.close()
    
    for road_id in range(num_roads):
        for day in range(7):
            speed_day[road_id][day] = np.mean(speed_day[road_id][day]) if len(speed_day[road_id][day]) > 0 else 0
        for hour in range(24):
            speed_hour[road_id][hour] = np.mean(speed_hour[road_id][hour]) if len(speed_hour[road_id][hour]) > 0 else 0
    
    speed_day = np.array(speed_day)
    speed_hour = np.array(speed_hour)

    return flow_day, flow_hour, speed_day, speed_hour

def fill_features(speed_day, speed_hour):

    num_roads = speed_day.shape[0]

    mask = speed_day != 0
    global_avg_day = np.mean(speed_day[mask])
    mask = speed_hour != 0
    global_avg_hour = np.mean(speed_hour[mask])

    bar = tqdm(total=num_roads, desc='filling missing features')

    for i in range(num_roads):

        day_periods = [(0,5),(5,7)]
        #hour_periods = [(-2,2),(2,6),(6,10),(10,14),(14,18),(18,22)]
        hour_periods = [(4*i, 4*i+4) for i in range(0,6)]

        for (begin,end) in day_periods:
            row = speed_day[i, begin:end]

            if np.all(row==0):
                speed_day[i, begin:end] = global_avg_day
            else:
                non_missing = row != 0
                missing = row == 0
                avg = np.mean(row[non_missing])
                speed_day[i, begin:end][missing] = avg
        
        for (begin,end) in hour_periods:
            row = speed_hour[i, begin:end]

            if np.all(row==0):
                speed_hour[i, begin:end] = global_avg_hour
            else:
                non_missing = row != 0
                missing = row == 0
                avg = np.mean(row[non_missing])
                speed_hour[i, begin:end][missing] = avg
        
        bar.update(1)

    bar.close()

    assert np.all(speed_day > 0)
    assert np.all(speed_hour > 0)

    return speed_day, speed_hour
        
def save_features(features, feature_names, save_path):

    assert len(features) == len(feature_names)

    for i in range(len(features)):
        np.save(os.path.join(save_path, feature_names[i]+'.npy'), features[i])
    
