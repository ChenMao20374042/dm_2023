from cmath import nan
import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import random

"""

    the feature for each road is:
    'highway','lanes','tunnel','bridge','alley','roundabout','length','maxspeed','width','flow_day','flow_hout','speed_day','speed_hour'

"""


def load_road_features(file_path, num_roads=38027):

    feature_name = ['highway','lanes','tunnel','bridge','alley','roundabout','length','maxspeed','width']
    num_feature = len(feature_name)
    road_features = np.zeros([num_roads, num_feature])

    data = pd.read_csv(file_path, header=0)
    bar = tqdm(total=len(data), desc='loading road features')

    for _,row in data.iterrows():
        road_id = row['id']
        for i in range(num_feature):
            road_features[road_id][i] = row[feature_name[i]]
        bar.update(1)
    
    bar.close()
    return road_features

def get_road_features(road_id, day, hour, road_features, flow_day, flow_hour, speed_day, speed_hour):

    flow_speed = np.array([flow_day[road_id][day], flow_hour[road_id][hour], speed_day[road_id][day], speed_hour[road_id][hour]])

    feature = np.concatenate([road_features[road_id], flow_speed])

    feature = np.array(feature) # clone

    return feature


def generate_traj_sequence(fmm_trajs, trajs, road_file_path, feature_dir, pad_val=-1):

    road_features = load_road_features(road_file_path)

    feature_names = ['flow_day', 'flow_hour', 'speed_day', 'speed_hour']
    features = []
    for name in feature_names:
        f = np.load(os.path.join(feature_dir, name+'.npy'))
        features.append(f)
    flow_day, flow_hour, speed_day, speed_hour = features

    x = []
    y = []
    max_len = 0

    bar = tqdm(total=len(fmm_trajs), desc='generating traj data')

    for traj_id in fmm_trajs.keys():
        
        begin_time = trajs[traj_id][0][2]
        dt_object = datetime.fromtimestamp(begin_time)
        weekday = dt_object.weekday()
        hour = dt_object.hour
        end_time = trajs[traj_id][-1][2]
        deta_time = end_time-begin_time
        deta_times = [trajs[traj_id][i+1][2]-trajs[traj_id][i][2] for i in range(len(trajs[traj_id])-1)]
        deta_times.insert(0,0)

        traj_feature = []
        for i in range(len(fmm_trajs[traj_id])):
            road_id =  fmm_trajs[traj_id][i]
            traj_feature.append(get_road_features(road_id, weekday, hour, road_features, flow_day, flow_hour, speed_day, speed_hour))
        
        max_len = max(max_len, len(traj_feature))

        x.append(traj_feature)
        y.append(deta_time)

        bar.update(1)
    
    bar.close()

    dimension = len(x[0][0])
    for i in range(len(x)):
        if len(x[i]) < max_len:
            x[i] += [[pad_val for j in range(dimension)] for k in range(max_len-len(x[i]))]
    
    x = np.array(x)
    y = np.array(y)

    return x,y

def split_train_val_set(x, y, train_ratio, seed=1):

    index = [i for i in range(x.shape[0])]

    train_num = int(x.shape[0]*train_ratio)

    random.seed(seed)
    train_index = random.sample(index, train_num) 
    val_index = [i for i in index if i not in train_index]

    train_x = x[train_index]
    train_y = y[train_index]
    val_x = x[val_index]
    val_y = y[val_index]
    return train_x, train_y, val_x, val_y

def generate_test_sequence(eta_match_result, eta_file_path, road_file_path, feature_dir, pad_val=-1):

    road_features = load_road_features(road_file_path)

    feature_names = ['flow_day', 'flow_hour', 'speed_day', 'speed_hour']
    features = []
    for name in feature_names:
        f = np.load(os.path.join(feature_dir, name+'.npy'))
        features.append(f)
    flow_day, flow_hour, speed_day, speed_hour = features

    x = []
    data = pd.read_csv(eta_file_path, header=0)
    bar = tqdm(total=len(data), desc='generating eta test sequence')
    max_len = -1

    for _, row in data.iterrows():
        bar.update(1)
        if type(row['time']) is not str:
            continue
        traj_id = row['traj_id']
        dt_object = datetime.strptime(row["time"], "%Y-%m-%dT%H:%M:%SZ")
        weekday = dt_object.weekday()
        hour = dt_object.hour
        if traj_id not in eta_match_result.keys():
            continue
        cpath = eta_match_result[traj_id]
        max_len = max(max_len, len(cpath))
        traj_feature = []
        for road_id in cpath:
            traj_feature.append(get_road_features(road_id, weekday, hour, road_features, flow_day, flow_hour, speed_day, speed_hour))
        x.append(traj_feature)
        
    
    bar.close()

    dimension = len(x[0][0])
    for i in range(len(x)):
        if len(x[i]) < max_len:
            x[i] += [[pad_val for j in range(dimension)] for k in range(max_len-len(x[i]))]
    
    x = np.array(x)
    return x


def standardize_feature(train_x, val_x, test_x, std_cols, pad_val=-1):

    for col in std_cols:
        feature = train_x[:, :, col]
        mask = feature != pad_val
        mean = np.mean(feature[mask])
        std = np.std(feature[mask])
        feature[mask] = (feature[mask] - mean) / std

        feature = val_x[:, :, col]
        mask = feature != pad_val
        feature[mask] = (feature[mask] - mean) / std

        feature = test_x[:, :, col]
        mask = feature != pad_val
        feature[mask] = (feature[mask] - mean) / std
    
    return train_x, val_x, test_x
