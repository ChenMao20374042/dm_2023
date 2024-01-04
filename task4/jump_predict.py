import json
from datetime import datetime, timedelta
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import pandas as pd
import math


class LSTM(nn.Module):
    def __init__(self, input_size_, hidden_size_, output_size_):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size_, hidden_size_, batch_first=True, dtype=torch.float)
        self.fc = nn.Linear(hidden_size_, output_size_, dtype=torch.float)

    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed_x)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.fc(out[:, -1, :])
        return out


def predict_next_way(prev_path, car_id_):
    model.eval()
    with torch.no_grad():
        softmax = nn.Softmax(dim=1)
        inputs_ = torch.tensor([[[way, car_id_] for way in prev_path]], dtype=torch.float)
        length_ = torch.tensor([len(prev_path)])
        outputs_ = softmax(model(inputs_, length_))
        last_way = prev_path[-1]
        next_ways = next_roads[last_way]
        predict = outputs_[0][next_ways]
        predict_way = torch.argmax(predict)
        return next_ways[predict_way.item()]


def haversine(lon1, lat1, lon2, lat2):
    r = 6371000
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_ = r * c
    return distance_


def interpolate_coordinates(start_lon, start_lat, end_lon, end_lat, distance_):
    d = haversine(start_lon, start_lat, end_lon, end_lat)
    lat1, lon1, lat2, lon2 = map(math.radians, [start_lat, start_lon, end_lat, end_lon])
    ratio = distance_ / d
    lat = lat1 + ratio * (lat2 - lat1)
    lon = lon1 + ratio * (lon2 - lon1)
    lat, lon = math.degrees(lat), math.degrees(lon)
    return lon, lat


def distance_to_way(start_, end_, point_):
    start_, end_, point_ = np.array(start_), np.array(end_), np.array(point_)
    start_end = end_ - start_
    start_point = point_ - start_
    way_length = np.linalg.norm(start_end)
    projection = np.dot(start_point, start_end) / way_length
    if projection < 0:
        distance_ = np.linalg.norm(point_ - start_)
    elif projection > way_length:
        distance_ = np.linalg.norm(point_ - end_)
    else:
        distance_ = np.linalg.norm(start_point - projection * start_end / way_length)
    return distance_


# m/s
def get_speed(cur_time, road_id_):
    cur_week = int(cur_time.strftime("%w"))
    cur_hour = int(cur_time.strftime("%H"))
    return (speed_day[road_id_][cur_week] + speed_hour[road_id_][cur_hour]) / 7.2


speed_day = np.load('../data/speed_day.npy')
speed_hour = np.load('../data/speed_hour.npy')
input_size = 2
hidden_size = 64
output_size = 38027
model = LSTM(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('models/2023-12-11_16-15-34__0.6671481634337598.pth'))

next_roads = {}
relations = pd.read_csv('../data/rel.csv', index_col=0)
for _, relation in tqdm(relations.iterrows()):
    origin, destination = relation["origin_id"], relation["destination_id"]
    if origin in next_roads:
        next_roads[origin].append(destination)
    else:
        next_roads[origin] = [destination]

trip_id2path = {}
trip_id2points = {}
with open('../data/jump_fmmr.txt') as f:
    for idx, line in enumerate(tqdm(f)):
        cols: list[str] = line.split(";")
        if idx == 0:
            assert cols[6] == 'cpath'
            assert cols[8] == 'mgeom'
            continue
        trip_id = int(cols[0])
        if len(cols[6]) == 0 or len(cols[8]) == 0:
            continue
        trip_id2path[trip_id] = list(map(int, cols[6].split(",")))
        trip_id2points[trip_id] = list(map(lambda point: tuple(map(float, point.split(' '))),
                                           cols[8].lstrip('LINESTRING(').rstrip(')\n').split(',')))

trails = pd.read_csv('../data/jump_task.csv', index_col=0)
trip_id2predict_time_span = {}
trip_id2start_time = {}
trip_id2car_id = {}
tid = trails.iloc[0]["traj_id"]
timestamps = []
for _, trail in tqdm(trails.iterrows()):
    trip_id2car_id[trail["traj_id"]] = trail["entity_id"]
    if trail["traj_id"] != tid:
        if len(timestamps) >= 2:
            trip_id2predict_time_span[tid] = int((timestamps[-1] - timestamps[-2]).total_seconds())
            trip_id2start_time[tid] = timestamps[-2]
        timestamps = []
        tid = trail["traj_id"]
    parsed_time = datetime.strptime(trail["time"], "%Y-%m-%dT%H:%M:%SZ")
    timestamps.append(parsed_time)

roads = pd.read_csv('../data/road.csv', index_col=0)
scopes = []
distance = []
for rid, road in tqdm(roads.iterrows()):
    coordinates = json.loads(road["coordinates"])
    scope = []
    for p in coordinates:
        scope.append((p[0], p[1]))
    scopes.append(scope)
    distance.append(int(road["length"]))


trip_id2predict_position = {}
for trip_id in tqdm(trip_id2path):
    path = trip_id2path[trip_id]
    points = trip_id2points[trip_id]
    start_time = trip_id2start_time[trip_id]
    time_span = trip_id2predict_time_span[trip_id]
    car_id = trip_id2car_id[trip_id]
    scope = scopes[path[-1]]
    last_point_distances = [distance_to_way(scope[i], scope[i + 1], points[-1]) for i in range(len(scope) - 1)]

    last_way_seg = last_point_distances.index(min(last_point_distances))
    cur_road = path[-1]
    cur_seg_distance = haversine(*scope[last_way_seg], *points[-1])
    cur_timestamp = start_time
    cur_remain_time = time_span

    while True:
        for cur_seg_index in range(last_way_seg, len(scopes[cur_road]) - 1):
            seg_distance = haversine(*scopes[cur_road][cur_seg_index],
                                     *scopes[cur_road][cur_seg_index + 1])
            cur_speed = get_speed(cur_timestamp, cur_road)
            if cur_seg_distance + cur_speed * cur_remain_time > seg_distance:
                cur_seg_distance = 0
                cost = (seg_distance - cur_seg_distance) / cur_speed
                cur_timestamp += timedelta(seconds=cost)
                cur_remain_time -= cost
            else:
                target = interpolate_coordinates(*scopes[cur_road][cur_seg_index],
                                                 *scopes[cur_road][cur_seg_index + 1],
                                                 cur_seg_distance + cur_speed * cur_remain_time)
                trip_id2predict_position[trip_id] = target
                break
        if trip_id in trip_id2predict_position:
            break
        last_way_seg = 0
        if cur_road not in next_roads:
            trip_id2predict_position[trip_id] = scopes[cur_road][-1]
            break
        cur_road = predict_next_way(path, car_id)
        path.append(cur_road)

with open('jump_predict.json', 'w+') as f:
    f.write(json.dumps(trip_id2predict_position, indent=4))
