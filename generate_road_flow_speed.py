
from task2.generate_flow_speed import load_fmm_cpath, load_new_trajs, generate_features, fill_features
import numpy as np
import os


if __name__ == '__main__':

    fmm_result = load_fmm_cpath('./data/fmmr.txt')

    trajs = load_new_trajs('./data/new_traj.csv')

    flow_day, flow_hour, speed_day, speed_hour = generate_features(fmm_result, trajs)

    speed_day, speed_hour = fill_features(speed_day, speed_hour)

    features = [flow_day, flow_hour, speed_day, speed_hour]
    feature_names = ['flow_day', 'flow_hour', 'speed_day', 'speed_hour']
    for i in range(len(features)):
        np.save(os.path.join('./data/', feature_names[i]+'.npy'), features[i])
