from task2.generate_flow_speed import load_fmm_cpath, load_new_trajs
from task3.generate_train_val_sequence import generate_traj_sequence, split_train_val_set, \
                                standardize_feature, generate_test_sequence
import numpy as np
import os


if __name__ == '__main__':

    fmm_result = load_fmm_cpath('./data/fmmr.txt')
    eta_result = load_fmm_cpath('./data/eta_mr.txt')
    new_trajs = load_new_trajs('./data/new_traj.csv')
    x, y = generate_traj_sequence(fmm_result, new_trajs, road_file_path='./data/road.csv', feature_dir='./data/')
    train_x, train_y, val_x, val_y = split_train_val_set(x, y, train_ratio=0.8, seed=1)
    
    test_x = generate_test_sequence(eta_result, './data/eta_task.csv', './data/road.csv', './data/')
    train_x, val_x, test_x = standardize_feature(train_x, val_x, test_x, std_cols=[6,7,8,9,10,11,12])


    names = ['train_x', 'train_y', 'val_x', 'val_y', 'test_x']
    datas = [train_x, train_y, val_x, val_y, test_x]

    for i in range(len(names)):
        np.save(os.path.join('./data/', names[i]+'.npy'), datas[i])
        print('%s shape: %s' %(names[i], str(datas[i].shape)))