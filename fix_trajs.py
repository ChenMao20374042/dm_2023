from task1.fix_traj import load_original_trajs, fix_trajs, dump_new_trajs

if __name__ == '__main__':
    trajs = load_original_trajs('./data/traj.csv')
    trajs = fix_trajs(trajs, drift_thresh=1, missing_thresh=0.3)
    dump_new_trajs('./data/new_traj.csv', trajs)