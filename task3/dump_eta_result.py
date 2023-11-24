from xmlrpc.client import DateTime
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime


def copy_and_dump(eta_match_result, eta_task_src_path, eta_task_dst_path, eta_result_path):

    data = pd.read_csv(eta_task_src_path, header=0)
    data = data.copy()

    eta_result = np.load(eta_result_path)
    count = 0

    bar = tqdm(total=len(data), desc='dump eta result')
    begin_timestamp = None

    for index, row in data.iterrows():
        bar.update(1)
        if row['traj_id'] not in eta_match_result.keys():
            continue
        traj_id = row['traj_id']

        if type(row['time']) is str:
            dt_object = datetime.strptime(row["time"], "%Y-%m-%dT%H:%M:%SZ")
            begin_timestamp = dt_object.timestamp()
        else:
            end_timestamp = begin_timestamp + eta_result[count]
            count += 1
            dt_object = datetime.fromtimestamp(end_timestamp)
            time = dt_object.strftime("%Y-%m-%dT%H:%M:%SZ")
            data.at[index, 'time'] = time
    
    bar.close()
    data.to_csv(eta_task_dst_path, index=False, header=0)
