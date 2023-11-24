import shutil
from task2.generate_flow_speed import load_fmm_cpath
from task3.dump_eta_result import copy_and_dump

if __name__ == '__main__':

    eta_result = load_fmm_cpath('./data/eta_mr.txt')
    copy_and_dump(eta_result, './data/eta_task.csv', './data/eta_result.csv', './data/test_y.npy')
