# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 20:09:54 2022

@author: wangy79
Test the rectification algoithm on multiple cpus
- Test data: ~50 trajectories
- rectify(df): process all trajectories sequentially
 
"""
import multiprocessing as mp
import time
import queue
import utils_optimization as opt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_test_data(file_path):
    '''
    car: dataframe
    make n_copies of cars, each polluted with Gaussian noises
    concatenate to a df
    '''
    df = pd.read_csv(file_path, nrows=10000)
    print("Number of trajectories: ",df.ID.nunique())
    n_frames = []
    for id, car in df.groupby("ID"):
        n_frames.append(max(car["Frame #"].values)-min(car["Frame #"].values))
    print("Shortest track: {} frames".format(min(n_frames)))
    print("Longest track: {} frames".format(max(n_frames)))
    
    pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']

    # add some noise
    noise = np.random.multivariate_normal([0,0,0,0,0,0,0,0], np.diag([0.3, 0.3]*4))
    df.loc[:, pts] += noise
    
    return df

def wrapper(q,id,args):
    
    test, rec_args = args
    
    start = time.time()
    
    # the main rectification task    
    test = opt.rectify_sequential(test, rec_args)

    run_time = time.time() - start
    q.put([id,run_time])

if __name__ == '__main__':
    n_processes_list = [1, 2, 4,8,16,32,68,100,120,140,180] # number of parallel processes

    # set up rectify arguments
#     file_path = r"E:\I24-postprocess\benchmark\TM_1000_GT.csv"
    file_path = "../data/TM_1000_GT.csv"
    test = get_test_data(file_path)
    rec_args = (0.9, 0.9, 3) # lamx, lamy, order
    args = (test, rec_args)
    
    mp.set_start_method('spawn')
    avg_time = [] # total of all processes/n_processes
    max_time = []
    for n_processes in n_processes_list:
        start_time = time.time()
        q = mp.Queue()
        process_list = []
        for id in range(n_processes):
            p = mp.Process(target=wrapper, args=(q,id,args,))
            p.start()
            process_list.append(p)
        for p in process_list:    
            p.join()
        
        end_time = time.time()
        max_time.append(end_time-start_time)
        print("{} processes - max time: {}s".format(n_processes, end_time-start_time))
        # at the end, read results off of queue
        total_time = 0
        times = []
        while not q.empty():
            [id,process_time] = q.get()
            times.append(process_time)
            total_time += process_time
            print("Process {} took {} s".format(id, process_time))
       
        print("{} processes - total time: {}s".format(n_processes, total_time))
        print("{} processes - average time: {}s".format(n_processes, total_time/n_processes))
        avg_time.append(total_time/n_processes)
        
    #%% plot 
    fig, ax = plt.subplots()
    ax.plot(n_processes_list, avg_time, color = "blue", label='avg run time/test')
#     ax.plot(n_processes_list, max_time, color = "red", label='n_processes runtime')
    ax.set_xlabel("No. of parallel processes")
    ax.legend()
    fig.savefig('fig_avg.png')
    plt.close(fig)
    
    fig, ax = plt.subplots()
    ax.plot(n_processes_list, max_time, color = "red", label='max runtime')
    ax.set_xlabel("No. of parallel processes")
    ax.legend()
    fig.savefig('fig_max.png')
    plt.close(fig)
    
    fig, ax = plt.subplots()
    ax.plot(n_processes_list, total_time, color = "blue", label='total time')
    ax.set_xlabel("No. of parallel processes")
    ax.legend()
    fig.savefig('fig_total.png')
    plt.close(fig)
    # ax.set_ylabel("Avg run time / test")
