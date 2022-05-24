#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 15:17:59 2022

@author: yanbing_wang
"""
import queue
from collections import deque 
from i24_database_api.db_reader import DBReader
from i24_database_api.db_writer import DBWriter
import os
from i24_configparse.parse import parse_cfg
from utils.data_structures import Fragment, PathCache, MOT_Graph
import time
from i24_logger.log_writer import logger 
import min_cost_flow as mcf


# TODO:
    # 1. check if the answers agree with nx.edmond_karp
    # 2. add more intelligent enter/exiting cost based on the direction and where they are relative to the road
    # 3. collapse path into human-readable ones, e.g., [1001, 1002, etc]
    
 


        
        
    
if __name__ == '__main__':
    
    # get parameters
    cwd = os.getcwd()
    cfg = "../config"
    config_path = os.path.join(cwd,cfg)
    os.environ["user_config_directory"] = config_path
    parameters = parse_cfg("TEST", cfg_name = "test_param.config")
    
    # read to queue
    gt_ids = [i for i in range(100,150)]
    fragment_queue,actual_gt_ids = mcf.read_to_queue(gt_ids=gt_ids, gt_val=25, lt_val=45, parameters=parameters)
    print("actual_gt_ids: ", len(actual_gt_ids))
    s1 = fragment_queue.qsize()
    stitched_trajectory_queue = queue.Queue()
    
    # start stitching
    print("MCF Batch...")
    t1 = time.time()
    online = mcf.min_cost_flow_batch(fragment_queue, stitched_trajectory_queue, parameters)
    s2 = stitched_trajectory_queue.qsize()
    t2 = time.time()
    print("{} fragment stitched to {} trajectories, taking {:.2f} sec".format(s1, s2, t2-t1))
    
    
    # read to queue
    fragment_queue,actual_gt_ids = mcf.read_to_queue(gt_ids=gt_ids, gt_val=25, lt_val=45, parameters=parameters)
    s1 = fragment_queue.qsize()
    stitched_trajectory_queue = queue.Queue()
    
    # start stitching
    print("MCF Online...")
    t1 = time.time()
    batch = mcf.min_cost_flow_online(fragment_queue, stitched_trajectory_queue, parameters)
    s2 = stitched_trajectory_queue.qsize()
    t2 = time.time()
    print("{} fragment stitched to {} trajectories, taking {:.2f} sec".format(s1, s2, t2-t1))
    
    
    for path_o in online:
        if path_o not in batch:
            print(path_o)
    
    # plot runtime
    # import matplotlib.pyplot as plt
    # import numpy as np
    # plt.figure()
    # plt.scatter(np.arange(s1), time_arr, label="run time (sec)")
    # plt.xlabel("# fragments")
    # plt.ylabel("cumulative run time (sec)")
    
    # plt.figure()
    # plt.scatter(np.arange(s1), cache_arr, label = "cache size")
    # plt.xlabel("# fragments")
    # plt.ylabel("memory size (# fragments)")
    
    # plt.figure()
    # plt.scatter(np.arange(s1), nodes_arr, label = "cache size")
    # plt.xlabel("# fragments")
    # plt.ylabel("# nodes in G")
    
    
    