

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 15:17:59 2022

@author: yanbing_wang
"""
import queue
import os
from i24_configparse import parse_cfg
import time
from collections import defaultdict
import sys
sys.path.append('../')
from stitcher import stitch_raw_trajectory_fragments
import min_cost_flow as mcf


# TODO:
    # 1. check if the answers agree with nx.edmond_karp
    # 2. add more intelligent enter/exiting cost based on the direction and where they are relative to the road
    # 3. collapse path into human-readable ones, e.g., [1001, 1002, etc]
    
# Examine test results in ID
# print fragments and ID switches
base = 100000

def test_fragments(gt_ids, paths):
    '''
    Count the number of fragments (under-stitch) from the output of the stitcher
    '''        
    gt_id_st_fgm_ids = defaultdict(set) # key: (int) gt_id, val: (set) corresponding stitcher fragment_ids
    IDS = 0

    for i,path in enumerate(paths):
        corr_gt_ids = set()
        for node in path:
            node = float(node)
            corr_gt_ids.add(node//base)
            gt_id_st_fgm_ids[node//base].add(i)
            
        if len(corr_gt_ids) > 1:
            print("ID switches: ", corr_gt_ids)
            IDS += len(corr_gt_ids) - 1
    
    FGMT = 0
    for key,val in gt_id_st_fgm_ids.items():
        if len(val) > 1:
            print("fragments: ", [paths[i] for i in val])
            FGMT += len(val)-1
                

    return FGMT, IDS



        
        
    
if __name__ == '__main__':
    
    # get parameters
    cwd = os.getcwd()
    cfg = "../config"
    config_path = os.path.join(cwd,cfg)
    os.environ["user_config_directory"] = config_path
    os.environ["my_config_section"] = "DEBUG"
    parameters = parse_cfg("my_config_section", cfg_name = "test_param.config")
    
    # read to queue
    gt_ids = [i for i in range(100,150)]
    # gt_ids = [131, 108]
    gt_val = 30
    lt_val = 40
    
    # fragment_queue,actual_gt_ids,_ = mcf.read_to_queue(gt_ids=gt_ids, gt_val=gt_val, lt_val=lt_val, parameters=parameters)
    # print("actual_gt_ids: ", len(actual_gt_ids))
    # s1 = fragment_queue.qsize()
    # stitched_trajectory_queue = queue.Queue()
    
    # start stitching
    # print("MCF Batch...")
    # t1 = time.time()
    # mcf.min_cost_flow_batch(fragment_queue, stitched_trajectory_queue, parameters)
    # # stitch_raw_trajectory_fragments("west", fragment_queue,stitched_trajectory_queue, parameters)
    # batch = list(stitched_trajectory_queue.queue)
    # s2 = stitched_trajectory_queue.qsize()
    # t2 = time.time()
    # print("{} fragment stitched to {} trajectories, taking {:.2f} sec".format(s1, s2, t2-t1))
    
    # # test
    # FGMT, IDS = test_fragments(gt_ids, batch)
    # print("FGMT: {}, IDS: {}".format(FGMT, IDS))
    
    
    # read to queue
    fragment_queue,actual_gt_ids, _ = mcf.read_to_queue(gt_ids=gt_ids, gt_val=gt_val, lt_val=lt_val, parameters=parameters)
    s1 = fragment_queue.qsize()
    stitched_trajectory_queue = queue.Queue()
    
    # start stitching
    print("MCF Online...")
    t1 = time.time()
    mcf.min_cost_flow_online_neg_cycle("west", fragment_queue, stitched_trajectory_queue, parameters)
    online = list(stitched_trajectory_queue.queue)
    s2 = stitched_trajectory_queue.qsize()
    t2 = time.time()
    print("{} fragment stitched to {} trajectories, taking {:.2f} sec".format(s1, s2, t2-t1))
    
    # test
    FGMT, IDS = test_fragments(gt_ids, online)
    print("FGMT: {}, IDS: {}".format(FGMT, IDS))
    
    
    # for path_o in online:
    #     if path_o not in batch:
    #         print("difference: ", path_o)
    
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