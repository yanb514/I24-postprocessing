#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:38:44 2022

@author: yanbing_wang
"""
import queue

import time

from bson.objectid import ObjectId

from i24_database_api import DBClient
import matplotlib.pyplot as plt
from utils.utils_mcf import MOTGraphSingle
from i24_database_api import DBClient

# from utils.utils_reconciliation import receding_horizon_2d_l1, resample, receding_horizon_2d, combine_fragments, rectify_2d
from utils.utils_opt import combine_fragments, resample, opt1, opt2, opt1_l1, opt2_l1
from min_cost_flow import min_cost_flow_online_alt_path
from data_feed import add_filter
from reconciliation import reconcile_single_trajectory
from _evaluation.eval_stitcher import plot_traj, plot_stitched

if __name__ == '__main__':

    
    import json
    with open("config/parameters.json") as f:
        parameters = json.load(f)
    parameters["raw_trajectory_queue_get_timeout"] = 0.1

    raw_collection = "morose_caribou--RAW_GT1" # collection name is the same in both databases
    rec_collection = "morose_caribou--RAW_GT1__escalates"
    
    dbc = DBClient(**parameters["db_param"])
    raw = dbc.client["trajectories"][raw_collection]
    rec = dbc.client["reconciled"][rec_collection]
    # gt = DBClient(**parameters["db_param"], database_name = trajectory_database, collection_name="groundtruth_scene_1")
    

    fragment_queue = queue.Queue()
    #morose caribou escalates
    f_ids = [ ObjectId('62fd0daf46a150340fcd2170'), ObjectId('62fd0dc546a150340fcd2198')]
    # get parameters for fitting
    RES_THRESH_X = parameters["residual_threshold_x"]
    RES_THRESH_Y = parameters["residual_threshold_y"]
    CONF_THRESH = parameters["conf_threshold"],
    REMAIN_THRESH = parameters["remain_threshold"]
    
    
    for f_id in f_ids:
        f = raw.find_one({"_id": f_id})
        # print(f_id, "fity ", f["fity"])
        f = add_filter(f, raw.collection, RES_THRESH_X, RES_THRESH_Y, 
                       CONF_THRESH, REMAIN_THRESH)
        # print(f["filter"])
        fragment_queue.put(f)
    s1 = fragment_queue.qsize()
    
    
    
    # --------- start online stitching --------- 
    # fragment_queue,actual_gt_ids,_ = read_to_queue(gt_ids=gt_ids, gt_val=gt_val, lt_val=lt_val, parameters=parameters)
    stitched_trajectory_queue = queue.Queue()
    t1 = time.time()
    min_cost_flow_online_alt_path("west", fragment_queue, stitched_trajectory_queue, parameters)
    # online = list(stitched_trajectory_queue.queue)
    s2 = stitched_trajectory_queue.qsize()
    t2 = time.time()
    print("{} fragment stitched to {} trajectories, taking {:.2f} sec".format(s1, s2, t2-t1))
    
    
    
    # reconciliation
    reconciliation_args = parameters["reconciliation_args"]
    reconciled_queue = queue.Queue()
    while not stitched_trajectory_queue.empty():
        fragment_list, filters = stitched_trajectory_queue.get(block=False)
        combined_trajectory = combine_fragments(raw, fragment_list, filters)
        reconcile_single_trajectory(reconciliation_args, combined_trajectory, reconciled_queue)
        # doc["timestamp"] = np.array(doc["timestamp"])
        # doc["x_position"] = np.array(doc["x_position"])
        # doc["y_position"] = np.array(doc["y_position"])
        # rec_doc = rectify_2d(doc, reg = "l1", **reconciliation_args)  
     
    
    print("final queue size: ",reconciled_queue.qsize())
    r = reconciled_queue.get()
        
    
    #%% plot
    from datetime import datetime
    import matplotlib.dates as md
    axs = plot_traj(fragment_list, raw)
    dates = [datetime.utcfromtimestamp(t) for t in r["timestamp"]]
    dates=md.date2num(dates)
    axs[0].scatter(dates, r["x_position"], s=1)
    axs[1].scatter(dates, r["y_position"],  s=1)
    axs[2].scatter(r["x_position"], r["y_position"],  s=1)
    # plt.scatter(combined_trajectory["timestamp"], combined_trajectory["x_position"], s= 1, c="b")
    # filter = np.array(doc["filter"], dtype=bool)
    # plt.scatter(np.array(doc["timestamp"])[~filter],np.array(doc["x_position"])[~filter], c="lightgrey")
    # plt.scatter(r["timestamp"], r["x_position"], s=1, c="orange")
    # print(r["x_score"], r["y_score"])
    
    
    
    
    
