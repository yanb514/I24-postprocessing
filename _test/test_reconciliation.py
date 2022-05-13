#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:45:58 2022

@author: yanbing_wang
"""

import time
import os
import queue
import matplotlib.pyplot as plt
from i24_logger.log_writer import logger
reconciliation_logger = logger
from i24_database_api.db_writer import DBWriter
from i24_database_api.db_reader import DBReader
from i24_configparse.parse import parse_cfg
import sys
sys.path.append('../')
from utils.reconciliation_module import receding_horizon_2d_l1, resample, receding_horizon_2d, combine_fragments

cwd = os.getcwd()
cfg = "../config"
config_path = os.path.join(cwd,cfg)
os.environ["user_config_directory"] = config_path
parameters = parse_cfg("TEST", cfg_name = "test_param.config")

#%%
reconciliation_args = {"lam2_x": parameters.lam2_x,
                       "lam2_y": parameters.lam2_y,
                       # "lam1_x": parameters.lam1_x, 
                       # "lam1_y": parameters.lam1_y,
                       "PH": parameters.ph,
                       "IH": parameters.ih}

# already stitched fragments
raw = DBReader(host=parameters.default_host, port=parameters.default_port, 
            username=parameters.readonly_user, password=parameters.default_password,
            database_name=parameters.db_name, collection_name=parameters.raw_collection)

gt = DBReader(host=parameters.default_host, port=parameters.default_port, 
            username=parameters.readonly_user, password=parameters.default_password,
            database_name=parameters.db_name, collection_name=parameters.gt_collection)

gt_ids = [1,2]
fragment_ids = []
gt_res = gt.read_query(query_filter = {"ID": {"$in": gt_ids}},
                       limit = 0)

stitched_queue = queue.Queue()
for gt_doc in gt_res:
    print(gt_doc["ID"])
    stitched_doc = {key: gt_doc[key] for key in ["_id", "ID", "fragment_ids"]}
    stitched_queue.put(stitched_doc)
print("Queue size: ", stitched_queue.qsize())

dbw = DBWriter(host=parameters.default_host, port=parameters.default_port, 
           username=parameters.default_username, password=parameters.default_password,
           database_name=parameters.db_name, collection_name = parameters.reconciled_collection, 
           server_id=1, process_name=1, process_id=1, session_config_id=1, 
           schema_file=parameters.reconciled_schema_path)

# %% start reconciliation worker

while True:

    try:
        next_to_reconcile = stitched_queue.get(block=False)
    except:
        break
    plt.figure()
    
    t0 = time.time()
    combined_trajectory = combine_fragments(raw.collection, next_to_reconcile)
    plt.scatter(combined_trajectory["timestamp"],combined_trajectory["x_position"], s=0.1, label = "raw")
    
    t1 = time.time()
    resampled_trajectory = resample(combined_trajectory)
    plt.scatter(resampled_trajectory["timestamp"],resampled_trajectory["x_position"], s=1, label = "resampled")
    
    t2 = time.time()
    finished_trajectory = receding_horizon_2d(resampled_trajectory, **reconciliation_args)
    plt.scatter(finished_trajectory["timestamp"],finished_trajectory["x_position"], s=0.5, label = "rectified")
    plt.legend()
    
    t3 = time.time()
    dbw.write_one_trajectory(**finished_trajectory)
    
    print("combine: {:.2f}, resample: {:.2f}, rectify: {:.2f}".format(t1-t0, t2-t1, t3-t2))
    
#%% examin stitched_trajectories collection
dbw = dbw = DBWriter(host=parameters.default_host, port=parameters.default_port, 
                username=parameters.default_username, password=parameters.default_password,
                database_name=parameters.db_name, collection_name=parameters.reconciled_collection,
                server_id=1, process_name=1, process_id=1, session_config_id=1, schema_file=None)

stitched = dbw.db["stitched_trajectories"]
reconciled = dbw.db["reconciled_trajectories"]
print(stitched.count_documents({}), reconciled.count_documents({}))
# stitched.drop()
# reconciled.drop()






