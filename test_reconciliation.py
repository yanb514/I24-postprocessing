#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:45:58 2022

@author: yanbing_wang
"""

import parameters, db_parameters
from utils.reconciliation_module import receding_horizon_2d_l1, resample, receding_horizon_2d, combine_fragments
from db_reader import DBReader
from db_writer import DBWriter
import queue
import matplotlib.pyplot as plt
import time

#%%
reconciliation_args = {
                        "lam2_x": parameters.LAM2_X,
                        "lam2_y": parameters.LAM2_Y,
                        # "lam1_x": parameters.LAM1_X, 
                       # "lam1_y": parameters.LAM1_Y,
                       "PH": parameters.PH ,
                       "IH": parameters.IH}

# already stitched fragments
raw = DBReader(host=db_parameters.DEFAULT_HOST, port=db_parameters.DEFAULT_PORT, username=db_parameters.DEFAULT_USERNAME,   
               password=db_parameters.DEFAULT_PASSWORD,
               database_name=db_parameters.DB_NAME, collection_name=db_parameters.RAW_COLLECTION)

gt = DBReader(host=db_parameters.DEFAULT_HOST, port=db_parameters.DEFAULT_PORT, username=db_parameters.DEFAULT_USERNAME,   
               password=db_parameters.DEFAULT_PASSWORD,
               database_name=db_parameters.DB_NAME, collection_name=db_parameters.GT_COLLECTION)

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


# %% start reconciliation worker
i = 0
while i < 5:
    i += 1
    plt.figure()
    next_to_reconcile = stitched_queue.get(block=False)

    # t0 = time.time()
    combined_trajectory = combine_fragments(raw.collection, next_to_reconcile)
    # t1 = time.time()
    plt.scatter(combined_trajectory["timestamp"],combined_trajectory["x_position"], s=0.1, label = "raw")
    print(len(combined_trajectory["timestamp"]))
    resampled_trajectory = resample(next_to_reconcile)
    # t2 = time.time()

    plt.scatter(resampled_trajectory["timestamp"],resampled_trajectory["x_position"], s=1, label = "resampled")
    print(len(resampled_trajectory["timestamp"]))
    finished_trajectory = receding_horizon_2d(resampled_trajectory, **reconciliation_args)
    # t3 = time.time()

    plt.scatter(finished_trajectory["timestamp"],finished_trajectory["x_position"], s=0.5, label = "rectified")
    plt.legend()
    # print("combine: {:.2f}, resample: {:.2f}, rectify: {:.2f}".format(t1-t0, t2-t1, t3-t2))
    
#%% examin stitched_trajectories collection
dbw = DBWriter(host=db_parameters.DEFAULT_HOST, port=db_parameters.DEFAULT_PORT, username=db_parameters.DEFAULT_USERNAME,   
               password=db_parameters.DEFAULT_PASSWORD,
               database_name=db_parameters.DB_NAME, server_id=1, process_name=1, process_id=1, session_config_id=1)

stitched = dbw.db["stitched_trajectories"]
reconciled = dbw.db["reconciled_trajectories"]
print(stitched.count_documents({}), reconciled.count_documents({}))
stitched.drop()
reconciled.drop()