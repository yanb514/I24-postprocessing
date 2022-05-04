#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:45:58 2022

@author: yanbing_wang
"""

import parameters, db_parameters
from utils.reconciliation_module import receding_horizon_2d_l1, resample, receding_horizon_2d
from db_reader import DBReader
import queue
import matplotlib.pyplot as plt
reconciliation_args = {
                        "lam2_x": parameters.LAM2_X,
                        "lam2_y": parameters.LAM2_Y,
                        # "lam1_x": parameters.LAM1_X, 
                       # "lam1_y": parameters.LAM1_Y,
                       "PH": parameters.PH ,
                       "IH": parameters.IH}

# create some fake data
raw = DBReader(host=db_parameters.DEFAULT_HOST, port=db_parameters.DEFAULT_PORT, username=db_parameters.DEFAULT_USERNAME,   
               password=db_parameters.DEFAULT_PASSWORD,
               database_name=db_parameters.DB_NAME, collection_name=db_parameters.RAW_COLLECTION)
raw_res = raw.read_query(query_filter = {"last_timestamp": {"$lt": 400}},
                         query_sort = [("last_timestamp", "ASC")])

fragment_queue = queue.Queue()
for doc in raw_res:
    # print(doc["ID"])
    fragment_queue.put(doc)
fragment_size = fragment_queue.qsize()
print("Queue size: ", fragment_size)

# %% start reconciliation worker
i = 0
while i < 5:
    i += 1
    plt.figure()
    next_to_reconcile = fragment_queue.get(block=False)
    plt.scatter(next_to_reconcile["timestamp"],next_to_reconcile["x_position"], s=1, label = "raw")
    print('got a job')
    resampled_trajectory = resample(next_to_reconcile)
    print("resampled")
    plt.scatter(resampled_trajectory["timestamp"],resampled_trajectory["x_position"], s=1, label = "resampled")
    finished_trajectory = receding_horizon_2d(resampled_trajectory, **reconciliation_args)
    print("rectified")
    plt.scatter(finished_trajectory["timestamp"],finished_trajectory["x_position"], s=1, label = "rectified")
    plt.legend()
    
    