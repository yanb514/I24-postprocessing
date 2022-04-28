#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:26:36 2022

@author: yanbing_wang
"""
from db_reader import DBReader
from db_writer import DBWriter
import db_parameters
import stitcher_parameters
from stitcher import stitch_raw_trajectory_fragments
import queue

# %% Fill queue once 
raw = DBReader(host=db_parameters.DEFAULT_HOST, port=db_parameters.DEFAULT_PORT, username=db_parameters.DEFAULT_USERNAME,   
               password=db_parameters.DEFAULT_PASSWORD,
               database_name=db_parameters.DB_NAME, collection_name=db_parameters.RAW_COLLECTION)

gt = DBReader(host=db_parameters.DEFAULT_HOST, port=db_parameters.DEFAULT_PORT, username=db_parameters.DEFAULT_USERNAME,   
               password=db_parameters.DEFAULT_PASSWORD,
               database_name=db_parameters.DB_NAME, collection_name=db_parameters.GT_COLLECTION)

gt_ids = [1]
fragment_ids = []
gt_res = gt.read_query(query_filter = {"ID": {"$in": gt_ids}},
                       limit = 0)

for gt_doc in gt_res:
    print(gt_doc["ID"])
    fragment_ids.extend(gt_doc["fragment_ids"])

raw_res = raw.read_query(query_filter = {"_id": {"$in": fragment_ids}},
                         query_sort = [("last_timestamp", "ASC")])

fragment_queue = queue.Queue()
for doc in raw_res:
    # print(doc["ID"])
    # if len(doc["timestamp"])>=2: # TODO: is this acceptable?
    fragment_queue.put(doc)
    
fragment_size = fragment_queue.qsize()
print("Queue size: ", fragment_size)

# % plot fragment_queue
# import matplotlib.pyplot as plt
# plt.figure()
# while not fragment_queue.empty():
#     doc = fragment_queue.get()
#     if doc["ID"] in [100165, 100166]:
#         print(doc["ID"],len(doc["timestamp"]))
#         plt.scatter(doc["timestamp"], doc["x_position"], s=0.01)
    
# % Run stitcher with a pre-filled queue
stitched_trajectories_queue = queue.Queue()
stitch_raw_trajectory_fragments(fragment_queue, stitched_trajectories_queue, log_queue=None)
stitched = DBReader(host=db_parameters.DEFAULT_HOST, port=db_parameters.DEFAULT_PORT, username=db_parameters.DEFAULT_USERNAME,   
               password=db_parameters.DEFAULT_PASSWORD,
               database_name=db_parameters.DB_NAME, collection_name=db_parameters.STITCHED_COLLECTION)

print("{} fragments stitched to {} trajectories".format(fragment_size,stitched_trajectories_queue.qsize()))


#%% print stitched_trajectories_queue
stitched_paths = []
while not stitched_trajectories_queue.empty():
    path = stitched_trajectories_queue.get()
    print([raw.find_one("ID", raw_id)["ID"] for raw_id in path])
    stitched_paths.append(path)

# %% Check results

from collections import defaultdict
# get all the maping from gt_id to stitched_id and vice versa
gt_ids = defaultdict(set) # key: (int) gt_id, val: (set) corresponding fragment_ids
pred_ids = defaultdict(set) # key: (int) stitched_traj_id, val: (set) corresponding gt_ids

for gt_id in gt.collection.find({}):
    fragment_ids = gt.collection[gt_id]["fragment_ids"]
    for fragment_id in fragment_ids:
        corr_stitched_id = stitched.collection.find_one(filter={"fragment_ids": {"$in": fragment_id}}) # this line needs to be tested
        gt_ids[gt_id].add(corr_stitched_id)
        
for stitched_id in stitched.collection.find({}):
    fragment_ids = stitched.collection[stitched_id]["fragment_ids"]
    for fragment_id in fragment_ids:
        corr_gt_id = gt.collection.find_one(filter={"fragment_ids": {"$in": fragment_id}}) # this line needs to be tested
        pred_ids[corr_gt_id].add(corr_gt_id)
        
# Compute fragments
FRAG = 0
for gt_id in gt_ids:
    FRAG += len(gt_ids[gt_id])-1
    
# Compute ID switches
IDS = 0
for pred_id in pred_ids:
    IDS += len(pred_ids[pred_id])-1


# %% Run stitcher with multiprocessing (continuously refill queue)
