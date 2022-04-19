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

res = raw.read_query(query_filter = {"last_timestamp": {"$gt": 0, "$lt":500}}, query_sort = [("last_timestamp", "ASC")],
                   limit = 0)
fragment_queue = queue()
for doc in res:
    fragment_id = doc["_id"]
    fragment_queue.put(doc)
    
# %% Run stitcher with a pre-filled queue
stitched_trajectories_queue = queue()
stitch_raw_trajectory_fragments(fragment_queue, stitched_trajectories_queue, log_queue=None)
stitched = DBReader(host=db_parameters.DEFAULT_HOST, port=db_parameters.DEFAULT_PORT, username=db_parameters.DEFAULT_USERNAME,   
               password=db_parameters.DEFAULT_PASSWORD,
               database_name=db_parameters.DB_NAME, collection_name=db_parameters.STITCHED_COLLECTION)


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
