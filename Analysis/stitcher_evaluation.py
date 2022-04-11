#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 09:13:45 2022
@author: wangy79

Read from database
- ground_truth_trajectories
- raw_trajectories
- stitched_trajectories

Evaluation criteria
- # fragments
- # ID switches

"""
from mongodb_handler import DataHandler
import parameters


dh = DataHandler(**parameters.DB_PARAMS)

gt = dh.db['ground_truth_trajectories']
raw = dh.db['raw_trajectories']
stitched = dh.db['stitched_trajectories']

How to query such that 
gt['fragment_id'] returns a gt_id?

for gt_id in gt:
    frag_id = gt[gt_id]["fragment_ids"]
    

for stitched in stitched_trajectories:
    stitched_ids = stitched["fragment_ids"]
    
    # find all gt_ids that have stitched_ids
    

