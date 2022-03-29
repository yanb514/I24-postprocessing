#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:03:39 2022

@author: yanbing_wang
"""
# 1. Insert ground truth trajectories to ground_truth_trajectories collection
# -- make ID temporary index
# -- leave fragment_ids blank


# 2. Insert raw trajectories to raw_trajectories collection
# -- a trajectory is ready to be written to database if max_x (min_x) is reached 
max_x = 8000 # whatever the length of roadsegment is. if direction = -1 set min_x = 0
traj = {} # a dictionary to keep track of all the ongoing trajectories up to current row
in_database = set() # a set to keep track of all trajectories that are already written to database

for f in all_the_csv_files:
    for row in f:
        curr_frame = row['frame']
        ID = row['ID'] # simulation ID from csv files
        x = row['x']
        if ID not in traj:
            traj[ID] = [row] # create a new dictionary for this trajectory
        else:
            traj[ID].append([row])
            # check if this ID is ready to be written to database
            if x >= max_x and ID not in in_database: # or if x <= min_x if direction = -1
                insert traj[ID] to raw_trajectories
                in_database.add(ID)

# -- write everything left in traj to raw_trajectories collection
for ID, doc in traj:
    insert doc to raw_trajectories 
    

# 3. Go through each document in raw_trajectories, retrieve _id and write to ground_truth collection
for doc in raw_trajectories:
    _id = doc['_id'] # db generated id
    raw_ID = doc['ID']  # simulation id from csv file
    gt_ID = raw_ID // 1000 # corresponding ground truth ID, if base=1000. need to consult @Zi on this
    ground_truth_trajectories[gt_ID].append(_id) # append the db-generated raw_trajectory id to the corresponding ground truth collection


# 4. TODO: Remove ID field in both collections
    raw_trajectories.remove("ID")
    ground_truth_trajectories.remove("ID")
    
