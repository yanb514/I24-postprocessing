#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 20:23:56 2023

@author: yanbing_wang
"""


from i24_database_api import DBClient


def evaluate_this_timestamp(time_doc_gps, time_doc_traj):
    
    if not time_doc_gps or not time_doc_traj:
        # do nothing if any is None (document not found in collection)
        return None
    
    # TODO: do your matching and error calculations here
    error = 1
    return error


def main(config, gps_collection_name, traj_collection_name, sample_rate=1):
    
    client = DBClient(**config)
    db_time = client.db
    gps_time_col = db_time[gps_collection_name]
    traj_time_col = db_time[traj_collection_name]
    
    # time_cursor = gps_time_col.find({}) # select all documents. this is equivalent to sample_rate=1
    # sample timestamps based on sample_rate to speed up. sample_rate=[0,1]. see https://www.mongodb.com/docs/manual/reference/operator/aggregation/sampleRate/
    min_t_gps = gps_time_col.find_one({},sort=[("timestamp", 1)])["timestamp"]
    max_t_gps = gps_time_col.find_one({},sort=[("timestamp", -1)])["timestamp"]
    min_t_traj = traj_time_col.find_one({},sort=[("timestamp", 1)])["timestamp"]
    max_t_traj = traj_time_col.find_one({},sort=[("timestamp", -1)])["timestamp"]
    
    # select the overlapping timestamps between two collections
    time_cursor = gps_time_col.aggregate([
                                          {"$match": {"$and": [
                                              {"timestamp": { "$gt": max(min_t_gps, min_t_traj), "$lt": min(max_t_gps, max_t_traj) } }, 
                                              {"$sampleRate": sample_rate}] } },
                                          ])
    
    # this for loop can be replaced by multithreading
    error_arr = []
    for time_doc_gps in time_cursor:
        # find the corresponding traj time document of the same timestamp
        time_doc_traj = traj_time_col.find_one({"timestamp":time_doc_gps["timestamp"]})
        error = evaluate_this_timestamp(time_doc_gps, time_doc_traj)
        error_arr.append(error)
        
    return error_arr

        
        

if __name__ == '__main__':
    
    config= {
        # "host": "127.0.0.1",
        "host": "10.80.4.91",
        "port": 27017,
        "username": "mongo-admin",
        "password": "i24-data-access",
        "database_name": "transformed_beta"
    }    
    gps_collection_name = "6376625b40527bf2daa5932c_CIRCLES_GPS" # 1668686775.56
    traj_collection_name = "637581f040527bf2daa5932b__thursday" #  1668686399.92
    
    error_arr = main(config, gps_collection_name, traj_collection_name, 0.1)
    