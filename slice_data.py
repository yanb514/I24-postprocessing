#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 12:54:43 2023

@author: yanbing_wang
slice a portion of MOTION data based on specified time and space range
write to database
"""
from i24_database_api import DBClient
import json
import os
import pandas as pd
import numpy as np
import sys
import datetime
import time


def slice(dbc, tmin, tmax, xmin, xmax, write_collection_name, direction=None):
    """
    slice data from dbc (with database name and collection name already specified)
    write the new data in the same database as write_collection_name
    query with tmin, tmax, xmin, xmax and additional_query if provided
    """
    
    # first query for all trajectories whose first_timestamp is in [tmin, tamx] OR last_timestamp is in [tmin, tamx] OR (first_timestamp < tmin AND last_timestamp > tmax)
    # and then slice based on spatial range
    
    write_col = dbc.db[write_collection_name]
    count = write_col.estimated_document_count()
    if count > 0:
        ans = input(f"{write_collection_name} has {count} documents. Overwrite? [y/n]")
        if ans in ["y", "Y"]:
            write_col.drop()
        else:
            return
       
    time_series_keys = ["timestamp", "x_position", "y_position", "velocity", 
                        "detection_confidence", "posterior_covariance", "raw timestamp",
                        "length", "width", "height"]
    skipped = 0
    
    # east bound (increasing x)
    if direction is None or direction == "eb":
        print("Processing EB trajectories...")
        east_res = dbc.collection.find({
          "$and": [
                {
                  "$or": [
                    { "first_timestamp": { "$gte": tmin, "$lte": tmax } },
                    { "last_timestamp": { "$gte": tmin, "$lte": tmax } },
                    { "$and": [ { "first_timestamp": { "$lte": tmin } }, { "last_timestamp": { "$gte": tmax } } ] }
                  ]
                },
                {
                  "$or": [
                    { "starting_x": { "$gte": xmin, "$lte": xmax } },
                    { "ending_x": { "$gte": xmin, "$lte": xmax } },
                    { "$and": [ { "starting_x": { "$lte": xmin } }, { "ending_x": { "$gte": xmax } } ] }
                  ]
                },
                { "direction": 1 }
              ]
            })
    
        
        
        for traj in east_res:
            n = len(traj["timestamp"])
            
            # make dimensions time-series
            if isinstance(traj["length"], list) and len(traj["length"])==1:
                traj["length"] = traj["length"]*n
                traj["width"] = traj["width"]*n
                traj["height"] = traj["height"]*n
            elif isinstance(traj["length"], float):
                traj["length"] = [traj["length"]]*n
                traj["width"] = [traj["width"]]*n
                traj["height"] = [traj["height"]]*n
                
            # first index that's within time bound
            t1 = 0
            while traj["timestamp"][t1] < tmin: t1 +=1 
            
            # first index that's witihin x bound
            x1 = 0
            while traj["x_position"][x1] < xmin: x1 +=1 
            
            # last index that's within time bound
            t2 = len(traj["timestamp"])-1
            while traj["timestamp"][t2] > tmax: t2 -=1
            
            # last index that's within x bound
            x2 = len(traj["timestamp"])-1
            while traj["x_position"][x2] > xmax: x2 -=1
            
            # get the intersect of bounds
            ind1 = max(t1, x1)
            ind2 = min(t2, x2)
            if ind2 - ind1 < 1:
                skipped +=1
                continue
            
                
            for key in time_series_keys:
                try:
                    traj[key] = traj[key][ind1:ind2]
                except KeyError:
                    pass
            
                
            traj["first_timestamp"] = traj["timestamp"][0]
            traj["last_timestamp"] = traj["timestamp"][-1]
            traj["starting_x"] = traj["x_position"][0]
            traj["ending_x"] = traj["x_position"][-1]
            
            # write to db
            write_col.insert_one(traj)
    
        print(f"EAST: wrote {write_col.estimated_document_count()} documents to {write_collection_name}, skipped {skipped} short fragments")
    


    # west bound (decreasing x)
    if direction is None or direction == "wb":
        print("Processing WB trajectories...")
        west_res = dbc.collection.find({
          "$and": [
                {
                  "$or": [
                    { "first_timestamp": { "$gte": tmin, "$lte": tmax } },
                    { "last_timestamp": { "$gte": tmin, "$lte": tmax } },
                    { "$and": [ { "first_timestamp": { "$lte": tmin } }, { "last_timestamp": { "$gte": tmax } } ] }
                  ]
                },
                {
                  "$or": [
                    { "starting_x": { "$gte": xmin, "$lte": xmax } },
                    { "ending_x": { "$gte": xmin, "$lte": xmax } },
                    { "$and": [ { "ending_x": { "$lte": xmin } }, { "starting_x": { "$gte": xmax } } ] }
                  ]
                },
                { "direction": -1 }
              ]
            })
        
        for traj in west_res:
            n = len(traj["timestamp"])
            
            # make dimensions time-series
            if isinstance(traj["length"], list) and len(traj["length"])==1:
                traj["length"] = traj["length"]*n
                traj["width"] = traj["width"]*n
                traj["height"] = traj["height"]*n
            elif isinstance(traj["length"], float):
                traj["length"] = [traj["length"]]*n
                traj["width"] = [traj["width"]]*n
                traj["height"] = [traj["height"]]*n
                
            # first index that's within time bound
            t1 = 0
            while traj["timestamp"][t1] < tmin: t1 +=1 
            
            # first index that's witihin x bound
            x1 = 0
            while traj["x_position"][x1] > xmax: x1 +=1 
            
            # last index that's within time bound
            t2 = len(traj["timestamp"])-1
            while traj["timestamp"][t2] > tmax: t2 -=1
            
            # last index that's within x bound
            x2 = len(traj["timestamp"])-1
            while traj["x_position"][x2] < xmin: x2 -=1
            
            # get the intersect of bounds
            ind1 = max(t1, x1)
            ind2 = min(t2, x2)
            if ind2 - ind1 < 1:
                skipped +=1
                continue
            
                
            for key in time_series_keys:
                try:
                    traj[key] = traj[key][ind1:ind2]
                except KeyError:
                    pass
                
            traj["first_timestamp"] = traj["timestamp"][0]
            traj["last_timestamp"] = traj["timestamp"][-1]
            traj["starting_x"] = traj["x_position"][0]
            traj["ending_x"] = traj["x_position"][-1]
            
            # write to db
            write_col.insert_one(traj)


        print(f"Total: wrote {write_col.estimated_document_count()} documents to {write_collection_name}, skipped {skipped} short fragments")
    return



if __name__ == "__main__":
    with open(os.path.join(os.environ["USER_CONFIG_DIRECTORY"], "db_param.json")) as f:
        db_param = json.load(f)

    collection_name = "ICCV_gt3_cleaned"
    write_collection_name = "ICCV_gt3_cropped"
    dbc = DBClient(**db_param, database_name="trajectories",collection_name=collection_name)
    col_start_x = dbc.get_min("starting_x")
    col_end_x = dbc.get_max("ending_x")
    
    start_time = dbc.get_min("first_timestamp")#'2022:11:16:08:23:00'
    end_time = dbc.get_max("last_timestamp")#'2022:11:16:08:37:00'
    xmin = col_start_x+110 #60.3 *5280 -309804.125
    xmax = col_end_x #61 * 5280 -309804.125
    
    # start_time = datetime.datetime.strptime(start_time, '%Y:%m:%d:%H:%M:%S')
    # end_time = datetime.datetime.strptime(end_time, '%Y:%m:%d:%H:%M:%S')
    
    # Convert the datetime object to a Unix timestamp
    tmin = int(start_time)+6.5 #.timestamp())
    tmax = int(end_time) #.timestamp())
    print("Selected time window: ", tmin, tmax)
    print("Selected space window: ", xmin, xmax)
    
    print("collection time window: ",int(dbc.get_min("first_timestamp")), int(dbc.get_max("last_timestamp")))
    print("collection space window: ",col_start_x, col_end_x)
    slice(dbc, tmin, tmax, xmin, xmax, write_collection_name)
    
    
    # for traj in dbc.collection.find({}):
    #     print(len(traj["timestamp"]),len(traj["x_position"]),len(traj["y_position"]),
    #           len(traj["length"]),len(traj["width"]),len(traj["height"]))
    
   
    
    
    
    