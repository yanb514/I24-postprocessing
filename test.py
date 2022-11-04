#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:38:44 2022

@author: yanbing_wang
"""
from i24_database_api import DBClient
import os
import json
import time
import numpy as np



if __name__ == '__main__':
    with open(os.environ["USER_CONFIG_DIRECTORY"]+"/db_param.json") as f:
        db_param = json.load(f)
        
    
    
    # dbc.transform()
    
    #%% BASIC INFO
    collection_name = "635997ddc8d071a13a9e5293"
    # print("collection name: ", collection_name)
    dbr = DBClient(**db_param, database_name = "trajectories", collection_name = collection_name)
    # print("number of traj: ", dbr.count())

    # # print("min ID: ", dbr.get_min("ID"))
    # # print("max ID: ", dbr.get_max("ID"))

    # print("min start time: ", dbr.get_min("first_timestamp"))
    # print("max start time: ", dbr.get_max("first_timestamp"))

    # print("min end time: ", dbr.get_min("last_timestamp"))
    # print("max end time: ", dbr.get_max("last_timestamp"))

    # print("min start x: ", dbr.get_min("starting_x"))
    # print("max start x: ", dbr.get_max("starting_x"))

    # print("min ending x: ", dbr.get_min("ending_x"))
    # print("max ending x: ", dbr.get_max("ending_x"))
    
    
    #%% TRANSFORM
    # dbc = DBClient(**db_param, database_name = "trajectories")
    # dbc.transform2(read_collection_name = "634ef772f8f31a6d48eab58e", chunk_size=50)
    # del dbc
    
    #%% quick fix ugh
    
    # from bson.objectid import ObjectId
    
    # collection_name = "634ef772f8f31a6d48eab58e__castigates"
    # dbc = DBClient(**db_param, database_name = "reconciled", collection_name = collection_name)
    
    # for doc in dbc.collection.find({}):
    #     # try:
    #     #     correct_dir = int(np.sign(doc["x_position"][-1] - doc["x_position"][0]))
    #     #     if correct_dir == 0:
    #     #         if doc["y_position"][0] >0:
    #     #             correct_dir = 1
    #     #         else:
    #     #             correct_dir = -1
            
    #     # except ValueError:
    #     #     print(doc["_id"])
    #     #     correct_dir = int(np.sign(doc["x_position"][-2] - doc["x_position"][0]))
    #     y = [-yy for yy in doc["y_position"]]
    #     dbc.collection.update_one({"_id": doc["_id"]}, {"$set": {"y_position": y}})
            
        
    # print("complete")
    
    #%%
    
    