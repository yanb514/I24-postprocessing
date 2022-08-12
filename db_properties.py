#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:02:50 2022

@author: yanbing_wang
"""
from i24_database_api import DBClient
from i24_configparse import parse_cfg
import os
import json

# parameters path
if __name__ == '__main__':
    with open("config/db_param1.json") as f:
        parameters = json.load(f)
    
    # parameters["database_name"] = None
    # parameters["collection_name"] = ""
    client = DBClient(**parameters)
    print(client)
    print(client.client.list_database_names())
    
    # client.reset_collection()
    # print(client.collection_name)
    
    #%%
    # dbr.create_index(["ID", "first_timestamp", "last_timestamp", "starting_x", "ending_x"])
    # print("collection name: ", client.collection_name)
    # print("number of traj: ", client.count())
    
    # print("min start time: ", client.get_min("first_timestamp"))
    # print("max start time: ", client.get_max("first_timestamp"))
    
    # print("min end time: ", client.get_min("last_timestamp"))
    # print("max end time: ", client.get_max("last_timestamp"))
    
    # print("min start x: ", client.get_min("starting_x"))
    # print("max start x: ", client.get_max("starting_x"))
    
    # print("min ending x: ", client.get_min("ending_x"))
    # print("max ending x: ", client.get_max("ending_x"))
    
    # print(len(client.list_collection_names()))
    # print(client.list_collection_names())
    # print(client.collection_name)
    
    # for col in client.list_collection_names():
    #     if client.db[col].count_documents({}) == 0:
    #         client.db[col].drop()
    #         print("dropped ", col)
    
    
    #%%
    # client.transform()
    # col = client.get_latest_collection()
    
    #
    # col_list = ['morose_panda--RAW_GT1_cajoles','morose_panda--RAW_GT1_castigates',
    #             'morose_panda--RAW_GT1_disputes','morose_panda--RAW_GT1_boggles']
    # newdb = client.client["reconciled"]
    # for col in col_list:
    #     newcol = newdb[col]
    #     collection = client.db[col]
    #     for doc in collection.find({}):
    #         newcol.insert_one(doc)
    #     collection.drop()
    #     print("dropped ", col)
                
    
    #%% clean up GT
    # import numpy as np
    # gt = client.collection
    # start = 1623877091
    # end = 1623877181
    # newgt = client.db["groundtruth_scene_1_130"]
    # for doc in gt.find({}):
    #     b1 = np.array(doc["timestamp"]) >= start
    #     b2 = np.array(doc["timestamp"]) <= end
    #     time_select = np.logical_and(b1,b2)
    #     b3 = np.array(doc["x_position"]) >= 0
    #     b4 = np.array(doc["x_position"]) <= 2000
    #     x_select = np.logical_and(b3,b4)
    #     total_select = np.logical_and(time_select, x_select)
    #     print(len(doc["timestamp"]), total_select.sum())
        
    #     if total_select.any():
    #         doc["timestamp"] = list(np.array(doc["timestamp"])[total_select])
    #         doc["raw timestamp"] = list(np.array(doc["raw timestamp"])[total_select])
    #         doc["first_timestamp"] = doc["timestamp"][0]
    #         doc["last_timestamp"] = doc["timestamp"][-1]
    #         doc["x_position"] = list(np.array(doc["x_position"])[total_select])
    #         doc["y_position"] = list(np.array(doc["y_position"])[total_select])
    #         doc["starting_x"] = doc["x_position"][0]
    #         doc["ending_x"] = doc["x_position"][-1]
                
    #         newgt.insert_one(doc)
        
    
    #%% clean stitched database
    # raw_db = client.client["trajectories"]
    # rec_db = client.client["reconciled"]
    # st_db = client.client["stitched"]
    
    # for rec_col in rec_db.list_collection_names():
    #     if rec_db[rec_col].count_documents({}) < 100:
    #         inp = input(f"drop {rec_col}? Y/N")
    #         if inp == 'Y':
    #             rec_db[rec_col].drop()
    #             st_db[rec_col].drop()
    #             print(f"dropped {rec_col} from rec_db and st_db" )
            
    
            
    # for st_col in st_db.list_collection_names():
    #     if st_db[st_col].count_documents({}) < 100 or st_col not in rec_db.list_collection_names():
    #         inp = input(f"drop {st_col}? Y/N")
    #         if inp == 'Y':
    #             st_db[st_col].drop()
    #             print(f"dropped {st_col} from st_db" )
            
    
    #%% connect with SSH doesn't work!
    # from sshtunnel import SSHTunnelForwarder
    # import pymongo
    # import pprint
    
    # MONGO_HOST = "devprocess1.isis.vanderbilt.edu"
    # MONGO_DB = "trajectories"
    # MONGO_USER = "i24-data"
    # MONGO_PASS = "mongodb@i24"
    
    # server = SSHTunnelForwarder(
    #     MONGO_HOST,
    #     # ssh_username=MONGO_USER,
    #     # ssh_password=MONGO_PASS,
    #     ssh_username = "wangy79",
    #     ssh_password = "i24database",
    #     remote_bind_address=('127.0.0.1', 22)
    # )
    
    # server.start()
    # print(server.local_bind_port)
    # client = pymongo.MongoClient('127.0.0.1', server.local_bind_port) # server.local_bind_port is assigned local port
    # db = client[MONGO_DB]
    # pprint.pprint(db.list_collection_names())
    
    # server.stop()
    
    
    
