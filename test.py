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
    # collection_name = "6362b1057c61e6427c5ad504__testallnodes"
    # print("collection name: ", collection_name)
    # dbr = DBClient(**db_param, database_name = "reconciled", collection_name = collection_name)
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
    dbc = DBClient(**db_param, database_name = "reconciled")
    dbc.transform2(read_collection_name = "636332547c61e6427c5ad508_short__testallnodes2", chunk_size=50)
    del dbc
    