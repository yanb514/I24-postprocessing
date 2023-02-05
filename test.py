#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:38:44 2022

@author: yanbing_wang
"""
from i24_database_api import DBClient
from bson.objectid import ObjectId
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    db_param= {
        "host": "10.80.4.91",
        "port": 27017,
        "username": "readonly",
        "password": "mongodb@i24",
    }    
    
    #%% get id of long trajectories
    
    # dbc = DBClient(**db_param, database_name="reconciled", collection_name = "6371a86bf0cfb3de3928f51a__firstcronjob")
    # # 637d8ea678f0cb97981425dd__post3
    # # 6371a86bf0cfb3de3928f51a__firstcronjob
    # # 637b023440527bf2daa5932f__post1
    
    # duration = 0 # min
    # duration_ts = duration * 60 # sec
    # x_range = 0.7 # mile
    # x_range_ft = x_range * 5280 # ft
    
    # # select long-range trajectories
    # cursor = dbc.collection.find({})
    # for traj in cursor:
    #     t_span = traj["timestamp"][-1]-traj["timestamp"][0]
    #     x_span = abs(traj["x_position"][-1]-traj["x_position"][0])
    #     if (t_span>duration_ts) and (x_span>x_range_ft):
    #         print(traj["_id"], round(t_span,1), round(x_span,1))
            
    #%% plot trajectories
    dbc = DBClient(**db_param, database_name="trajectories", collection_name="6380728cdd50d54aa5af0cf5")
    # traj = dbc.collection.find_one({"_id": ObjectId("63732b74e1fa5a45ae0c2fdd")})
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(traj["x_position"], traj["y_position"], traj["timestamp"], marker="o")
    
    
    