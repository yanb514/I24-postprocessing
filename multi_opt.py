#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 10:33:54 2022

@author: yanbing_wang
"""
import json
import numpy as np
from bson.objectid import ObjectId
import os
import matplotlib.pyplot as plt
from i24_database_api import DBClient


def plot_track(tracks):
    fig,ax = plt.subplots(1,2,figsize=(9,4))
    for track in tracks:
        ax[0].fill_between(track["timestamp"], track["x_position"], np.array(track["x_position"]) + np.array(track["direction"]*track["length"]), alpha=0.5)
        ax[1].fill_between(track["timestamp"], np.array(track["y_position"]) + 0.5*np.array(track["width"]), np.array(track["y_position"]) - 0.5*np.array(track["width"]), alpha=0.5)
    plt.show()
    return


def resample(tracks):
    


if __name__ == '__main__':
    
    # initialize parameters
    with open("config/parameters.json") as f:
        parameters = json.load(f)
        
    with open(os.path.join(os.environ["USER_CONFIG_DIRECTORY"], "db_param.json")) as f:
        db_param = json.load(f)
        
    RES_THRESH_X = parameters["residual_threshold_x"] 
    RES_THRESH_Y = parameters["residual_threshold_y"]
    CONF_THRESH = parameters["conf_threshold"],
    REMAIN_THRESH = parameters["remain_threshold"]
    
    # reconciliation_args={}
    # for key in ["lam3_x","lam3_y", "lam2_x", "lam2_y", "lam1_x", "lam1_y"]:
    #     reconciliation_args[key] = parameters[key]
    reconciliation_args = parameters["reconciliation_args"]
    
    test_dbr = DBClient(**db_param, database_name = "trajectories", collection_name = "sanctimonious_beluga--RAW_GT1")
    ids = [ObjectId('62fd2a29b463d2b0792821c1'), ObjectId('62fd2a2bb463d2b0792821c6')]
    docs = []
    
    for doc in test_dbr.collection.find({"_id": {"$in": ids}}):
        docs.append(doc)
    
    
    #%% plot
    plot_track(docs)
    
    