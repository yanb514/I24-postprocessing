#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 17:49:47 2022

@author: wangy79
write simulation data from csv to database following the schema
csv is sorted by timestamp

"""

import csv
from i24_database_api import DBClient
import json
import os
from collections import OrderedDict
import random


def write_csv_to_db(db_param, write_db, write_col, GTFilePath):
    
    dbc = DBClient(**db_param, database_name=write_db, collection_name=write_col)
    lru = OrderedDict()
    idle_time = 0.1
    
    write_probability = 1
    line = 0
    
    try:
        with open (GTFilePath,'r') as f:
            reader=csv.reader(f)
            next(reader) # skip the header
            
            for row in reader:
                if not (row):
                    print("empty row")
                    continue
                line += 1
                ID = int(float(row[3]))
                curr_time = float(row[2])
                curr_x = float(row[40])
                
                if line % 10000 == 0:
                    print("line: {}, curr_time: {:.2f}, x:{:.2f},  lru size: {} ".format(line, curr_time, curr_x, len(lru)))
                
                if ID not in lru: # create new
                    traj = {}
                    traj['configuration_id']=1
                    traj['local_fragment_id']=1
                    traj['compute_node_id']=1
                    traj['coarse_vehicle_class']=int(row[4])
                    traj['fine_vehicle_class']=1
                    traj['timestamp']=[float(row[2])]
                    traj['raw_timestamp']=[float(1.0)]
                    
                    traj['road_segment_id']=[int(row[48])]
                    traj['x_position']=[3.2808*float(row[40])]
                    traj['y_position']=[3.2808*float(row[41])]
                    
                    traj['length']=[3.2808*float(row[44])]
                    traj['width']=[3.2808*float(row[43])]
                    traj['height']=[3.2808*float(row[45])]
                    traj['direction']=int(float(row[36]))
                    traj['ID']=float(row[3])
                    
                    lru[ID] = traj
                    
                else:
                    traj = lru[ID]
                    traj['timestamp'].extend([float(row[2])])
                    traj['raw_timestamp'].extend([float(1.0)])
            
                    traj['road_segment_id'].extend([float(row[48])])
                    traj['x_position'].extend([3.2808*float(row[40])])
                    traj['y_position'].extend([3.2808*float(row[41])])
                    
                    traj['length'].extend([3.2808*float(row[44])])
                    traj['width'].extend([3.2808*float(row[43])])
                    traj['height'].extend([3.2808*float(row[45])])
                    
                    lru.move_to_end(ID)
                    
                
                    
                while lru[next(iter(lru))]["timestamp"][-1] < curr_time - idle_time:
                    ID, traj = lru.popitem(last=False) #FIFO
                    traj['first_timestamp']=traj['timestamp'][0]
                    traj['last_timestamp']=traj['timestamp'][-1]
                    traj['starting_x']=traj['x_position'][0]
                    traj['ending_x']=traj['x_position'][-1]
                    traj['flags'] = ['none']
                    if random.random() < write_probability:
                        dbc.write_one_trajectory(thread=True, **traj)
            
        f.close()
    except Exception as e:
        print("get to exception {}".format(str(e)))
        pass

    
    
    print("flush out all the rest of LRU of size {}".format(len(lru)))
    while lru:
        ID, traj = lru.popitem(last=False) #FIFO
        traj['first_timestamp']=traj['timestamp'][0]
        traj['last_timestamp']=traj['timestamp'][-1]
        traj['starting_x']=traj['x_position'][0]
        traj['ending_x']=traj['x_position'][-1]
        traj['flags'] = ['none']
        
        # if random.random() < write_probability:
        # col.insert_one(traj)
        dbc.write_one_trajectory(thread=True, **traj)

    print("complete")
    return


def add_fragment_ids(db_param, gt_collection, raw_collection):
    '''
    raw trajectories have to be written to raw_collection first to invoke this function
    find all raw-> check ID-> get correponding gt ID-> update the gt document from gt_collection
    '''
    # check if raw_collection has any data
    dbc = DBClient(**db_param, database_name = "transmodeler")
    db = dbc.db
    if raw_collection not in db.list_collection_names() or db[raw_collection].count_documents({}) == 0:
        print("raw collection has to be written first to invoke add_fragment_ids function")
        return
    if gt_collection not in db.list_collection_names() or db[gt_collection].count_documents({}) == 0:
        print("gt collection has to be written first to invoke add_fragment_ids function")
        return
    
    print("Adding fragment IDs")
    
    colraw = db[raw_collection]
    colgt = db[gt_collection]
    
    indices = ["_id", "ID"]
    for index in indices:
        colraw.create_index(index)
        colgt.create_index(index)
        
    cnt = 0
    for rawdoc in colraw.find({}): # loop through all raw fragments
        _id = rawdoc.get('_id')
        raw_ID=rawdoc.get('ID')
        gt_ID=raw_ID//100000
        cnt += 1
        
        if colgt.count_documents({ 'ID': gt_ID }, limit = 1) != 0: # if gt_ID exists in colgt
            # update
            colgt.update_one({'ID':gt_ID},{'$push':{'fragment_ids':_id}},upsert=True)
            if cnt % 100 == 0:
                print(f"Updated {cnt} documents")
                
    print("Complete updating fragment ids.")
    return


if __name__ == "__main__":
    with open(os.path.join(os.environ["USER_CONFIG_DIRECTORY"], "db_param.json")) as f:
        db_param = json.load(f)

    # GTFilePath='/isis/home/teohz/Desktop/data_for_mongo/GT_sort_by_ID/'
    # GTFilePath = "/Volumes/Untitled 1/GT/23-34min.csv" # GT is ordered by timstsamp
    GTFilePath = "/Volumes/Elements/pollute/23-34min.csv"
    #TMFilePath='/isis/home/teohz/Desktop/data_for_mongo/pollute/'

    # read the top n rows of csv file as a dataframe
    # import pandas as pd
    # df = pd.read_csv(GTFilePath, nrows=100)

    write_csv_to_db(db_param, "transmodeler", "raw_23_34", GTFilePath)
    
    
    
    

