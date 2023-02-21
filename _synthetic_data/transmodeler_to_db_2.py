#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 10:05:05 2023

@author: yanbing_wang
write Junyi falsified_data_v2.csv to mongodb
"""

#import pandas as pd

import csv
from i24_database_api import DBClient
import json
import os
from collections import OrderedDict
import pandas as pd
import numpy as np
from collections import defaultdict


# CSV schema: (ordered in time)
# 1,id
# 2,vehicle_class
# 3,timestamp
# 4,x_position
# 5,y_position
# 6,length
# 7,width
# 8,height
# 9,direction
# 10,configuration_ID
# 11,mask
# 12,real_id

def resample(car, dt=0.04, fillnan=False):
    # resample timestamps to 30hz, leave nans for missing data
    '''
    resample the original time-series to uniformly sampled time series in 1/dt Hz
    car: document
    leave empty slop as nan
    '''

    # Select time series only
    time_series_field = ["timestamp", "x_position", "y_position"]
    data = {key: car[key] for key in time_series_field}
    
    # Read to dataframe and resample
    df = pd.DataFrame(data, columns=data.keys()) 
    index = pd.to_timedelta(df["timestamp"], unit='s')
    df = df.set_index(index)
    df = df.drop(columns = "timestamp")
    
    # resample but leave nans
    freq = str(dt)+"S"
    df = df.resample(freq).mean() # leave nans
    if fillnan:
        df = df.interpolate(method=fillnan)
    df.index = df.index.values.astype('datetime64[ns]').astype('int64')*1e-9

    # resample to 25hz, fill nans
    # df=df.groupby(df.index.floor('0.04S')).mean().resample('0.04S').asfreq()
    # df.index = df.index.values.astype('datetime64[ns]').astype('int64')*1e-9
    # df = df.interpolate(method='linear')
    # df=df.groupby(df.index.floor('0.04S')).mean().resample('0.04S').asfreq() # resample to 25Hz snaps to the closest integer
    
    # do not extrapolate for more than 1 sec
    first_valid_time = pd.Series.first_valid_index(df['x_position'])
    last_valid_time = pd.Series.last_valid_index(df['x_position'])
    first_time = max(min(car['timestamp']), first_valid_time-1)
    last_time = min(max(car['timestamp']), last_valid_time+1)
    df=df[first_time:last_time]
    
    car['x_position'] = df['x_position'].values
    car['y_position'] = df['y_position'].values
    car['timestamp'] = df.index.values
        
    return car

def add_noise(traj):
    return traj

def write_csv_to_db(db_param, write_db, write_col, GTFilePath, mode):

    dbc = DBClient(**db_param, database_name=write_db, collection_name=write_col)
    dbc.collection.drop()
    
    lru = OrderedDict()
    idle_time = 1.1
    
    line = 0
    if mode == "gt":
        idx_x = 4
        idx_y = 5
        idx_id = 12
    else:
        idx_x = 4
        idx_y = 5
        idx_id = 1
        
    
    # try:
    with open (GTFilePath,'r') as f:
        reader=csv.reader(f)
        next(reader) # skip the header
        
        for row in reader:
            if not (row):
                print("empty row")
                continue
            line += 1
            ID = int(float(row[idx_id]))
            curr_time = float(row[3])
            curr_x = float(row[idx_x])
            
            if line % 1000 == 0:
                print("line: {}, curr_time: {:.2f}, x:{:.2f},  lru size: {} ".format(line, curr_time, curr_x, len(lru)))
            
            if ID not in lru: # create new
                traj = {}
                traj['configuration_id']=9
                traj['compute_node_id']=1
                traj['coarse_vehicle_class']=int(row[2])
                traj['fine_vehicle_class']=int(row[2])
                traj['timestamp']=[float(row[3])]

                traj['x_position']=[float(row[idx_x])]
                traj['y_position']=[float(row[idx_y])]
                
                traj['ID']=float(row[idx_id])
                
                if mode != "gt":
                    traj['length']=[float(row[6])]
                    traj['width']=[float(row[7])]
                    traj['height']=[float(row[8])]
                else:
                    traj['length']=float(row[6])
                    traj['width']=float(row[7])
                    traj['height']=float(row[8])
                
                
                lru[ID] = traj
                
            else:
                traj = lru[ID]
                traj['timestamp'].extend([float(row[3])])
                traj['direction'] = np.sign(curr_x - traj['x_position'][-1])
                traj['x_position'].extend([float(row[idx_x])])
                traj['y_position'].extend([float(row[idx_y])])
                
                
                if mode != "gt":
                    traj['length'].extend([float(row[6])])
                    traj['width'].extend([float(row[7])])
                    traj['height'].extend([float(row[8])])
                
                lru.move_to_end(ID)
                
        
            while lru[next(iter(lru))]["timestamp"][-1] < curr_time - idle_time:
                ID, traj = lru.popitem(last=False) #FIFO
                if len(traj["timestamp"])==1:
                    print(f"skip ID {ID} because it only has 1 time sample")
                    continue
                    
                # resample to 25hz
                try:
                    traj = resample(traj, fillnan="cubic")
                except ValueError:
                    traj = resample(traj, fillnan="linear")
                if mode == "raw":
                    traj = add_noise(traj)
                for key, val in traj.items():
                    if isinstance(val,np.ndarray):
                        traj[key] = list(val)
                        
                traj['first_timestamp']=traj['timestamp'][0]
                traj['last_timestamp']=traj['timestamp'][-1]
                traj['starting_x']=traj['x_position'][0]
                traj['ending_x']=traj['x_position'][-1]
                traj['flags'] = ['none']
                dbc.write_one_trajectory(thread=True, **traj)
        
    f.close()
        
    # except Exception as e:
    #     print("get to exception {}".format(str(e)))
    #     pass
    
    
    
    print("flush out all the rest of LRU of size {}. current time: {}".format(len(lru), curr_time))
    while lru:
        ID, traj = lru.popitem(last=False) #FIFO
        try:
            traj = resample(traj, fillnan="cubic")
        except ValueError:
            traj = resample(traj, fillnan="linear")
        if mode == "raw":
            traj = add_noise(traj)
        for key, val in traj.items():
            if isinstance(val,np.ndarray):
                traj[key] = list(val)
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
    


def add_fragment_ids(db_param, gt_database, gt_collection, 
                     raw_database, raw_collection,
                     csv_file):
    '''
    raw trajectories have to be written to raw_collection first to invoke this function
    find all raw-> check ID-> get correponding gt ID-> update the gt document from gt_collection
    '''
    # check if raw_collection has any data
    dbc = DBClient(**db_param)
    if raw_collection not in dbc[raw_database].list_collection_names() or dbc[raw_database][raw_collection].count_documents({}) == 0:
        print("raw collection has to be written first to invoke add_fragment_ids function")
        return
    if gt_collection not in dbc[gt_database].list_collection_names() or dbc[gt_database][gt_collection].count_documents({}) == 0:
        print("gt collection has to be written first to invoke add_fragment_ids function")
        return
    
    print("Adding fragment IDs")
    
    # read csv files to save correspondence in a dictionary
    id_map = defaultdict(list) # key: gt_ID, val: a list of fragment ids
    
    with open (csv_file,'r') as f:
        reader=csv.reader(f)
        next(reader) # skip the header
        
        for row in reader:
            gt_id = row[12]
            raw_id = row[1]
            id_map[gt_id].append(raw_id)
    
    colraw = dbc[raw_database][raw_collection]
    colgt = dbc[gt_database][gt_collection]
    
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

    FilePath = "falsified_data_v2.csv"
    write_csv_to_db(db_param, "transmodeler", "raw_tm_900", FilePath, "raw")
    
    
    
    

