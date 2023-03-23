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
import pandas as pd
import numpy as np
import sys
sys.path.insert(0,'../utils')
from utils_opt import opt2_l1_constr



# CSV schema: (ordered in real id and then in ID)
# https://docs.google.com/document/d/1bSTX5_71Vi6R7S76FkaJclePxT888xHEg63JVEbbnII/edit#heading=h.nh02g8sndr9j
# 1,id
# 2,vehicle_class
# 3,timestamp
# 4,x_position (Front bumper)
# 5,y_position (center)
# 6,length
# 7,width
# 8,height
# 9,direction
# 10,configuration_ID
# 11,mask
# 12,real_id

idle_time = 0.3
start_time = 0
end_time = 900

reconciliation_args= {
    "lam2_x": 0,
    "lam2_y": 0,
    "lam3_x": 1e-7,
    "lam3_y": 1e-7,
    "lam1_x": 1e-3,
    "lam1_y": 0
}

def resample(car, dt=0.04, fillnan=False):
    # resample timestamps to 30hz, leave nans for missing data
    '''
    resample the original time-series to uniformly sampled time series in 1/dt Hz
    car: document
    leave empty slop as nan
    '''

    # Select time series only
    time_series_field = ["timestamp", "x_position", "y_position","length","width","height"]
    data = {key: car[key] for key in time_series_field}
    
    # Read to dataframe and resample
    df = pd.DataFrame(data, columns=data.keys()) 
    index = pd.to_timedelta(df["timestamp"], unit='s')
    df = df.set_index(index)
    df = df.drop(columns = "timestamp")
    
    # resample but leave nans
    freq = str(dt)+"S"

    # resample to 25hz, fill nans
    df=df.groupby(df.index.floor(freq)).mean().resample(freq).asfreq() # snap to integer divisible by freq
    df.index = df.index.values.astype('datetime64[ns]').astype('int64')*1e-9
    if fillnan:
        df = df.interpolate(method='linear')
    
    # do not extrapolate for more than 1 sec
    first_valid_time = pd.Series.first_valid_index(df['x_position'])
    last_valid_time = pd.Series.last_valid_index(df['x_position'])
    first_time = max(min(car['timestamp']), first_valid_time-1)
    last_time = min(max(car['timestamp']), last_valid_time+1)
    df=df[first_time:last_time]
    
    car['x_position'] = df['x_position'].values
    car['y_position'] = df['y_position'].values
    car['length'] = df['length'].values
    car['width'] = df['width'].values
    car['height'] = df['height'].values
    car['timestamp'] = df.index.values
        
    return car

def add_noise(traj):
    N = len(traj["timestamp"])
    x = np.array(traj["x_position"])+np.random.normal(0,1,N)
    y = np.array(traj["y_position"])+np.random.normal(0,0.3,N)
    # l = np.array(traj["length"])+np.random.normal(0,0.1,N)
    # w = np.array(traj["width"])+np.random.normal(0,0.02,N)
    
    traj["x_position"] = list(x)
    traj["y_position"] = list(y)
    # traj["length"] = list(l)
    # traj["width"] = list(w)
    return traj

def write_csv_to_db(db_param, write_db, write_col, GTFilePath, mode):

    dbc = DBClient(**db_param, database_name=write_db, collection_name=write_col)
    dbc.collection.drop()
    
    traj = {}
    prev_id = -999
    
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
            if mode == "raw" and int(float(row[11])) == 0: # masked for fragment
                continue
            line += 1
            ID = int(float(row[idx_id]))
            curr_time = float(row[3])
            
            # SELECT TIME WINDOW HERE!
            if curr_time < start_time or curr_time > end_time:
                continue
            curr_x = float(row[idx_x]) - float(row[6])
            
            if line % 1000 == 0:
                print("line: {}, curr_time: {:.2f}, x:{:.2f},  ID: {} ".format(line, curr_time, curr_x, ID))
            
            if ID != prev_id: # create new
                # first write old
                if traj:
                    if len(traj["timestamp"])<5:
                        print(f"skip ID {ID} because too short")
                        continue
                        
                    # resample to 25hz
                    traj = resample(traj, fillnan=True)
                    if mode == "raw":
                        traj = add_noise(traj)
                    else:
                        try:
                            traj = opt2_l1_constr(traj, **reconciliation_args)  
                        except Exception as e:
                            print(str(e))
                            continue
            
                    for key, val in traj.items():
                        if isinstance(val,np.ndarray):
                            traj[key] = list(val)
                            
                    traj['first_timestamp']=traj['timestamp'][0]
                    traj['last_timestamp']=traj['timestamp'][-1]
                    traj['starting_x']=traj['x_position'][0]
                    traj['ending_x']=traj['x_position'][-1]
                    traj['flags'] = ['none']
                    dbc.write_one_trajectory(thread=True, **traj)
                    
                # then create new   
                traj = {}
                traj['configuration_id']=9
                traj['compute_node_id']=1
                traj['road_segment_ids'] = [-1]
                traj['fine_vehicle_class'] = -1
                traj['local_fragment_id'] = -1
                traj['coarse_vehicle_class']=int(float(row[2]))
                traj['fine_vehicle_class']=int(float(row[2]))
                traj['timestamp']=[float(row[3])]
                traj['x_position']=[float(row[idx_x])- float(row[6])]
                traj['y_position']=[float(row[idx_y])]
                
                traj['ID']= ID
                
                if mode != "gt":
                    traj['length']=[float(row[6])]
                    traj['width']=[float(row[7])]
                    traj['height']=[float(row[8])]
                else:
                    traj['length']=float(row[6])
                    traj['width']=float(row[7])
                    traj['height']=float(row[8])
                
                
                # lru[ID] = traj
                
            else:
                # traj = lru[ID]
                traj['timestamp'].extend([float(row[3])])
                traj['direction'] = np.sign(curr_x - traj['x_position'][-1])
                traj['x_position'].extend([float(row[idx_x])- float(row[6])])
                traj['y_position'].extend([float(row[idx_y])])
                
                
                if mode != "gt":
                    traj['length'].extend([float(row[6])])
                    traj['width'].extend([float(row[7])])
                    traj['height'].extend([float(row[8])])
            prev_id = ID
            
    f.close()
        

    if traj:
        if len(traj["timestamp"])<5:
            print(f"skip ID {ID} because too short")
        else:
            
            # resample to 25hz
            traj = resample(traj, fillnan=True)
            if mode == "raw":
                traj = add_noise(traj)
            else:
                traj = opt2_l1_constr(traj, **reconciliation_args)
            for key, val in traj.items():
                if isinstance(val,np.ndarray):
                    traj[key] = list(val)
                    
            traj['first_timestamp']=traj['timestamp'][0]
            traj['last_timestamp']=traj['timestamp'][-1]
            traj['starting_x']=traj['x_position'][0]
            traj['ending_x']=traj['x_position'][-1]
            traj['flags'] = ['none']
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
    dbc = DBClient(**db_param).client
    if raw_collection not in dbc[raw_database].list_collection_names() or dbc[raw_database][raw_collection].count_documents({}) == 0:
        print("raw collection has to be written first to invoke add_fragment_ids function")
        return
    if gt_collection not in dbc[gt_database].list_collection_names() or dbc[gt_database][gt_collection].count_documents({}) == 0:
        print("gt collection has to be written first to invoke add_fragment_ids function")
        return
    
    print("Adding fragment IDs")
    colraw = dbc[raw_database][raw_collection]
    colgt = dbc[gt_database][gt_collection]
    indices = ["_id", "ID"]
    for index in indices:
        colraw.create_index(index)
        colgt.create_index(index)
        
    colraw.update_many({},{"$unset": {
                           "gt_ID": "",
                           "fragment_IDs": "",
                           "gt_id": "",
                           "fragment_ids": "",
                                   } })
    colgt.update_many({},{"$unset": {
                            "gt_ID": "",
                           "fragment_IDs": "",
                           "gt_id": "",
                           "fragment_ids": "",
                                   } })
    # read csv files to save correspondence in a dictionary
    curr_set = set()
    prev_id = -1
    with open (csv_file,'r') as f:
        reader=csv.reader(f)
        next(reader) # skip the header
        
        for row in reader:
            if int(float(row[11])) == 0: # masked
                continue
            
            # SELECT TIME WINDOW HERE!
            curr_time = float(row[3])
            if curr_time < start_time or curr_time > end_time:
                continue
            
            gt_id = int(float(row[12]))
            raw_id = int(float(row[1]))
            
            # if still processing this gt_id
            if gt_id == prev_id:
                curr_set.add(raw_id)
    
            else: # write everything in curr set to database
                
                if len(curr_set)!= 0:
                    curr_gt_id = prev_id
                    # select the curr_raw_ids that exist in db
                    for curr_raw_id in curr_set.copy():
                        query = {"ID":{"$eq": curr_raw_id}}
                        if colraw.count_documents(query)==0:
                            curr_set.remove(curr_raw_id)
                            
                    
                    # IF curr_set is empty, no fragment_ids associated with curr_Gt_id, then delete curr_Gt_id
                    if len(curr_set) == 0:
                        print(f"empty fragment_ids, delete {curr_gt_id} from gt")
                        colgt.delete_one({'ID': curr_gt_id})
                    else: 
                        curr_raw_ids = list(curr_set)   
                        query = colraw.find({'ID':{"$in":curr_raw_ids}},{"_id": 1})
                        curr_raw_obj_ids = [doc["_id"] for doc in query]
                        query = colgt.find({'ID':curr_gt_id},{"_id": 1})
                        curr_gt_obj_id = [doc["_id"] for doc in query]
                        assert len(curr_gt_obj_id) == 1
                        print(curr_gt_id, curr_raw_ids)
                        colgt.update_one({'ID':curr_gt_id},
                                         {'$set':{'fragment_IDs':curr_raw_ids,
                                                  "fragment_ids":curr_raw_obj_ids}},upsert=False)
                        
                        colraw.update_many({'ID':{"$in":curr_raw_ids}},
                                           {'$set':{'gt_ID':curr_gt_id,
                                                    "gt_id":curr_gt_obj_id},},upsert=False)
                
                # reinitialize current set
                curr_set = set()
                curr_set.add(raw_id)
            prev_id =  gt_id

    if len(curr_set) != 0:
        curr_gt_id = prev_id
        for curr_raw_id in curr_set.copy():
            query = {"ID":{"$eq": curr_raw_id}}
            if colraw.count_documents(query)==0:
                curr_set.remove(curr_raw_id)
                
        # IF curr_set is empty, no fragment_ids associated with curr_Gt_id, then delete curr_Gt_id
        if len(curr_set) == 0:
            print(f"empty fragment_ids, delete {curr_gt_id} from gt")
            colgt.delete_one({'ID': curr_gt_id})
        else:
            curr_raw_ids = list(curr_set)   
            query = colraw.find({'ID':{"$in":curr_raw_ids}},{"_id": 1})
            curr_raw_obj_ids = [doc["_id"] for doc in query]
            query = colgt.find({'ID':curr_gt_id},{"_id": 1})
            curr_gt_obj_id = [doc["_id"] for doc in query]
            assert len(curr_gt_obj_id) <= 1
            print(curr_gt_id, curr_raw_ids)
            colgt.update_one({'ID':curr_gt_id},
                             {'$set':{'fragment_IDs':curr_raw_ids,
                                      "fragment_ids":curr_raw_obj_ids}},upsert=False)
            
            colraw.update_many({'ID':{"$in":curr_raw_ids}},
                               {'$set':{'gt_ID':curr_gt_id,
                                        "gt_id":curr_gt_obj_id},},upsert=False)
        
    print("Complete updating fragment ids.")
    return


if __name__ == "__main__":
    with open(os.path.join(os.environ["USER_CONFIG_DIRECTORY"], "db_param.json")) as f:
        db_param = json.load(f)

    FilePath = "falsified_data_v4.1.csv"
    write_csv_to_db(db_param, "transmodeler", "tm_900_gt_v4.1", FilePath, "gt")
    # write_csv_to_db(db_param, "transmodeler", "tm_900_raw_v4.1", FilePath, "raw")

    add_fragment_ids(db_param=db_param, 
                      gt_database="transmodeler", 
                      gt_collection="tm_900_gt_v4.1", 
                      raw_database="transmodeler", 
                      raw_collection="tm_900_raw_v4.1",
                      csv_file=FilePath)
    
    # from bson.objectid import ObjectId
    # dbc = DBClient(**db_param, database_name="transmodeler",collection_name="tm_200_gt_v4.1")
    # doc = dbc.collection.find_one({"_id":ObjectId('64126889c02832c98f47e91f')})
    # dbc1 = DBClient(**db_param, database_name="reconciled",collection_name="tm_200_raw_v4.1__1")
    # doc1 = dbc1.collection.find_one({"_id":ObjectId('6412688bc02832c98f47e9af')})
    
    # import matplotlib.pyplot as plt
    # plt.figure()
    # doc = resample(doc, fillnan=False)
    # plt.scatter(doc["timestamp"], doc["x_position"], s=1, alpha=0.2)
    # plt.scatter(doc1["timestamp"], doc1["x_position"], s=1, alpha=0.2)

