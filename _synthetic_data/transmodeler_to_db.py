#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 27 2022

@author: wangy79
write simulation data from transmodeler to database following the schema

"""

#import pandas as pd

import urllib.parse
import csv
#import pymongo
from pymongo import MongoClient
#from collections import OrderedDict

username = urllib.parse.quote_plus('i24-data')
password = urllib.parse.quote_plus('mongodb@i24')
client = MongoClient('mongodb://%s:%s@10.2.218.56' % (username, password))
db=client["anomaly_detection"]
col=db["normal_freeflow"]
col.drop()
col=db["normal_freeflow"]
GTFilePath='/isis/home/wangy79/Documents/TransModeler/Data/normal_freeflow/trajectory/'

# CSV schema:
1. ID
2. Class
3. Time (sec)
4, Segment
5. Dir
6. Lane
7. Offset

#%%
#files=['0-12min.csv','12-23min.csv','23-34min.csv','34-45min.csv','45-56min.csv','56-66min.csv','66-74min.csv','74-82min.csv','82-89min.csv']
files1=['trajectory42.csv']
#lru = OrderedDict()

prevID = -1 # if curr_ID!= prevID, write to database - current csv is sorted by ID already
traj = {} # to start
traj['timestamp'] = []
traj['raw_timestamp'] = []
traj['road_segment_id'] = []
traj['x_position'] = []
traj['y_position'] = []
traj['configuration_id']=1
traj['local_fragment_id']=1
traj['compute_node_id']=1
traj['coarse_vehicle_class']= 0
traj['fine_vehicle_class']=1
                
                
for file in files1:
    print("In file {}".format(file))
    line = 0
    with open (GTFilePath+file,'r') as f:
        reader=csv.reader(f)
        next(reader) # skip the header
        
        for row in reader:
            line += 1
            ID = int(float(row[4]))
            curr_time = float(row[3])
            curr_x = float(row[41])
#            print(ID, curr_time, curr_x)
            
            
            if line % 10000 == 0:
                print("line: {}, curr_time: {:.2f}, x:{:.2f},  gtID: {} ".format(line, curr_time, curr_x, ID))
#                break
            
            if ID!=prevID and prevID!=-1: 
                # write prevID to database
                traj['db_write_timestamp'] = 0
                traj['first_timestamp']=traj['timestamp'][0]
                traj['last_timestamp']=traj['timestamp'][-1]
                traj['starting_x']=traj['x_position'][0]
                traj['ending_x']=traj['x_position'][-1]
                traj['flags'] = [file]
                traj['ID']=prevID
                
#                print("** write {} to db".format(ID))
                col.insert_one(traj)
                
                traj = {} # create a new trajectory for the next iteration
                traj['configuration_id']=1
                traj['local_fragment_id']=1
                traj['compute_node_id']=1
                traj['coarse_vehicle_class']=int(row[5])
                traj['fine_vehicle_class']=1
                traj['timestamp']=[float(row[3])]
                traj['raw_timestamp']=[float(1.0)]
                traj['road_segment_id']=[int(row[49])]
                traj['x_position']=[3.2808*float(row[41])]
                traj['y_position']=[3.2808*float(row[42])]
                traj['direction']=int(float(row[37]))
                traj['ID']=ID
                
                
            
            else: #  keep augmenting trajectory
                traj['timestamp'].extend([float(row[3])])
                traj['raw_timestamp'].extend([float(1.0)])
                traj['road_segment_id'].extend([float(row[49])])
                traj['x_position'].extend([3.2808*float(row[41])])
                traj['y_position'].extend([3.2808*float(row[42])])
                traj['length']=[3.2808*float(row[45])]
                traj['width']=[3.2808*float(row[44])]
                traj['height']=[3.2808*float(row[46])]
            
            prevID = ID
            
    f.close()
    
    
    
# flush out the last trajectory in cache
print("writing the last trajectory")
traj['db_write_timestamp'] = 0
traj['first_timestamp']=traj['timestamp'][0]
traj['last_timestamp']=traj['timestamp'][-1]
traj['starting_x']=traj['x_position'][0]
traj['ending_x']=traj['x_position'][-1]
traj['flags'] = ['gt']
traj['direction']=int(float(row[37]))
traj['ID']=ID
traj['length']=[3.2808*float(row[45])]
traj['width']=[3.2808*float(row[44])]
traj['height']=[3.2808*float(row[46])]

print("** write {} to db".format(ID))
col.insert_one(traj)





#%%
#import sys
#sys.path.append('../')
#from data_handler import DataReader

#print("Adding fragment IDs")
#
#colraw = db["raw_trajectories_two"]
#colgt = db['ground_truth_two']
#
#indices = ["_id", "ID"]
#for index in indices:
#    colraw.create_index(index)
#    colgt.create_index(index)
#    
#for rawdoc in colraw.find({}): # loop through all raw fragments
#    _id = rawdoc.get('_id')
#    raw_ID=rawdoc.get('ID')
#    gt_ID=raw_ID//100000
#    if colgt.count_documents({ 'ID': gt_ID }, limit = 1) != 0: # if gt_ID exists in colgt
#        # update
#        colgt.update_one({'ID':gt_ID},{'$push':{'fragment_ids':_id}},upsert=True)

    
    
    
    
    