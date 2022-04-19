#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 17:49:47 2022

@author: wangy79
"""

#import pandas as pd
#inserts raw trajectories that are not sorted by ID
import urllib.parse
import csv
import pymongo
from pymongo import MongoClient
from collections import OrderedDict

# read the top n rows of csv file as a dataframe
#df = pd.read_csv('/isis/home/teohz/Desktop/data_for_mongo/pollute/0-12min.csv', nrows=1000)

username = urllib.parse.quote_plus('i24-data')
password = urllib.parse.quote_plus('mongodb@i24')
client = MongoClient('mongodb://%s:%s@10.2.218.56' % (username, password))
db=client["trajectories"]
col=db["raw_trajectories_one"]
col.drop()
col=db["raw_trajectories_one"]
#GTFilePath='/isis/home/teohz/Desktop/data_for_mongo/GT_sort_by_ID/'
TMFilePath='/isis/home/teohz/Desktop/data_for_mongo/pollute/'

X_MAX = 10000 # a cutoff threshold

#%%
#files=['0-12min.csv','12-23min.csv','23-34min.csv','34-45min.csv','45-56min.csv','56-66min.csv','66-74min.csv','74-82min.csv','82-89min.csv']
files1=['0-12min.csv']
lru = OrderedDict()
idle_time = 1

for file in files1:
    print("In file {}".format(file))
    line = 0
    with open (TMFilePath+file,'r') as f:
        reader=csv.reader(f)
        next(reader) # skip the header
        
        for row in reader:
            line += 1
            
            ID = int(float(row[3]))
            curr_time = float(row[2])
            curr_x = float(row[40])
            if curr_x > X_MAX:
                continue
            
            if line % 10000 == 0:
                print("line: {}, curr_time: {:.2f}, x:{:.2f},  lru size: {} ".format(line, curr_time, curr_x, len(lru)))
#                break
            
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
        #        d=datetime.utcnow()
        #        traj['db_write_timestamp']=calendar.timegm(d.timetuple()) #epoch unix time
                traj['db_write_timestamp'] = 0
                traj['first_timestamp']=traj['timestamp'][0]
                traj['last_timestamp']=traj['timestamp'][-1]
                traj['starting_x']=traj['x_position'][0]
                traj['ending_x']=traj['x_position'][-1]
                traj['flags'] = ['fragment']
                
#                print("** write {} to db".format(ID))
                col.insert_one(traj)
        
    f.close()
    
#%% flush out all fragmentes in cache
print("flush out all the rest of LRU of size {}".format(len(lru)))
for ID, traj in lru.items():
    traj['db_write_timestamp'] = 0
    traj['first_timestamp']=traj['timestamp'][0]
    traj['last_timestamp']=traj['timestamp'][-1]
    traj['starting_x']=traj['x_position'][0]
    traj['ending_x']=traj['x_position'][-1]
    traj['flags'] = ['fragment']
    
    col.insert_one(traj)
