#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 10:49:53 2022

@author: wangy79
"""

   
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:49:07 2022
@author: liuc36
"""

#inserts raw trajectories that are not sorted by ID
import urllib.parse
import csv
import pymongo
from pymongo import MongoClient
import time
from datetime import date
from datetime import datetime
import calendar

username = urllib.parse.quote_plus('i24-data')
password = urllib.parse.quote_plus('mongodb@i24')
client = MongoClient('mongodb://%s:%s@10.2.218.56' % (username, password))
db=client["trajectories"]
col=db["raw_trajectories_two"]

TMFilePath='/isis/home/teohz/Desktop/data_for_mongo/pollute/'


max_x=8000
min_x=0
count=0
inprogress={}
indatabase=set()
files=['0-12min.csv','12-23min.csv','23-34min.csv','34-45min.csv','45-56min.csv','56-66min.csv','66-74min.csv','74-82min.csv','82-89min.csv']

files2=['0-12min.csv','12-23min.csv']






for file in files:
    count=count+1
    print('starting '+file+" "+str(count))
    print('inprogress length: '+str(len(inprogress)))
    with open (TMFilePath+file,'r') as f:
        reader=csv.reader(f)
        traj={}

        next (f)
        
        for row in reader:
            # print (row)
            # break
            ID=float(row[3])
            x=float(row[40])
            
            if ID not in inprogress:
               
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
                
                traj['flags']=['anomaly_y']
                traj['length']=[3.2808*float(row[44])]
                traj['width']=[3.2808*float(row[43])]
                traj['height']=[3.2808*float(row[45])]
                traj['direction']=int(float(row[36]))
                traj['ID']=float(row[3])
                inprogress[ID]=traj
                
                
            else:
                
                curTraj=inprogress[ID]
                curTraj['timestamp'].extend([float(row[2])])
                curTraj['raw_timestamp'].extend([float(1.0)])
        
                curTraj['road_segment_id'].extend([float(row[48])])
                curTraj['x_position'].extend([3.2808*float(row[40])])
                curTraj['y_position'].extend([3.2808*float(row[41])])
                
                curTraj['flags'].extend(['anomaly_y'])
                curTraj['length'].extend([3.2808*float(row[44])])
                curTraj['width'].extend([3.2808*float(row[43])])
                curTraj['height'].extend([3.2808*float(row[45])])
        
                
            if(x<=min_x and ID not in indatabase):
                curTraj=inprogress[ID]
                d=datetime.utcnow()
                curTraj['db_write_timestamp']=calendar.timegm(d.timetuple()) #epoch unix time
                curTraj['first_timestamp']=curTraj['timestamp'][0]
                curTraj['last_timestamp']=curTraj['timestamp'][-1]
                curTraj['starting_x']=curTraj['x_position'][0]
                curTraj['ending_x']=curTraj['x_position'][-1]
                #print(curTraj)
                inprogress.pop(ID)
                #break
                col.insert_one(curTraj)
                indatabase.add(ID)
        
            traj={}
        
        f.close()
          
    if(count== 9 or count % 2==0):
        print('moving on to inprogress with '+str(count))
        for doc in inprogress:
            curTraj=inprogress[doc]
            d=datetime.utcnow()
            curTraj['db_write_timestamp']=calendar.timegm(d.timetuple())
            curTraj['first_timestamp']=curTraj['timestamp'][0]
            curTraj['last_timestamp']=curTraj['timestamp'][-1]
            curTraj['starting_x']=curTraj['x_position'][0]
            curTraj['ending_x']=curTraj['x_position'][-1]
            #print(curTraj)
            col.insert_one(curTraj)
              
        print('clearing...')
        inprogress.clear()
    
