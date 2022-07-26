#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 21:45:29 2022

@author: yanbing_wang
Get the statistics of a collection
- tmin/tmax/xymin/xymax/# trajectories

Compare raw and reconciled (unsupervised)
- what fragments are filtered out
- unmatched fragments
- (done)y deviation
- (done)speed distribution
- (done)starting / ending x distribution
- (done)collision 
- length, width, height distribution
- density? flow?
- (done)lane distribution

Examine problematic stitching
- plot a list of fragments
- plot the reconciled trajectories

Statics output write to
- DB (?)
- file
- log.info(extra={})

TODO
1. make traj_eval faster using MongoDB projection instead of python
"""

from i24_database_api.db_reader import DBReader
from i24_database_api.db_writer import DBWriter
from i24_database_api import transform

import matplotlib.pyplot as plt
import warnings
from bson.objectid import ObjectId
import pprint
from threading import Thread
import json
import numpy as np
from collections import defaultdict
from multiprocessing.pool import ThreadPool


    
class UnsupervisedEvaluator():
    
    def __init__(self, config, trajectory_database="trajectories", timestamp_database = "transformed", collection_name=None, num_threads=100):
        '''
        Parameters
        ----------
        config : Dictionary
            store all the database-related parameters.
        collection1 : str
            Collection name.
        '''
        self.collection_name = collection_name
        self.dbr_v = DBReader(config, host = config["host"], username = config["username"], password = config["password"], port = config["port"], database_name = trajectory_database, collection_name=collection_name)
        
        # start transform trajectory-indexed collection to time-indexed collection if not already exist
        # this will create a new collection in the "transformed" database with the same collection name as in "trajectory" database
        if collection_name not in self.dbr_v.client[timestamp_database].list_collection_names():
            print("Transform to time-indexed collection first")
            transform(host=config["host"], 
                      port=config["port"], 
                      username="i24-data", 
                      password=config["password"], 
                      read_database_name=trajectory_database, 
                      read_collection_name=collection_name)
            
        self.dbr_t = DBReader(config, host = config["host"], username = config["username"], password = config["password"], port = config["port"], database_name = timestamp_database, collection_name=collection_name)
        print("connected to pymongo client")
        self.res = defaultdict(dict) # min, max, avg, stdev
        self.num_threads = num_threads
        
        self.res["collection"] = self.collection_name
        self.res["traj_count"] = self.dbr_v.count()
        self.res["timestamp_count"] = self.dbr_t.count()
        
        self.lanes = [i*12 for i in range(-1,12)]
            
       
    def __del__(self):
        try:
            del self.dbr_v
            del self.dbr_t
        except:
            pass
        
    def thread_pool(self, func, iterable = None):
        if iterable is None:
            iterable = self.dbr_v.collection.find({})
        
        pool = ThreadPool(processes=self.num_threads)
        res = []
        for item in iterable:
            async_result = pool.apply_async(func, (item,)) # tuple of args for foo
            res.append(async_result) 

        pool.close()
        pool.join()
        res = [r.get() for r in res] # non-blocking
        return res
    
    
    def traj_evaluate(self):
        '''
        Results aggregated by evaluating each trajectories
        '''
        # local functions
        def _get_duration(traj):
            return traj["last_timestamp"] - traj["first_timestamp"]
        
        def _get_x_traveled(traj):
            x = abs(traj["ending_x"] - traj["starting_x"])
            return x
        
        def _get_y_traveled(traj):
            return max(traj["y_position"]) - min(traj["y_position"])
        
        def _get_max_vx(traj):
            dx = np.diff(traj["x_position"]) * traj["direction"]
            dt = np.diff(traj["timestamp"])
            try: return max(dx/dt)
            except: return np.nan
        
        def _get_min_vx(traj):
            dx = np.diff(traj["x_position"]) * traj["direction"]
            dt = np.diff(traj["timestamp"])
            try: return min(dx/dt)
            except: return np.nan
        
        def _get_backward_cars(traj):
            dx = np.diff(traj["x_position"]) * traj["direction"]
            if np.any(dx < 0):
                return str(traj['_id'])
            return None
        
        def _get_max_vy(traj):
            dy = np.diff(traj["y_position"])
            dt = np.diff(traj["timestamp"])
            try: return max(dy/dt)
            except: return np.nan
        
        def _get_min_vy(traj):
            dy = np.diff(traj["y_position"])
            dt = np.diff(traj["timestamp"])
            try: return min(dy/dt)
            except: return np.nan
        
        def _get_min_ax(traj):
            ddx = np.diff(traj["x_position"], 2)
            dt = np.diff(traj["timestamp"])[:-1]
            try: return min(ddx/(dt**2))
            except: return np.nan
        
        def _get_max_ax(traj):
            ddx = np.diff(traj["x_position"], 2)
            dt = np.diff(traj["timestamp"])[:-1]
            try: return max(ddx/(dt**2))
            except: return np.nan
            
        def _get_lane_changes(traj):
            '''
            count number of times y position is at another lane according to lane marks
            '''
            lane_idx = np.digitize(traj["y_position"], self.lanes)
            lane_change = np.diff(lane_idx)
            # count-nonzeros
            return np.count_nonzero(lane_change)
        

        # distributions - all the functions that return a single value
        # TODO: put all functions in a separate script
        functions = [_get_duration, _get_x_traveled,
                      _get_y_traveled, _get_max_vx, _get_min_vx,
                      _get_max_vy, _get_min_vy,_get_max_ax,_get_min_ax,_get_lane_changes]
        # functions = [_get_lane_changes]
        
        for fcn in functions:
            traj_cursor = self.dbr_v.collection.find({})
            res = self.thread_pool(fcn, iterable=traj_cursor) # cursor cannot be reused
            
            attr_name = fcn.__name__[5:]
            print(f"Evaluating {attr_name}...")
            self.res[attr_name]["min"] = np.nanmin(res)
            self.res[attr_name]["max"] = np.nanmax(res)
            self.res[attr_name]["median"] = np.nanmedian(res)
            self.res[attr_name]["avg"] = np.nanmean(res)
            self.res[attr_name]["stdev"] = np.nanstd(res)
           
        # get ids - all the functions that return other information
        functions = [_get_backward_cars]
        for fcn in functions:
            traj_cursor = self.dbr_v.collection.find({})
            res = self.thread_pool(fcn, iterable=traj_cursor) # cursor cannot be reused
            
            attr_name = fcn.__name__[5:]
            print(f"Evaluating {attr_name}...")
            self.res[attr_name] = [r for r in res if r]
            

        return 
        
    
    
    
    def time_evaluate(self, sample_rate=10):
        '''
        Evaluate using time-indexed collection
        sample_rate: (int) select every sample_rate timestamps to evaluate
        '''
        # matries to convert [x,y,len,wid] to [lefttop_x, lefttop_y, bottomright_x, bottomright_y]
        east_m = np.array([[1, 0,0,0], [0,1,0,0.5], [1,0,1,0], [0,1,0,-0.5]]).T
        west_m = np.array([[1,0,-1,0], [0,1,0,0.5], [1, 0,0,0], [0,1,0,-0.5]]).T
        
        def doOverlap(pts1, pts2):
            '''
            pts: [lefttop_x, lefttop_y, bottomright_x, bottomright_y]
            return True if two rectangles overlap
            '''
            # by separating axix theorem
            return not (pts1[0] > pts2[2] or pts1[1] < pts2[3] or pts1[2] < pts2[0] or pts1[3] > pts2[1] )

        def calc_space_gap(pts1, pts2):
            '''
            pts: [lefttop_x, lefttop_y, bottomright_x, bottomright_y]
            if two cars are on the same lane, get the space gap
            '''
            if abs(pts1[1] + pts1[3] - pts2[1] - pts2[3])/2 < 6: # if two cars are likely to be on the same lane
                return max(pts2[0] - pts1[2], pts1[0] - pts2[2])
            else:
                return None
            
        def _get_overlaps(time_doc):
            '''
            Calculate pair-wise overlaps and space gap at a given timestamp
            '''
            veh_ids = time_doc['id']
            pos = time_doc["position"]
            try:
                dims = time_doc["dimensions"]
                time_doc_dict = {veh_ids[i]: pos[i] + dims[i][:2] for i in range(len(veh_ids))}
                has_dimension = True
            except KeyError:
                time_doc_dict = {veh_ids[i]: pos[i] for i in range(len(veh_ids))}
                has_dimension = False
                
            if has_dimension:
                pipeline = [
                    {"$match": {"$and" : [{"_id": {"$in": veh_ids}}, {"direction": {"$eq":1}}]}},
                    {'$project':{ '_id': 1 } },
                              ]
                query = self.dbr_v.collection.aggregate(pipeline)
                east_ids = [doc["_id"] for doc in query]
                west_ids = list(set(veh_ids) - set(east_ids))
                east_b = np.array([time_doc_dict[veh_id] for veh_id in east_ids])
                west_b = np.array([time_doc_dict[veh_id] for veh_id in west_ids])
                
            else:
                east_pipeline = [
                    {"$match": {"$and" : [{"_id": {"$in": veh_ids}}, {"direction": {"$eq":1}}]}},
                    {'$project':{ '_id': 1, 'length':1, 'width': 1 } },
                              ]
                east_query = self.dbr_v.collection.aggregate(east_pipeline)
                east_dict = {doc["_id"]: [doc["length"], doc["width"]] for doc in east_query} # get dimension info
                west_pipeline = [
                    {"$match": {"$and" : [{"_id": {"$in": veh_ids}}, {"direction": {"$eq":-1}}]}},
                    {'$project':{ '_id': 1, 'length':1, 'width': 1 } },
                              ]
                west_query = self.dbr_v.collection.aggregate(west_pipeline)
                west_dict = {doc["_id"]: [doc["length"], doc["width"]] for doc in west_query} # get dimension info
                
                east_ids = list(east_dict.keys())
                west_ids = list(west_dict.keys())
                east_b = np.array([time_doc_dict[veh_id]+east_dict[veh_id] for veh_id in east_ids])
                west_b = np.array([time_doc_dict[veh_id]+west_dict[veh_id] for veh_id in west_ids])
                
  
            overlap = []
            space_gap = []

            #a = M*b, where a=[lx, ly, rx, ry], b =[x,y,len,wid]
            # vectorize to all vehicles: A = M*B
            try:
                east_pts = np.matmul(east_b, east_m)
            except ValueError: # ids are empty
                east_pts = []

            for i, pts1 in enumerate(east_pts):
                for j, pts2 in enumerate(east_pts[i+1:]):
                    # check if two boxes overlap
                    if doOverlap(pts1, pts2):
                        overlap.append((str(east_ids[i]),str(east_ids[j])))
                    # get space gap
                    gap = calc_space_gap(pts1, pts2)
                    if gap: space_gap.append(gap)

            # west bound
            try:
                west_pts = np.matmul(west_b, west_m)
            except ValueError:
                west_pts = []
            # this can be optimized - not time consuming
            for i, pts1 in enumerate(west_pts):
                for j, pts2 in enumerate(west_pts[i+1:]):
                    # check if two boxes overlap
                    if doOverlap(pts1, pts2):
                        overlap.append((str(west_ids[i]),str(west_ids[j])))
                    # get space gap
                    gap = calc_space_gap(pts1, pts2)
                    if gap: space_gap.append(gap)
                        
            return overlap, space_gap

                    
        # start thread_pool for each timestamp
        time_cursor = self.dbr_t.collection.find({})
        res = self.thread_pool(_get_overlaps, iterable=time_cursor) # [[overlap, spacegap], [overlap, spacegap],...]
        # overlaps = set(overlaps) # get the unique values only - unhashable
        # overlaps = [r[0] for r in res if r[0]] # only report the overlaps
        # print([r[0] for r in res if r[0]])
        overlaps = set()
        dummy = [overlaps.add(rr) for r in res for rr in r[0]]
        space_gaps = [min(r[1]) for r in res if r[1]] 
        
        print("Evaluating overlaps...")
        self.res["overlaps"] = list(overlaps)
        
        print("Evaluating min_space_gaps...")
        try:
            self.res["min_space_gaps"]["min"] = min(space_gaps)
            self.res["min_space_gaps"]["max"] = max(space_gaps)
            self.res["min_space_gaps"]["avg"] = np.mean(space_gaps)
            self.res["min_space_gaps"]["median"] = np.median(space_gaps)
        except: # empty space_gaps
            pass
        
        return
    

        
    def print_res(self):
        pprint.pprint(self.res, width = 1)
    
    def save_res(self):
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)
    
        with open(f"res_{self.collection_name}.json", "w") as f:
            json.dump(self.res, f, indent=4, sort_keys=False,cls=NpEncoder)
        print("saved.")



        

if __name__ == '__main__':
    import time
    with open('config.json') as f:
        config = json.load(f)
    
    trajectory_database = "trajectories"
    timestamp_database = "transformed"
    # collection = "21_07_2022_gt1_alpha"
    # collection = "batch_5_07072022"
    # collection = "groundtruth_scene_1"
    collection = "pristine_sssnek--RAW_TRACKS"

    ue = UnsupervisedEvaluator(config, trajectory_database=trajectory_database, timestamp_database = timestamp_database,
                               collection_name=collection, num_threads=200)
    t1 = time.time()
    ue.traj_evaluate()
    ue.time_evaluate()
    t2 = time.time()
    
    print("time: ", t2-t1)
    # ue.print_res()
    ue.save_res()
    
    
    #%%
    # fragment_list = [ObjectId('62d5a345172006d4926ddae3'), ObjectId('62d5a240172006d4926ddab7')]
    # # rec_id = ObjectId("62c730078b650aa00a3b925f")
    # ue.plot_fragments(fragment_list)
    
    
    