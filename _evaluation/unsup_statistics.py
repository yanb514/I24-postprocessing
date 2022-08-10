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

from i24_database_api import DBClient

import matplotlib.pyplot as plt
from bson.objectid import ObjectId
import pprint
import json
import numpy as np
from collections import defaultdict
from multiprocessing.pool import ThreadPool
import time

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
    
def _get_avg_vx(traj):
    dx = np.diff(traj["x_position"])
    dt = np.diff(traj["timestamp"])
    try: return np.abs(np.average(dx/dt))
    except: return np.nan

def _get_avg_vy(traj):
    dy = np.diff(traj["y_position"])
    dt = np.diff(traj["timestamp"])
    try: return np.average(dy/dt)
    except: return np.nan
    
def _get_avg_ax(traj):
    ddx = np.diff(traj["x_position"], 2)
    dt = np.diff(traj["timestamp"])[:-1]
    try: return np.mean(ddx/(dt**2))
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
    
def _get_avg_ax(traj):
    ddx = np.diff(traj["x_position"], 2)
    dt = np.diff(traj["timestamp"])[:-1]
    try: return np.mean(ddx/(dt**2))
    except: return np.nan
 

def _get_ax(traj):
    '''
    return point-wise acceleration
    '''
    ddx = np.diff(traj["x_position"], 2)
    dt = np.diff(traj["timestamp"])[:-1]
    return ddx/(dt**2)
    
    
def _get_vx(traj):
    '''
    return point-wise acceleration
    '''
    dy = np.diff(traj["x_position"])
    dt = np.diff(traj["timestamp"])
    return np.abs(dy/dt)

    
    
def _get_residual(traj):
    try:
        return traj["x_score"]
    except: # field is not available
        return 0


def _get_lane_changes(traj, lanes = [i*12 for i in range(-1,12)]):
    '''
    count number of times y position is at another lane according to lane marks
    '''
    lane_idx = np.digitize(traj["y_position"], lanes)
    lane_change = np.diff(lane_idx)
    # count-nonzeros
    return np.count_nonzero(lane_change)

def doOverlap(pts1, pts2,xpad = 0,ypad = 0):
    '''
    pts: [lefttop_x, lefttop_y, bottomright_x, bottomright_y]
    return True if two rectangles overlap
    '''
    # by separating axix theorem
    if xpad != 0:
        return not (pts1[0] > xpad + pts2[2] or pts1[1] + ypad < pts2[3] or pts1[2] + xpad < pts2[0] or pts1[3] > pts2[1] + ypad )
    else:
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
  
def _get_min_spacing(time_doc, lanes = [i*12 for i in range(-1,12)]):
    '''
    get the minimum x-difference at all lanes for a given timestamp
    TODO: consider vehicle dimension for space gap
    '''
    # get the lane assignments
    veh_ids = np.array(time_doc['id'])
    x_pos = np.array([pos[0] for pos in time_doc["position"]])
    y_pos = np.array([pos[1] for pos in time_doc["position"]])
    lane_asg = np.digitize(y_pos, lanes) # lane_id < 6: east

    # for each lane, sort by x - This is not the best way to calculate!
    lane_dict = defaultdict(list) # key: lane_id, val: x_pos
    for lane_id in np.unique(lane_asg):
        in_lane_idx = np.where(lane_asg==lane_id)[0] # idx of vehs that are in lane_id
        in_lane_ids = veh_ids[in_lane_idx]
        in_lane_xs = x_pos[in_lane_idx]
        sorted_idx = np.argsort(in_lane_xs)
        sorted_xs = in_lane_xs[sorted_idx]
        sorted_ids = in_lane_ids[sorted_idx] # apply the same sequence to ids
        lane_dict[lane_id] = [sorted_xs, sorted_ids]
    
    # get x diff for each lane
    # pprint.pprint(lane_dict)
    min_spacing = 10e6
    for lane_id, vals in lane_dict.items():
        try:
            sorted_xs, sorted_ids = vals
            delta_x = np.diff(sorted_xs)
            min_idx = np.argmin(delta_x)
            min_spacing_temp = delta_x[min_idx]
            if min_spacing_temp < min_spacing:
                min_spacing = min_spacing_temp
                min_pair = (sorted_ids[min_idx], sorted_ids[min_idx+1])
                
        except ValueError:
            pass
        
    return min_spacing 

    



class UnsupervisedEvaluator():
    
    def __init__(self, config, collection_name=None, num_threads=100):
        '''
        Parameters
        ----------
        config : Dictionary
            store all the database-related parameters.
        collection1 : str
            Collection name.
        '''
        # print(config)
        self.collection_name = collection_name
        
        client = DBClient(**config)
        db_time = client.client["transformed"]
        db_raw = client.client["trajectories"]
        db_rec = client.client["reconciled"]
        
        # print("N collections before transformation: {} {} {}".format(len(db_raw.list_collection_names()),len(db_rec.list_collection_names()),len(db_time.list_collection_names())))
        # start transform trajectory-indexed collection to time-indexed collection if not already exist
        # this will create a new collection in the "transformed" database with the same collection name as in "trajectory" database
        if collection_name not in db_time.list_collection_names(): # always overwrite
            print("Transform to time-indexed collection first")
            client.transform(read_database_name=config["database_name"], 
                      read_collection_name=collection_name)
           
        # print("N collections after transformation: {} {} {}".format(len(db_raw.list_collection_names()),len(db_rec.list_collection_names()),len(db_time.list_collection_names())))
        
        # print(config,collection_name)
        self.dbr_v = DBClient(**config, collection_name = collection_name)
        self.dbr_t = DBClient(host=config["host"], port=config["port"], username=config["username"], password=config["password"],
                              database_name = "transformed", collection_name = collection_name)
        print("connected to pymongo client")
        self.res = defaultdict(dict) # min, max, avg, stdev
        self.num_threads = num_threads
        
        self.res["collection"] = self.collection_name
        self.res["traj_count"] = self.dbr_v.count()
        self.res["timestamp_count"] = self.dbr_t.count()
            
       
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

        # distributions - all the functions that return a single value
        # TODO: put all functions in a separate script
        functions = [_get_duration, _get_x_traveled,
                      _get_y_traveled, _get_max_vx, _get_min_vx,
                      _get_max_vy, _get_min_vy,_get_max_ax,_get_min_ax,_get_avg_vx,_get_avg_vy,_get_avg_ax,_get_residual,
                      _get_vx, _get_ax,
                      _get_lane_changes]
        # functions = [_get_lane_changes]
        
        for fcn in functions:
            traj_cursor = self.dbr_v.collection.find({})
            res = self.thread_pool(fcn, iterable=traj_cursor) # cursor cannot be reused
            attr_name = fcn.__name__[5:]
            print(f"Evaluating {attr_name}...")
            if attr_name in ["vx", "ax"]:
                res = [item for sublist in res for item in sublist] # flatten the nested list
                
            self.res[attr_name]["min"] = np.nanmin(res).item()
            self.res[attr_name]["max"] = np.nanmax(res).item()
            self.res[attr_name]["median"] = np.nanmedian(res).item()
            self.res[attr_name]["avg"] = np.nanmean(res).item()
            self.res[attr_name]["stdev"] = np.nanstd(res).item()
            self.res[attr_name]["raw"] = res
            
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
            # space_gap = []

            # east_pts = M*east_b, where east_pts=[lx, ly, rx, ry], east_b =[x,y,len,wid]
            # vectorize to all vehicles: A = M*B
            try:
                east_pts = np.matmul(east_b, east_m)
            except ValueError: # ids are empty
                east_pts = []

            for i, pts1 in enumerate(east_pts):
                for j, pts2 in enumerate(east_pts[i+1:]):
                    # check if two boxes overlap, if so append the pair ids
                    if doOverlap(pts1, pts2):
                        overlap.append((str(east_ids[i]),str(east_ids[j])))

            # west bound
            try:
                west_pts = np.matmul(west_b, west_m)
            except ValueError:
                west_pts = []
            for i, pts1 in enumerate(west_pts):
                for j, pts2 in enumerate(west_pts[i+1:]):
                    # check if two boxes overlap
                    if doOverlap(pts1, pts2):
                        overlap.append((str(west_ids[i]),str(west_ids[j])))
                        
            return overlap

                    
        # start thread_pool for each timestamp
        functions = [_get_min_spacing, _get_overlaps]
        # functions = [_get_min_spacing]
        for fcn in functions:
            time_cursor = self.dbr_t.collection.find({})
            attr_name = fcn.__name__[5:]
            print(f"Evaluating {attr_name}...")
            if "overlap" in attr_name:
                overlaps = set()
                count = 0
                for time_doc in time_cursor:
                    if count % sample_rate == 0:
                        overlap_t = _get_overlaps(time_doc)
                        for pair in overlap_t:
                            overlaps.add(pair)
                    count += 1
                
                self.res[attr_name] = list(overlaps)
            else:
                res = self.thread_pool(fcn, iterable=time_cursor) 
                self.res[attr_name]["min"] = np.nanmin(res).item()
                self.res[attr_name]["max"] = np.nanmax(res).item()
                self.res[attr_name]["median"] = np.nanmedian(res).item()
                self.res[attr_name]["avg"] = np.nanmean(res).item()
                self.res[attr_name]["stdev"] = np.nanstd(res).item()
                self.res[attr_name]["raw"] = res
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
 
    
def plot_histogram(data, title=""):
    bins = min(int(len(data)/10), 100)
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.show()
    
def call(db_param,collection):    
    ue = UnsupervisedEvaluator(db_param, collection_name=collection, num_threads=200)
    t1 = time.time()
    ue.traj_evaluate()
    ue.time_evaluate(sample_rate = 25)
    t2 = time.time()
    
    print("time: ", t2-t1)
    #ue.print_res()
    ue.save_res()
    
    return ue.res
    
    
    
if __name__ == '__main__':
    # with open('config.json') as f:
    #     config = json.load(f)
      
    collection = "pristine_stork--RAW_GT1__mumbles"
    # collection = "morose_panda--RAW_GT1__juxtaposes"
    if "__" in collection:
        database_name = "reconciled"
    else:
        databas_name = "trajectories"
        
    param = {
      "host": "10.2.218.56",
      "port": 27017,
      "username": "i24-data",
      "password": "mongodb@i24",
      "database_name": database_name # db that the collection to evaluate is in
    }
    try:
        with open(f"res_{collection}.json", "rb") as f:
            res = json.load(f)
            print("loaded res from local json file")
    except:
        res = call(param, collection)
        
    # %% plot 
    plot_histogram(res["vx"]["raw"], "vx")
    
    # %% examine large accelerations
    # dbc = DBClient(**param, collection_name = collection)
    # col = dbc.collection 
    # for doc in col.find():
    #     r = _get_avg_ax(doc)
    #     if r and r < -50:
    #         print(doc["_id"])
            
    
    
    
