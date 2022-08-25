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
import torch

# =================== CALCULATE DYNAMICS ====================    
def _get_duration(traj):
    return traj["last_timestamp"] - traj["first_timestamp"]

def _get_x_traveled(traj):
    x = abs(traj["ending_x"] - traj["starting_x"])
    return x

    
def _get_y_traveled(traj):
    return max(traj["y_position"]) - min(traj["y_position"])


def _get_backward_cars(traj):
    dx = np.diff(traj["x_position"]) * traj["direction"]
    if np.any(dx < 0):
        return str(traj['_id'])
    return None


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
    return point-wise x-velocity
    '''
    dx = np.diff(traj["x_position"])
    dt = np.diff(traj["timestamp"])
    return np.abs(dx/dt)

def _get_vy(traj):
    '''
    return point-wise y-velocity
    '''
    dy = np.diff(traj["y_position"])
    dt = np.diff(traj["timestamp"])
    return dy/dt
    
def _get_rotation(traj):
    '''
    return point-wise vehicle rotation (degree) relative to the direction
    '''
    vx = np.abs(_get_vx(traj)) # non-negative
    vy = np.abs(_get_vy(traj)) # non-negative
    theta = np.arctan2(vy, vx)*(180/3.14)
    return theta


def _get_lane_changes(traj, lanes = [i*12 for i in range(-1,12)]):
    '''
    count number of times y position is at another lane according to lane marks
    '''
    lane_idx = np.digitize(traj["y_position"], lanes)
    lane_change = np.diff(lane_idx)
    # count-nonzeros
    return np.count_nonzero(lane_change)





# =================== GET OTHER INFO ====================    
def _calc_feasibility(traj, start_time, end_time, buffer=1, xmin=0, xmax=2000):
    '''
    for each of the following, assign traj a score between 0-1. 1 is good, 0 is bad
    distance: % x covered
        start_time, end_time are time boundaries. buffer time is the tolerance for which traj
        starts buffer after start_time or ends buffer before end_time are tolerated
    backwards: % time where dx/dt is negative
    rotation: % of time where theta is > 30degree
    acceleration: % of time where abs(accel) > 10ft/s2
    conflicts: % of time traj conflicts with others
    '''
    # x distance traveled
    end = end_time if traj["last_timestamp"] >= end_time-buffer else traj["last_timestamp"]  
    if traj['direction'] == 1:
        start = xmin if traj["first_timestamp"] <= start_time + buffer else traj["starting_x"]
        end = xmax if traj["last_timestamp"] >= end_time - buffer else traj["ending_x"]
    else:
        # xmax, xmin = xmin, xmax
        start = xmax if traj["first_timestamp"] <= start_time + buffer else traj["starting_x"]
        end = xmin if traj["last_timestamp"] >= end_time - buffer else traj["ending_x"]
    dist = min(1, abs(end-start)/(xmax-xmin))
    
    # backward occurances
    dx = np.diff(traj["x_position"]) * traj["direction"]
    backward = 1-np.sum(np.array(dx) < 0)/len(dx)
    
    # rotation
    theta = np.abs(_get_rotation(traj))
    rotation = 1-np.sum(np.array(theta) >= 30)/len(theta)
    
    # acceleration
    accel = np.abs(_get_ax(traj))
    acceleration = 1-np.sum(np.array(accel) > 10)/len(accel)
    
    # conflicts - get the max occurances now
    duration = _get_duration(traj)
    try:
        time_conflict = max([item[1] for item in traj["conflicts"]])
    except KeyError:
        time_conflict = 0
    conflict = 1-time_conflict/duration
    
    return [dist, backward, rotation, acceleration, conflict]
    
    
    
    
def _get_residual(traj):
    try:
        return traj["x_score"]
    except: # field is not available
        return 0


# def _get_post_flags(traj, flag_dict):
#     '''
#     flag_dict is a shared dictionary amongst threads
#     key: "flag_name", val: num_of_occurances
#     '''
#     try:
#         for flag in traj["post_flag"]:
#             flag_dict[flag] += 1
#     except KeyError: # no post_flag
#         pass
    

    
# =================== TIME-INDEXED CALCULATIONS ==================== 
       
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

def _get_distance_score(traj):
    return traj["feasibility"]["distance"]
def _get_backward_score(traj):
    return traj["feasibility"]["backward"]
def _get_acceleration_score(traj):
    return traj["feasibility"]["acceleration"]
def _get_rotation_score(traj):
    return traj["feasibility"]["rotation"]
def _get_conflict_score(traj):
    return traj["feasibility"]["conflict"]

def _get_all_feasible(traj):
    '''
    if all the scores are 1
    '''
    feas = []
    for key,val in traj["feasibility"].items():
        if key=="distance":
            val = val if val < 0.8 else 1 # tolerate >80% distance score
        feas.append(val)
        
    if all(np.array(feas)==1):
        return traj["_id"]
    else:
        return None
    
def _get_any_infeasible(traj):
    '''
    if any of the scores is not one
    '''
    feas = []
    for key,val in traj["feasibility"].items():
        if key=="distance":
            val = val if val < 0.8 else 1 # tolerate >80% distance score
        feas.append(val)
            
    if any(np.array(feas)!=1):
        return traj["_id"]
    else:
        return None
    
def _get_feasibility_score(traj):
    '''
    score = sumprod(all_scores)
    '''
    score = 1
    for key,val in traj["feasibility"].items():
        score *= val
    return score
    
    

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
        
    def thread_pool(self, func, iterable = None, kwargs=None):
        if iterable is None:
            iterable = self.dbr_v.collection.find({})
        
        pool = ThreadPool(processes=self.num_threads)
        res = []
        if kwargs is not None:
            for item in iterable:
                async_result = pool.apply_async(func, (item, ), kwargs) # tuple of args for foo
                res.append(async_result) 
        else:
            for item in iterable:
                async_result = pool.apply_async(func, (item, )) # tuple of args for foo
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
        functions_hist = [_get_duration, _get_x_traveled,
                      _get_avg_vx,_get_avg_ax,_get_residual,
                      _get_vx, _get_ax, _get_lane_changes,
                      _get_distance_score, _get_backward_score, _get_acceleration_score, _get_rotation_score, _get_conflict_score,_get_feasibility_score
                      ]
        
        for fcn in functions_hist:
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
            
        # get ids - all the functions that return ids if a condition is met
        functions_ids = [_get_backward_cars, _get_all_feasible, _get_any_infeasible]
        for fcn in functions_ids:
            attr_name = fcn.__name__[5:]
            print(f"Evaluating {attr_name}...")
            traj_cursor = self.dbr_v.collection.find({})
            res = self.thread_pool(fcn, iterable=traj_cursor) # cursor cannot be reused
            self.res[attr_name] = [r for r in res if r]
            
        # get flags - all functions that write to a shared variable
        # functions_flags = [_get_post_flags]
        # for fcn in functions_flags:
        #     attr_name = fcn.__name__[5:]
        #     print(f"Evaluating {attr_name}...")
        #     self.flag_dict = defaultdict(int)
        #     traj_cursor = self.dbr_v.collection.find({})
        #     res = self.thread_pool(fcn, iterable=traj_cursor, kwargs={"flag_dict": self.flag_dict}) 
        #     self.res[attr_name] = self.flag_dict
        
        # pprint.pprint(self.res["post_flags"])
        return 
        
    
    
    
    def time_evaluate(self, step=1):
        '''
        Evaluate using time-indexed collection
        step: (int) select every step timestamps to evaluate
        '''
        # matries to convert [x,y,len,wid] to [lefttop_x, lefttop_y, bottomright_x, bottomright_y]
        east_m = np.array([[1, 0,0,0], [0,1,0,0.5], [1,0,1,0], [0,1,0,-0.5]]).T
        west_m = np.array([[1,0,-1,0], [0,1,0,0.5], [1, 0,0,0], [0,1,0,-0.5]]).T
        
        # cache vehicle dimension
        
        if "__" in self.collection_name: # reconciled collection
            cache_dim = {}
            for traj in self.dbr_v.collection.find({}):
                cache_dim[traj["_id"]] = [traj["length"], traj["width"]]

        east_pipeline = [
            {"$match": {"direction": {"$eq":1}}},
            {'$project':{ '_id': 1 } },
                      ]
        query = self.dbr_v.collection.aggregate(east_pipeline)
        east_ids = set([doc["_id"] for doc in query])
        
        west_pipeline = [
            {"$match": {"direction": {"$eq":-1}}},
            {'$project':{ '_id': 1 } },
                      ]
        query = self.dbr_v.collection.aggregate(west_pipeline)
        west_ids = set([doc["_id"] for doc in query])


        def _get_overlaps(time_doc):
            '''
            Calculate pair-wise overlaps and space gap at a given timestamp
            '''
            veh_ids = time_doc['id']
            # curr_time = time_doc["timestamp"]
            curr_east_ids = [veh_id for veh_id in veh_ids if veh_id in east_ids]
            curr_west_ids = [veh_id for veh_id in veh_ids if veh_id in west_ids]
            pos = time_doc["position"]
            try:
                dims = time_doc["dimensions"]
                time_doc_dict = {veh_ids[i]: pos[i] + dims[i][:2] for i in range(len(veh_ids))} #key:id, val:[x,y,l,w]
            except KeyError:
                time_doc_dict = {veh_ids[i]: pos[i] + cache_dim[veh_ids[i]] for i in range(len(veh_ids))}

            
            east_b = np.array([time_doc_dict[veh_id] for veh_id in curr_east_ids])
            west_b = np.array([time_doc_dict[veh_id] for veh_id in curr_west_ids])
  
            overlap = []

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
                        # overlap.append((str(curr_east_ids[i]),str(curr_east_ids[i+j+1])))
                        overlap.append((curr_east_ids[i],curr_east_ids[i+j+1]))

            # west bound
            try:
                west_pts = np.matmul(west_b, west_m)
            except ValueError:
                west_pts = []
            for i, pts1 in enumerate(west_pts):
                for j, pts2 in enumerate(west_pts[i+1:]):
                    # check if two boxes overlap
                    if doOverlap(pts1, pts2):
                        # overlap.append((str(curr_west_ids[i]),str(curr_west_ids[i+j+1])))
                        overlap.append((curr_west_ids[i],curr_west_ids[i+j+1]))
            return overlap

                    
        # start thread_pool for each timestamp
        functions = [_get_min_spacing, _get_overlaps]
        # functions = [_get_min_spacing]
        for fcn in functions:
            time_cursor = self.dbr_t.collection.find({})
            attr_name = fcn.__name__[5:]
            print(f"Evaluating {attr_name}...")
            if "overlap" in attr_name:
                overlaps = defaultdict(int) # key: (conflict pair), val: num of timestamps
                count = 0
                for time_doc in time_cursor:
                    if count % step == 0:
                        overlap_t = _get_overlaps(time_doc)
                        for pair in overlap_t:
                            # overlaps.add(pair)
                            overlaps[pair]+=1
                    count += 1
                for pair, occurances in overlaps.items():
                    overlaps[pair] = occurances * step /25 # convert to cumulative seconds
                # pprint.pprint(overlaps, width = 1)
                self.res[attr_name] = overlaps

                # id2idx = {}
                # cntr = 0
                # for i, pair in enumerate(overlaps):
                #     for j, _id in enumerate(pair):
                #         if _id not in id2idx:
                #             id2idx[_id] = cntr
                #             cntr+=1
                
                # overlaps_m = torch.zeros([cntr, cntr])
                # for pair, val in overlaps.items():
                #     id1, id2 = pair
                #     overlaps_m[id2idx[id1], id2idx[id2]] = val
                    
                # self.res["overlap_matrix"] = overlaps_m
                    
            else:
                res = self.thread_pool(fcn, iterable=time_cursor) 
                self.res[attr_name]["min"] = np.nanmin(res).item()
                self.res[attr_name]["max"] = np.nanmax(res).item()
                self.res[attr_name]["median"] = np.nanmedian(res).item()
                self.res[attr_name]["avg"] = np.nanmean(res).item()
                self.res[attr_name]["stdev"] = np.nanstd(res).item()
                self.res[attr_name]["raw"] = res
        
        # write to database
        self.update_db()
        
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
        
        
    def update_db(self):
        '''
        add feasibility information to db
        '''
        # if "__" not in self.collection_name:
        #     print("Skip flagging for raw collection")
        #     return
        
        col = self.dbr_v.collection
        start_time = self.dbr_v.get_min("first_timestamp")
        end_time = self.dbr_v.get_max("last_timestamp")
        x_min = self.dbr_v.get_min("starting_x")
        x_max = self.dbr_v.get_max("ending_x")
        print("x_min: ", x_min, "x_max: ", x_max)
        
        # reset fields
        col.update_many({},{"$unset": {"post_flag": "",
                                       "conflicts": "",
                                       "feasibility": "",
                                       } })
        
        # update conflicts
        for pair, occurances in self.res["overlaps"].items():
            # {'$push': {'tags': new_tag}}, upsert = True)
            id1, id2 = pair
            col.update_one({"_id": id1}, 
                            {"$push": {"conflicts": [id2, occurances]}},
                            upsert = True)
            col.update_one({"_id": id2}, 
                            {"$push": {"conflicts": [id1, occurances],
                                       }},
                            upsert = True)
            
        # update feasibility
        cursor = col.find({})
        for traj in cursor:
            feas = _calc_feasibility(traj, start_time, end_time, xmin=x_min, xmax=x_max)
            dist, backward, rotation, acceleration, conflict = feas
            col.update_one({"_id": traj["_id"]}, 
                            {"$set": {"feasibility.distance": dist,
                                      "feasibility.backward": backward,
                                      "feasibility.rotation": rotation,
                                      "feasibility.acceleration": acceleration,
                                      "feasibility.conflict": conflict,
                                       }},
                            upsert = True)
            
        
        print("Updated feasibility for {}".format(self.collection_name))

        
            
 
    
def plot_histogram(data, title="", ):
    bins = min(int(len(data)/10), 300)
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.show()
    
    
def call(db_param,collection): 
    
    ue = UnsupervisedEvaluator(db_param, collection_name=collection, num_threads=200)
    t1 = time.time()
    ue.time_evaluate(step = 1)
    ue.traj_evaluate()
    t2 = time.time()
    print("time: ", t2-t1)
    
    #ue.print_res()
    #ue.save_res()
    
    return ue.res
    

def conflict_graph(collection):
    '''
    visualize the relationship between conflicts
    '''
    import networkx as nx
    G = nx.DiGraph()
    # queries = collection.find({"conflicts":{"$exists": True}})
    queries = collection.find({"$or": [{"feasibility.distance":{"$lte":0.7}},
                                      {"conflicts":{"$exists":True}}]})
    
    for traj in queries:
        if traj["feasibility"]["distance"] < 0.7:
            color = "red"
        else:
            color = "#210070"
        G.add_node(traj["_id"], weight=len(traj["timestamp"]), color=color)
        try:
            edges = traj["conflicts"]
            dur = traj["last_timestamp"]-traj["first_timestamp"]
            for nei, edge_weight in edges:
                G.add_edge(traj["_id"], nei, weight=edge_weight/dur)
        except KeyError: # no conflicts
            pass
    
    # clean edges to keep only the larger weight of bi-direction
    remove_list = []
    for u,v,w in G.edges(data="weight"):
        if w > G.edges[(v,u)]["weight"]:
            remove_list.append((v,u))
        else:
            remove_list.append((u,v))
    G.remove_edges_from(remove_list)

    # visualize the graph
    fig, ax = plt.subplots(figsize=(12, 12))
    # Generate layout for visualization
    # pos = nx.kamada_kawai_layout(G)
    pos = nx.spring_layout(G, weight=0.00001)
    # Visualize graph components
    edgewidth = [e[2]*10 for e in G.edges(data="weight")]
    nodesize = [v for _,v in G.nodes(data="weight")]
    nodecolor = [v for _,v in G.nodes(data="color")]
    
    nx.draw_networkx_edges(G, pos, alpha=0.7, width=edgewidth, edge_color="m")
    nx.draw_networkx_nodes(G, pos, node_size=nodesize, node_color=nodecolor, alpha=0.5)
    # label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
    # nx.draw_networkx_labels(G, pos, font_size=6, bbox=label_options)


    # Title/legend
    font = {"fontname": "Helvetica", "color": "k", "fontweight": "bold", "fontsize": 10}
    ax.set_title("Conflict graph for {}".format(collection._Collection__name), font)
    # Change font color for legend
    font["color"] = "k"
    
    ax.text(
        0.80,
        0.10,
        "edge width = conflicting time/traj duration",
        horizontalalignment="center",
        transform=ax.transAxes,
        fontdict=font,
    )
    ax.text(
        0.80,
        0.08,
        "node size = traj length",
        horizontalalignment="center",
        transform=ax.transAxes,
        fontdict=font,
    )
    ax.text(
        0.80,
        0.06,
        "# nodes: {}, # edges: {}".format(len(G.nodes), len(G.edges)),
        horizontalalignment="center",
        transform=ax.transAxes,
        fontdict=font,
    )
    ax.text(
        0.80,
        0.04,
        "red nodes: short tracks (<0.7)",
        horizontalalignment="center",
        transform=ax.transAxes,
        fontdict=font,
    )
    
    # Resize figure for label readibility
    ax.margins(0.1, 0.05)
    fig.tight_layout()
    plt.axis("off")
    plt.show()
    
if __name__ == '__main__':
    # with open('config.json') as f:
    #     config = json.load(f)
      
    collection = "young_ox--RAW_GT2__calibrates"
    # collection = "sanctimonious_beluga--RAW_GT1__administers"
    if "__" in collection:
        database_name = "reconciled"
    else:
        database_name = "trajectories"
    db_param= {
        "host": "127.0.0.1",
        "port": 27017,
        "username": "mongo-admin",
        "password": "i24-data-access",
        "database_name": database_name
    }    
    # db_param = {
    #   "host": "10.2.218.56",
    #   "port": 27017,
    #   "username": "i24-data",
    #   "password": "mongodb@i24",
    #   "database_name": database_name # db that the collection to evaluate is in
    # }
    

    # res = call(db_param, collection)
    # ue = UnsupervisedEvaluator(db_param, collection_name=collection, num_threads=200)
    # ue.time_evaluate(step = 1)
    # ue.traj_evaluate()
    
    # print("all_feasible: ", len(res["all_feasible"]))
    # print("any_infeasible: ", len(res["any_infeasible"]))
        
    # %% plot 
    # plot_histogram(ue.res["feasibility_score"]["raw"], "conflict_score")
    # plot_histogram(ue.res["distance_score"]["raw"], "distance_score")
    
    dbc = DBClient(**db_param, collection_name=collection)
    conflict_graph(dbc.collection)
    
    
    
    
