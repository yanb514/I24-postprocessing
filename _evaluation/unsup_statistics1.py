#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 21:45:29 2022
@author: yanbing_wang

1. time-eval first
2. update conflicts information to database
3. _calc_feasibility for each trajectory
4. aggreagate results and update feasibility to each trajectory
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
from pymongo import UpdateOne
import os


# =================== CALCULATE PER TRAJECOTRY ====================      
def _calc_update_feasibility(traj, start_time=None, end_time=None, xmin=None, xmax=None, buffer=1):
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
    # duration
    duration = traj["last_timestamp"] - traj["first_timestamp"]
    
    # x_traveled
    x_traveled = abs(traj["ending_x"] - traj["starting_x"])

    # ids of backward traveled car (any time range)
    dx = np.diff(traj["x_position"]) * traj["direction"]
    if np.any(dx < 0):
        backward_id = traj['_id']
    else:
        backward_id = None
       
    # average vx
    dt = np.diff(traj["timestamp"])
    try: avg_vx = np.abs(np.average(dx/dt))
    except: avg_vx = np.nan

    # average ax
    ddx = np.diff(traj["x_position"], 2)
    try: avg_ax = np.mean(ddx/(dt[:-1]**2))
    except: avg_ax = np.nan
    
    # point-wise vx
    vx = np.abs(dx/dt)
    
    # point-wise ax
    ax = ddx/(dt[:-1]**2)

    # point-wise vy
    dy = np.diff(traj["y_position"])
    vy = dy/dt
    
    # rotation
    theta = np.arctan2(np.abs(vy), np.abs(vx))*(180/3.14)
    
    # scores
    # -- x distance traveled (with forgiveness on boundaries)
    end = end_time if traj["last_timestamp"] >= end_time-buffer else traj["last_timestamp"]  
    if traj['direction'] == 1:
        start = xmin if traj["first_timestamp"] <= start_time + buffer else traj["starting_x"]
        end = xmax if traj["last_timestamp"] >= end_time - buffer else traj["ending_x"]
    else:
        # xmax, xmin = xmin, xmax
        start = xmax if traj["first_timestamp"] <= start_time + buffer else traj["starting_x"]
        end = xmin if traj["last_timestamp"] >= end_time - buffer else traj["ending_x"]
        
    # # x distance traveled (no forgiveness)
    # start = traj["starting_x"]
    # end = traj["ending_x"]
    dist = min(1, abs(end-start)/(xmax-xmin))
    # backward occurances
    backward = 1-np.sum(np.array(dx) < 0)/len(dx)
    # rotation
    rotation = 1-np.sum(np.array(np.abs(theta)) >= 30)/len(theta)
    # acceleration
    acceleration = 1-np.sum(np.array(np.abs(ax)) > 10)/len(ax)  
    # conflicts - get the max occurances now
    try:
        time_conflict = max([item[1] for item in traj["conflicts"]])
    except KeyError:
        time_conflict = 0
    conflict = 1-time_conflict/duration
    
    feasibility = [dist, backward, rotation, acceleration, conflict]
    dist_tol = dist if dist < 0.8 else 1
    feasibility_tolerated  = np.array([dist_tol, backward, rotation, acceleration, conflict])
    
    # residual
    try: residual = traj["x_score"]
    except: # field is not available
        residual = 0
    
    # feasibility-related
    # -- all_feasible: scores are all 1
    all_feasible_id = traj["_id"] if all(feasibility_tolerated==1) else None
    
    # -- any infeasible: if any score is not 1 
    any_infeasible_id = traj["_id"] if any(feasibility_tolerated!=1) else None
    
    # -- total_score = sumprod(feasibility scores)
    score = np.prod(feasibility_tolerated)
    
    #TODO: add the update
    query = {"_id": traj["_id"]}
    update = {"$set": {"feasibility.distance": dist,
              "feasibility.backward": backward,
              "feasibility.rotation": rotation,
              "feasibility.acceleration": acceleration,
              "feasibility.conflict": conflict,
               }}
    update_cmd= UpdateOne(filter=query, update=update, upsert=True)
    
    attr_vals = {"duration": duration, "x_traveled":x_traveled, "avg_vx":avg_vx, "avg_ax":avg_ax, 
                    "vx":vx, "ax":ax, "theta":theta, "feasibility":feasibility_tolerated, 
                    "residual": residual, "all_feasible_id":all_feasible_id, "any_infeasible_id":any_infeasible_id, "backward_id": backward_id,
                    "feasibility_score": score, 
                    "backward_score": backward,
                    "distance_score": dist_tol,
                    "rotation_score": rotation,
                    "acceleration_score": acceleration,
                    "conflict_score": conflict,
                    "update_cmd":update_cmd}

    return attr_vals
    
    # return duration, x_dist, backward_id, avg_vx, avg_ax, vx, ax, theta, feasibility, residual, all_feasible_id, any_infeasible_id, score, query, update
    

    

    
# =================== TIME-INDEXED CALCULATIONS ==================== 
       
def doOverlap(pts1, pts2,xpad = 0,ypad = 0):
    '''
    pts: [lefttop_x, lefttop_y, bottomright_x, bottomright_y]
    return True if two rectangles overlap
    '''
    # by separating axix theorem
    # left hand rule - y sign is flipped
    # pts1[1], pts1[3], pts2[1], pts2[3] = -pts1[1], -pts1[3], -pts2[1], -pts2[3] # comment out this line if right hand rule
    return not (pts1[0] > xpad + pts2[2] or pts1[1] + ypad < pts2[3] or pts1[2] + xpad < pts2[0] or pts1[3] > pts2[1] + ypad )


def calc_space_gap(pts1, pts2):
    '''
    pts: [lefttop_x, lefttop_y, bottomright_x, bottomright_y]
    if two cars are on the same lane, get the space gap
    '''
    if abs(pts1[1] + pts1[3] - pts2[1] - pts2[3])/2 < 6: # if two cars are likely to be on the same lane
        return max(pts2[0] - pts1[2], pts1[0] - pts2[2])
    else:
        return None

    

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
        db_time = client.client["transformed_beta"]
        
        # print("N collections before transformation: {} {} {}".format(len(db_raw.list_collection_names()),len(db_rec.list_collection_names()),len(db_time.list_collection_names())))
        # start transform trajectory-indexed collection to time-indexed collection if not already exist
        # this will create a new collection in the "transformed" database with the same collection name as in "trajectory" database
        if collection_name not in db_time.list_collection_names(): # always overwrite
            print("Transform to time-indexed collection first")
            client.transform2(read_database_name=config["database_name"], 
                      read_collection_name=collection_name)
           
        # print("N collections after transformation: {} {} {}".format(len(db_raw.list_collection_names()),len(db_rec.list_collection_names()),len(db_time.list_collection_names())))
        
        # print(config,collection_name)
        self.dbr_v = DBClient(**config, collection_name = collection_name)
        self.dbr_t = DBClient(host=config["host"], port=config["port"], username=config["username"], password=config["password"],
                              database_name = "transformed_beta", collection_name = collection_name)
        print("connected to pymongo client")
        self.res = defaultdict(dict) # min, max, avg, stdev
        self.num_threads = num_threads
        
        self.res["collection"] = self.collection_name
        self.res["traj_count"] = self.dbr_v.est_count()
        self.res["timestamp_count"] = self.dbr_t.est_count()
        
        
            
       
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
        # reset fields
        self.dbr_v.collection.update_many({},{"$unset": {
                                               "feasibility": "",
                                       } })
        
        # distributions - all the functions that return a single value
        # TODO: put all functions in a separate script
        print("Evaluating trajectories...")
        traj_cursor = self.dbr_v.collection.find({})#.limit(10)
        start_time = self.dbr_v.get_min("first_timestamp")
        end_time = self.dbr_v.get_max("last_timestamp")
        x_min = self.dbr_v.get_min("starting_x")
        x_max = self.dbr_v.get_max("ending_x")
        print("x_min: ", x_min, "x_max: ", x_max)
        
        kwargs = {"start_time": start_time,
                  "end_time": end_time,
                  "xmin": x_min,
                  "xmax": x_max}

        res = self.thread_pool(_calc_update_feasibility, iterable=traj_cursor, kwargs=kwargs) # cursor cannot be reused
        single_attrs = [ "avg_vx", "avg_ax",  "duration", "x_traveled", "residual" , "feasibility_score", 
                            "backward_score",
                            "distance_score",
                            "rotation_score",
                            "acceleration_score",
                            "conflict_score"]
        series_attrs = ["vx", "ax", "theta"]
        ids_attrs =  ["backward_id", "all_feasible_id", "any_infeasible_id"]
        
        # initialize empty lists
        for attr_name in ids_attrs:
            self.res[attr_name] = [] 

        bulk_update = []
        
        # unpack res
        for attr_vals in res:
            # duration, x_traveled, backward, avg_vx, avg_ax, vx, ax, theta, feasibility, residual, all_feasible_id, any_infeasible_id, score, query, update = item
            for attr_name in single_attrs:
                try: self.res[attr_name]["raw"].append(attr_vals[attr_name])
                except: self.res[attr_name]["raw"] = [attr_vals[attr_name]]
            for attr_name in series_attrs:
                try: self.res[attr_name]["raw"].extend(attr_vals[attr_name])
                except: self.res[attr_name]["raw"] = attr_vals[attr_name]
            for attr_name in ids_attrs:
                if attr_vals[attr_name]:
                    self.res[attr_name].append(attr_vals[attr_name])

            
            bulk_update.append(attr_vals["update_cmd"])
            
        for attr_name in single_attrs+series_attrs:
            self.res[attr_name]["min"] = np.nanmin(self.res[attr_name]["raw"]).item()
            self.res[attr_name]["max"] = np.nanmax(self.res[attr_name]["raw"]).item()
            self.res[attr_name]["median"] = np.nanmedian(self.res[attr_name]["raw"]).item()
            self.res[attr_name]["avg"] = np.nanmean(self.res[attr_name]["raw"]).item()
            self.res[attr_name]["stdev"] = np.nanstd(self.res[attr_name]["raw"]).item()
        
        # write feasibility to collection
        print("writing feasibility to collection")
        self.dbr_v.collection.bulk_write(bulk_update, ordered=False)
        return 
        
    
    
   
    def time_evaluate2(self, step=1):
        '''
        Evaluate using time-indexed collection from transformed_beta database
        step: (int) select every [step] timestamps to evaluate
        '''
        # reset fields
        self.dbr_v.collection.update_many({},{"$unset": {
                                               "conflicts": "",
                                       } })
        
        # matries to convert [centerx,centery,len,wid] to [lefttop_x, lefttop_y, bottomright_x, bottomright_y]
        M = np.array([[1, 0,-0.5,0], [0,1,0,0.5], [1,0,0.5,0], [0,1,0,-0.5]]).T # right hand rule
        # M = np.array([[1, 0,-0.5,0], [0,1,0,-0.5], [1,0,0.5,0], [0,1,0,0.5]]).T # left hand rule

        def _get_overlaps(time_doc):
            '''
            Calculate pair-wise overlaps and space gap at a given timestamp
            time_doc has the schema in transformed_beta
            '''
            try:
                eb = time_doc["eb"]
            except KeyError:
                # print(time_doc["timestamp"], " has no eb")
                eb = {}
                
            try:
                wb = time_doc["wb"]
            except KeyError:
                # print(time_doc["timestamp"], " has no wb")
                wb = {}
                
            
            eb_ids, wb_ids, east_b, west_b = [],[],[],[]
            
            for _id,val in eb.items():
                eb_ids.append(_id)
                east_b.append(val[:4])
                
            for _id,val in wb.items():
                wb_ids.append(_id)
                west_b.append(val[:4])
                
            east_b = np.array(east_b) # [centerx, centery, l,w]
            west_b = np.array(west_b)
            
            overlap = []

            # east_pts = M*east_b, where east_pts=[lx, ly, rx, ry], east_b =[x,y,len,wid]
            # vectorize to all vehicles: A = M*B
            try:
                east_pts = np.matmul(east_b, M)
                
            except ValueError: # ids are empty
                east_pts = []

            for i, pts1 in enumerate(east_pts):
                for j, pts2 in enumerate(east_pts[i+1:]):
                    # check if two boxes overlap, if so append the pair ids
                    if doOverlap(pts1, pts2):
                        # overlap.append((str(curr_east_ids[i]),str(curr_east_ids[i+j+1])))
                        overlap.append((eb_ids[i], eb_ids[i+j+1]))
                        # print((eb_ids[i], eb_ids[i+j+1]))

            # west bound
            try:
                west_pts = np.matmul(west_b, M)
            except ValueError:
                west_pts = []
                
            for i, pts1 in enumerate(west_pts):
                for j, pts2 in enumerate(west_pts[i+1:]):
                    # check if two boxes overlap
                    if doOverlap(pts1, pts2):
                        # overlap.append((str(curr_west_ids[i]),str(curr_west_ids[i+j+1])))
                        overlap.append((wb_ids[i], wb_ids[i+j+1]))
                        # print((wb_ids[i], wb_ids[i+j+1]))
            return overlap

                    
        # start thread_pool for each timestamp
        functions = [_get_overlaps]
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
                    
            else:
                res = self.thread_pool(fcn, iterable=time_cursor) 
                self.res[attr_name]["min"] = np.nanmin(res).item()
                self.res[attr_name]["max"] = np.nanmax(res).item()
                self.res[attr_name]["median"] = np.nanmedian(res).item()
                self.res[attr_name]["avg"] = np.nanmean(res).item()
                self.res[attr_name]["stdev"] = np.nanstd(res).item()
                self.res[attr_name]["raw"] = res
        
        # write to database
        # update conflicts
        bulk_update = []
        for pair, occurances in self.res["overlaps"].items():
            # {'$push': {'tags': new_tag}}, upsert = True)
            id1, id2 = pair
            if isinstance(id1, str):
                id1, id2 = ObjectId(id1), ObjectId(id2)
            query = {"_id": id1}
            update = {"$push": {"conflicts": [id2, occurances]}}
            update_cmd= UpdateOne(filter=query, update=update, upsert=True)
            bulk_update.append(update_cmd)
            
            query = {"_id": id2}
            update = {"$push": {"conflicts": [id1, occurances]}}
            update_cmd= UpdateOne(filter=query, update=update, upsert=True)
            bulk_update.append(update_cmd)
            
        print("writing conflicts to collection")
        self.dbr_v.collection.bulk_write(bulk_update, ordered=False)
        
        return
    
    
    
    def print_res(self):
        pprint.pprint(self.res, width = 1)
    
    
    
def plot_histogram(data, title="", ):
    bins = min(int(len(data)/10), 300)
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.show()
    
    
def call(db_param,collection,step=5): 
    
    ue = UnsupervisedEvaluator(db_param, collection_name=collection, num_threads=200)
    # t1 = time.time()
    # ue.time_evaluate2(step=step)
    # t2 = time.time()
    # print("Time-evaluate takes: ", t2-t1)
    
    t1 = time.time()
    ue.traj_evaluate()
    t2 = time.time()
    print("Traj-evaluate takes: ", t2-t1)
    
    #ue.print_res()
    #ue.save_res()
    
    return ue.res
    

def conflict_graph(collection):
    '''
    visualize the relationship between conflicts
    '''
    import networkx as nx
    G = nx.DiGraph()
    queries = collection.find({"conflicts":{"$exists": True}})
    # queries = collection.find({"$or": [{"feasibility.distance":{"$lte":0.7}},
    #                                   {"conflicts":{"$exists":True}}]})
    
    for traj in queries:
        # if traj["feasibility"]["distance"] < 0.7:
        # if len(traj["merged_ids"]) > len(traj["fragment_ids"]):
        #     color = "red"
        # else:
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

    database_name = "trajectories"
    collection = "635997ddc8d071a13a9e5293"
    # collection = "634ef772f8f31a6d48eab58e"
    
    with open(os.path.join(os.environ["USER_CONFIG_DIRECTORY"], "db_param.json")) as f:
        db_param = json.load(f)
    db_param["database_name"] = database_name    

    res = call(db_param, collection, step=5) # based on 25hz data, step=5 means downsample to 5hz
    
    # ue = UnsupervisedEvaluator(db_param, collection_name=collection, num_threads=200)
    # ue.time_evaluate(step = 1)
    # ue.traj_evaluate()
    
    # print("all_feasible: ", len(res["all_feasible"]))
    # print("any_infeasible: ", len(res["any_infeasible"]))
        
    # %% plot 
    # plot_histogram(ue.res["feasibility_score"]["raw"], "conflict_score")
    # plot_histogram(ue.res["distance_score"]["raw"], "distance_score")
    
    # dbc = DBClient(**db_param, collection_name=collection)
    # conflict_graph(dbc.collection)
    
    
    
    
