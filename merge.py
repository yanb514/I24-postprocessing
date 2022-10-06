#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 14:32:37 2022

@author: yanbing_wang
MERGE fragments that have space-time overlaps

if two fragments have time overlap:
    resample them to 25hz
    consider only the time overlapping part
    calculate the bhattar distance on the overlap
    if below a threshold, consider merging them
"""
import networkx as nx
import numpy as np
import queue
import pandas as pd
from collections import OrderedDict
from functools import reduce
import time
from utils.misc import calc_fit_select, find_overlap_idx

# from i24_database_api import DBClient
import i24_logger.log_writer as log_writer
from i24_logger.log_writer import catch_critical
from utils.utils_stitcher_cost import bhattacharyya_distance
from utils.misc import SortedDLL
# import warnings
# warnings.filterwarnings('error')



@catch_critical(errors = (RuntimeWarning))
def merge_resample(traj, conf_threshold):
    '''
    traj: dict
    preprocess procedures
    1. filter based on conf
    2. resample to df
    return df
    TODO: DEAL WITH NAN after conf mask
    '''
    # try:
    conf = np.array(traj["detection_confidence"])
    # except KeyError:
    #     conf = np.ones(len(traj["timestamp"]))
    
    # get confidence mask
    highconf_mask = np.array(conf >= conf_threshold)
    num_highconf = np.count_nonzero(highconf_mask)
    if num_highconf < 4:
        return None
       
    time_series_field = ["timestamp", "x_position", "y_position", "length", "width", "height"]
    for key in time_series_field:
        traj[key] = np.array(traj[key])[highconf_mask]
        
    # resample to df
    data = {key: traj[key] for key in time_series_field} 
    df = pd.DataFrame(data, columns=data.keys()) 
    index = pd.to_timedelta(df["timestamp"], unit='s')
    df = df.set_index(index)
    df = df.drop(columns = "timestamp")
    
    # resample to 25hz
    df=df.groupby(df.index.floor('0.04S')).mean().resample('0.04S').asfreq()
    df.index = df.index.values.astype('datetime64[ns]').astype('int64')*1e-9
    df = df.interpolate(method='linear')
    
    # write to dict
    # TODO: cast to np array instead
    for key in df.columns:
        traj[key] = df[key].values


    traj["timestamp"] = df.index.values
    traj["first_timestamp"] = traj["timestamp"][0]
    traj["last_timestamp"] = traj["timestamp"][-1]
    traj["starting_x"] = traj["x_position"][0]
    traj["ending_x"] = traj["x_position"][-1]

    return traj


        
@catch_critical(errors = (Exception))
def merge_cost(track1, track2):
    '''
    track1 and 2 have to be resmplaed first
    only deals with two tracks that have time overlap!
    speeded up bhartt cost
    '''
    t1 = track1["timestamp"]#[filter1]
    t2 = track2["timestamp"]#[filter2]
    
    gap = t2[0] - t1[-1] 
    if gap >= 0: # if no time overlap, don't merge -> stitcher's job
        return 1e6
    
    s1, e1, s2, e2 = find_overlap_idx(t1, t2)
    x1,x2,y1,y2 = track1["x_position"], track2["x_position"], track1["y_position"], track2["y_position"]
    
    # check if the overalpped position deviates too much
    if np.nanmean(np.abs(x1[s1:e1] - x2[s2:e2])) > 30 or np.nanmean(np.abs(y1[s1:e1] - y2[s2:e2])) > 6:
        return 1e6
    
    # try to vectorize
    mu1_arr = np.array([x1[s1:e1], y1[s1:e1]]) # 2xK
    mu2_arr = np.array([x2[s2:e2], y2[s2:e2]]) # 2xK
    cov1 = np.diag([np.nanmean(track1["length"][s1:e1]), np.nanmean(track1["width"][s1:e1])])*1 #2x2
    cov2 = np.diag([np.nanmean(track2["length"][s2:e2]), np.nanmean(track2["width"][s2:e2])])*1 #2x2
    cov = (cov1 + cov2)/2
    mu = mu1_arr-mu2_arr # 2xK, difference
    inv_cov = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    det1 = np.linalg.det(cov1)
    det2 = np.linalg.det(cov2)
    nll = 0.125 * np.sum((mu.T@inv_cov@mu).diagonal())/(e1-s1+1) + 0.5 * np.log(det/np.sqrt(det1 * det2))
    
    # print("id1: {}, id2: {}, cost:{:.2f}".format(str(track1['_id'])[-4:], str(track2['_id'])[-4:], nll))
    # print("")
    
    return nll



def combine_merged_dict(unmerged):
    '''
    unmerged: a list of fragment-dicts
    '''
    if len(unmerged) == 1:
        traj = unmerged[0]
        traj["merged_ids"] = [traj["_id"]]
        return traj
    
    time_series_field = ["timestamp", "x_position", "y_position", "length", "width", "height"]
    dfs = []
    merged_ids = []
    for traj in unmerged:
        merged_ids.append(traj["_id"])
        data = {key: traj[key] for key in time_series_field} 
        df = pd.DataFrame(data, columns=data.keys()) 
        index = pd.to_timedelta(df["timestamp"], unit='s')
        df = df.set_index(index)
        df = df.drop(columns = "timestamp")
        dfs.append(df)

    df_merged = pd.concat(dfs).groupby(level=0, as_index=True, sort=True).mean()
    # merged = df_merged.to_dict('list')
    # overwrite traj with merged values
    traj = unmerged[0]
    for key in df_merged.columns:
        traj[key] = df_merged[key].values

    traj["merged_ids"] = merged_ids
    traj["timestamp"] = df_merged.index.values.astype('datetime64[ns]').astype('int64')*1e-9
    traj["first_timestamp"] = traj["timestamp"][0]
    traj["last_timestamp"] = traj["timestamp"][-1]
    traj["starting_x"] = traj["x_position"][0]
    traj["ending_x"] = traj["x_position"][-1]
    
    # print(len(traj["timestamp"]), len(traj["x_position"]))
    return traj



def dummy_merge(direction, fragment_queue, merged_queue, parameters):
    '''
    only resample data and write directly to the queue
    '''

    CONF_THRESH = parameters["conf_threshold"]
    TIMEOUT = parameters["merger_timeout"]
    
    while True:
        try:
            fragment = fragment_queue.get(timeout = TIMEOUT) # fragments are ordered in last_timestamp
        except queue.Empty:
            break
    
    merged = merge_resample(fragment, CONF_THRESH)
    merged_queue.put(merged)
    return

    
    
@catch_critical(errors = (RuntimeWarning))
def overlap_cost(df1, df2):
    '''
    use bhattacharyya_distance
    df1, df2 are dataframes with index of timestampss
    '''
    # no time overlap, return
    if df2.index[0] > df1.index[-1] or df2.index[-1] < df1.index[0]:
        return 1e6
    
    df3 = pd.merge(df1, df2, left_index=True, right_index=True) # inner join
    
    bd = []
    for i in range(len(df3)):
        mu1 = np.array([df3["x_position_x"].values[i], df3["y_position_x"].values[i]]) # track1
        mu2 = np.array([df3["x_position_y"].values[i], df3["y_position_y"].values[i]]) # track2
        cov1 = np.diag([df3["length_x"].values[i], df3["width_x"].values[i]]) # x-variance scales with length, y variance scales with width
        cov2 = np.diag([df3["length_y"].values[i], df3["width_y"].values[i]])

        bd.append(bhattacharyya_distance(mu1, mu2, cov1, cov2))
    

    nll = np.mean(bd)
    # except:
    #     print(df1.index.values-1.62808000e+09)
    #     print(df2.index.values-1.62808000e+09)
    
    # print("id1: {}, id2: {}, cost:{:.2f}".format(str(track1['_id'])[-4:], str(track2['_id'])[-4:], nll))
    # print("")
    
    return nll

    
@catch_critical(errors = (RuntimeWarning))   
def combine_merged(unmerged):
    '''
    unmerged: a list of (fragment-dict, fragment-df) tuples
    '''
    if len(unmerged) == 1:
        traj, df = unmerged[0]
        merged = df.to_dict('list')
        for key,val in merged.items():
            traj[key] = val
            
        traj["timestamp"] = list(df.index.values)
        traj["first_timestamp"] = traj["timestamp"][0]
        traj["last_timestamp"] = traj["timestamp"][-1]
        traj["starting_x"] = merged["x_position"][0]
        traj["ending_x"] = merged["x_position"][-1]
        traj["merged_ids"] = [traj["_id"]]
        
        # print(len(traj["timestamp"]), len(traj["x_position"]))   
        return traj
    
    dfs = []
    merged_ids = []
    for traj, df in unmerged:
        merged_ids.append(traj["_id"])
        dfs.append(df)

    
    # df_merged = reduce(lambda  left,right: pd.merge(left,right, left_index=True, right_index=True,
    #                                         how='outer'), dfs).mean()
    df_merged = pd.concat(dfs).groupby(level=0, as_index=True, sort=True).mean()
    merged = df_merged.to_dict('list')
    # overwrite traj with merged values
    traj = unmerged[0][0]
    for key,val in merged.items():
        traj[key] = val

    traj["merged_ids"] = merged_ids
    traj["timestamp"] = list(df_merged.index.values)
    traj["first_timestamp"] = traj["timestamp"][0]
    traj["last_timestamp"] = traj["timestamp"][-1]
    traj["starting_x"] = merged["x_position"][0]
    traj["ending_x"] = merged["x_position"][-1]
    
    # print(len(traj["timestamp"]), len(traj["x_position"]))
    return traj




def merge_fragments(direction, fragment_queue, merged_queue, parameters):
    '''
    graph structure
    if two nodes should be merged (by bhattar distance measure), then they are connected in graph
    node:
        node_id: fragment["_id"]
        raw: fragment dict
        data: resampled fragment df
    '''
    merge_logger = log_writer.logger
    merge_logger.set_name("merger_"+direction)
    merge_logger.info("Process started")
    
    DIST_THRESH = parameters["merge_thresh"] # bhattar distance distance threshold
    TIMEWIN = parameters["time_win"]
    CONF_THRESH = parameters["conf_threshold"]
    TIMEOUT = parameters["merger_timeout"]
    
    G = nx.Graph() # merge graph, two nodes are connected if they can be merged
    cntr = 0
    # TODO: keep a heap
    # lru = OrderedDict() # order connected components by last_timestamp
    sdll = SortedDLL()
    while True:
        try:
            fragment = fragment_queue.get(timeout = TIMEOUT) # fragments are ordered in last_timestamp
            cntr += 1
        except queue.Empty:
            merge_logger.warning("merger timed out after {} sec.".format(TIMEOUT))
            comps = nx.connected_components(G)
            for comp in comps:
                if len(comp) > 1:
                    merge_logger.debug("Merged {}".format(comp))
                unmerged = [ G.nodes[v]["data"] for v in list(comp)]
                merged = combine_merged_dict(unmerged)
                merged_queue.put(merged) 
            break
        
        resampled = merge_resample(fragment, CONF_THRESH) # convert to df ! last_timestamp might be changed!!!
            
        if resampled is None:
            merge_logger.info("skip {} in merging: LOW CONF".format(fragment["_id"]))
            continue
        
        
        curr_time = resampled["last_timestamp"]
        curr_id = resampled["_id"]
        
        # lru[fragment["_id"]] = curr_time # TODO DF LAST
        sdll.append({"id": curr_id, "tail_time": curr_time})
        
        curr_nodes = list(G.nodes(data=True)) # current nodes in G
        G.add_node(curr_id, data=resampled) 
        
        t1 = time.time()
        for node_id, node in curr_nodes:
            # if they have time overlaps
            dist = merge_cost(node["data"], resampled) # TODO: these two are not ordered in time,check time overlap within
            # print(str(node_id)[-4:], str(curr_id)[-4:], dist)
            if dist <= DIST_THRESH:
                G.add_edge(node_id, curr_id, weight = dist)
                # lru[node_id] = curr_time # TODO: reset to the larger last-tiemstamp of the two
                # lru.move_to_end(node_id, last=True) # move node_id to last of lru
                # larger = max([G.nodes[node_id]["data"]["last_timestamp"], G.nodes[curr_id]["data"]["last_timetsamp"]])
                # sdll.update(key=fragment["_id"], attr_val=larger)
                sdll.update(key=curr_id, attr_val=curr_time)
                sdll.update(key=node_id, attr_val=curr_time)
                
                # merge_logger.info("Merged {} and {}".format(node_id, fragment["_id"]))
        
        t2 = time.time()
        
        # find connected components in G and check for timeout
        # TODO: use better data structure to make the following more efficient
        if cntr % 10 == 0:
            merge_logger.debug("Graph nodes : {}, Graph edges: {}, cache: {}".format(G.number_of_nodes(), G.number_of_edges(), sdll.count()),extra = None)
            merge_logger.debug("Time elapsed for adding edge: {:.2f}".format(t2-t1))
            # t3 = time.time()
        to_remove = set()
        # check if the first in lru is timed out
        while True:
            # node_id, latest_time = next(iter(lru.items()))
            first = sdll.first_node()
            first_id = first.id
            first_tail = first.tail_time
            if first_tail < curr_time - TIMEWIN:
                comp = nx.node_connected_component(G, first_id)
                if len(comp) > 1:
                    merge_logger.debug("Merged {}".format(comp), extra=None)
                unmerged = [G.nodes[v]["data"] for v in list(comp)]
                merged = combine_merged_dict(unmerged)
                merged_queue.put(merged) 
                to_remove = to_remove.union(comp)
                # [lru.pop(v) for v in comp]
                [sdll.delete(v) for v in comp]
            else:
                break # no need to check lru further
        # clean up graph and lru
        G.remove_nodes_from(to_remove)
        
        
        
if __name__ == '__main__':

    
    import json
    import os
    from i24_database_api import DBClient
    from bson.objectid import ObjectId
    import multiprocessing as mp
    mp_manager = mp.Manager()
    
    with open("config/parameters.json") as f:
        parameters = json.load(f)
    with open(os.path.join(os.environ["USER_CONFIG_DIRECTORY"], "db_param.json")) as f:
        db_param = json.load(f)  
    parameters["merger_timeout"] = 0.01

    rec_collection = "funny_squirrel--RAW_GT2"
    raw_collection = "funny_squirrel--RAW_GT2" 
    
    dbc = DBClient(**db_param)
    raw = dbc.client["trajectories"][raw_collection]
    # rec = dbc.client["reconciled"][rec_collection]        
        
    fragment_queue = mp_manager.Queue() 
    merged_queue = mp_manager.Queue() 
    # funny_squirrel gt2
    # f_ids = [ObjectId('6320f56babd7d7253149373c'), ObjectId('6320f587abd7d725314937c7')] # low cost, but not merged
    # f_ids = [ObjectId('6320f576abd7d72531493774'), ObjectId('6320f579abd7d7253149378a')] #Y
    f_ids = [ObjectId('6320f578abd7d72531493781'), ObjectId('6320f592abd7d725314937fe'), ObjectId('6320f56babd7d7253149373c'), ObjectId('6320f587abd7d725314937c7')]# low cost, but not merged
    
    # merge solution
    # f_ids = [ObjectId('6320f575abd7d72531493772'), ObjectId('6320f56fabd7d7253149375a')] # Y
    # f_ids = [ObjectId('6320f587abd7d725314937cc'), ObjectId('6320f589abd7d725314937d2')] # Y
    # f_ids = [ObjectId('6320f5a7abd7d72531493857'), ObjectId('6320f5a2abd7d72531493838')] # y
    # f_ids = [ObjectId('6320f5acabd7d725314938df'), ObjectId('6320f5a7abd7d72531493852')] #maybe
    # f_ids = [ObjectId('6320f5acabd7d725314938ad'), ObjectId('6320f5aaabd7d72531493864')] # Y
    # f_ids = [ObjectId('6320f5acabd7d725314938a5'), ObjectId('6320f5acabd7d7253149386c')]
    trajs = []
    resampled = []
    
    for traj in raw.find({"_id": {"$in": f_ids}}).sort( "last_timestamp", 1 ):
        fragment_queue.put(traj)
        trajs.append(traj)
        # d = merge_resample(traj, parameters["conf_threshold"])
        # resampled.append(d)

    # nll = merge_cost(resampled[0], resampled[1])
    # print(nll)
    merge_fragments("west", fragment_queue, merged_queue, parameters)
    print("merged to ", merged_queue.qsize())
    # dummy_merge("west", fragment_queue, merged_queue, parameters)
    # from multi_opt import plot_track
    # plot_track(trajs)