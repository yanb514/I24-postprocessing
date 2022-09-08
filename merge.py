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

# from i24_database_api import DBClient
import i24_logger.log_writer as log_writer
from i24_logger.log_writer import catch_critical
from utils.utils_stitcher_cost import bhattacharyya_distance
from utils.misc import SortedDLL
# import warnings
# warnings.filterwarnings('error')


@catch_critical(errors = (RuntimeWarning))
def merge_preprocess(traj, conf_threshold):
    '''
    traj: dict
    preprocess procedures
    1. filter based on conf
    2. resample to df
    return df
TODO: DEAL WITH NAN after conf mask
    '''
    conf = np.array(traj["detection_confidence"])
    
    # get confidence mask
    highconf_mask = np.array(conf >= conf_threshold)
    num_highconf = np.count_nonzero(highconf_mask)
    if num_highconf < 4:
        return None
       
    time_series_field = ["timestamp", "x_position", "y_position", "length", "width", "height"]
    for key in time_series_field:
        traj[key] = list(np.array(traj[key])[highconf_mask])
        
    # resample to df
    data = {key: traj[key] for key in time_series_field} 
    df = pd.DataFrame(data, columns=data.keys()) 
    index = pd.to_timedelta(df["timestamp"], unit='s')
    df = df.set_index(index)
    df = df.drop(columns = "timestamp")
    # resample to 5hz
    df=df.groupby(df.index.floor('0.04S')).mean().resample('0.04S').asfreq()
    df.index = df.index.values.astype('datetime64[ns]').astype('int64')*1e-9
    df = df.interpolate(method='linear')
        
    return df
        
    
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
                unmerged = [(G.nodes[v]["raw"], G.nodes[v]["data"]) for v in list(comp)]
                merged = combine_merged(unmerged)
                merged_queue.put(merged) 
            break
        
        df = merge_preprocess(fragment, CONF_THRESH) # convert to df ! last_timestamp might be changed!!!
            
        if df is None:
            merge_logger.info("skip {} in merging: LOW CONF".format(fragment["_id"]))
            continue
        
        
        curr_time = fragment["last_timestamp"]
        
        # lru[fragment["_id"]] = curr_time # TODO DF LAST
        sdll.append({"id": fragment["_id"], "tail_time": df.index[-1]})
        
        curr_nodes = list(G.nodes(data=True)) # current nodes in G
        G.add_node(fragment["_id"], data=df, raw=fragment) #TODO: determine if raw fragment dict should be saved
        
        t1 = time.time()
        for node_id, node in curr_nodes:
            # if they have time overlaps
            dist = overlap_cost(node["data"], df) # TODO: these two are not ordered in time,check time overlap within
            if dist <= DIST_THRESH:
                G.add_edge(node_id, fragment["_id"], weight = dist)
                # lru[node_id] = curr_time # TODO: reset to the larger last-tiemstamp of the two
                # lru.move_to_end(node_id, last=True) # move node_id to last of lru
                larger = max([G.nodes[node_id]["data"].index[-1], G.nodes[fragment["_id"]]["data"].index[-1]])
                sdll.update(key=fragment["_id"], attr_val=larger)
                sdll.update(key=node_id, attr_val=larger)
                
                # merge_logger.info("Merged {} and {}".format(node_id, fragment["_id"]))
        
        t2 = time.time()
        
        # find connected components in G and check for timeout
        # TODO: use better data structure to make the following more efficient
        if cntr % 10 == 0:
            merge_logger.info("Graph nodes : {}, Graph edges: {}, cache: {}".format(G.number_of_nodes(), G.number_of_edges(), sdll.count()),extra = None)
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
                unmerged = [(G.nodes[v]["raw"], G.nodes[v]["data"]) for v in list(comp)]
                merged = combine_merged(unmerged)
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
    parameters["raw_trajectory_queue_get_timeout"] = 0.1

    raw_collection = "organic_forengi--RAW_GT2" # collection name is the same in both databases
    # rec_collection = "morose_caribou--RAW_GT1__escalates"
    
    dbc = DBClient(**db_param)
    raw = dbc.client["trajectories"][raw_collection]
    # rec = dbc.client["reconciled"][rec_collection]        
        
    fragment_queue = mp_manager.Queue() 
    merged_queue = mp_manager.Queue() 
    
    f_ids = [ObjectId('6307d66ff0b73edfc22af090'), ObjectId('6307d666f0b73edfc22af071')]
    
    for traj in raw.find({"_id": {"$in": f_ids}}):
        fragment_queue.put(traj)
        
    merge_fragments("west", fragment_queue, merged_queue, parameters)
    print("merged to ", merged_queue.qsize())
    