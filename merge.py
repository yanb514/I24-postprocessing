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
import signal
import sys
from bson.objectid import ObjectId

# from i24_database_api import DBClient
from utils.misc import calc_fit_select, find_overlap_idx
import i24_logger.log_writer as log_writer
from i24_logger.log_writer import catch_critical
from utils.utils_stitcher_cost import bhattacharyya_distance
from utils.misc import SortedDLL
# import warnings
# warnings.filterwarnings('error')

class SIGINTException(SystemExit):
    pass

def soft_stop_hdlr(sig, action):
    '''
    Signal handling for SIGINT
    Soft terminate current process. Close ports and exit.
    '''
    raise SIGINTException # so to exit the while true loop

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
    time_series_field = ["timestamp", "x_position", "y_position", "length", "width", "height"]
    
    if "detection_confidence" in traj:
        conf = np.array(traj["detection_confidence"])
        nl = min(len(conf), len(traj["timestamp"]))
        
        # get confidence mask
        highconf_mask = np.array(conf >= conf_threshold)
        num_highconf = np.count_nonzero(highconf_mask)
        if num_highconf < 4:
            return None

        for key in time_series_field:
            traj[key] = np.array(traj[key][:nl])[highconf_mask[:nl]]
    
    else:
        for key in time_series_field:
            traj[key] = np.array(traj[key])
        
    # resample to df
    data = {key: traj[key] for key in time_series_field} 
    df = pd.DataFrame(data, columns=data.keys()) 
    index = pd.to_timedelta(df["timestamp"], unit='s')
    df = df.set_index(index)
    df = df.drop(columns = "timestamp")
    
    # resample to 100hz
    # to avoid the bias introduced by "floor" resample..., first upsample to 10ms, and then downsample to 25hz,
    # this method does not "snap" the timestamps to floor
    df=df.resample('10L').mean().interpolate(method="linear").resample('40L').asfreq()#.resample('0.04S')#.asfreq()#.interpolate(method="linear")
    # df=df.groupby(df.index.floor('0.04S')).mean().resample('0.04S').asfreq()


    df.index = df.index.values.astype('datetime64[ns]').astype('int64')*1e-9
    # df = df.interpolate(method='linear')
    
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


import warnings
warnings.filterwarnings('error')
        
# @catch_critical(errors = (Exception))
def merge_cost(track1, track2):
    '''
    track1 and 2 have to be resmplaed first
    only deals with two tracks that have time AND space(x) overlap (in the general case by separation of axis)
    speeded up bhartt cost
    '''
    t1 = track1["timestamp"]#[filter1]
    t2 = track2["timestamp"]#[filter2]
    x1,x2,y1,y2 = track1["x_position"], track2["x_position"], track1["y_position"], track2["y_position"]
    
    # Only proceed if both time and space have overlaps
    sx1, ex1 = min(x1[0], x1[-1]), max(x1[0], x1[-1])
    sx2, ex2 = min(x2[0], x2[-1]), max(x2[0], x2[-1])
    
    # Adjust based on length
    l1, l2 = np.nanmean(track1["length"]), np.nanmean(track2["length"])
    if track1["direction"] == 1:
        ex1, ex2 = ex1 + l1, ex2 + l2
    else:
        sx1, sx2 = sx1 - l1, sx2 - l2
    
    if t1[-1]<=t2[0] or t2[-1]<=t1[0] or sx1>ex2 or sx2>ex1: # if no time&space overlap, don't merge -> stitcher's job
        # print("no time space overlap")
        return 1e5

    s1, e1, s2, e2 = find_overlap_idx(t1, t2)
    
    # check if the overalpped position deviates too much
    # try:
    # if np.nanmean(np.abs(x1[s1:e1] - x2[s2:e2])) > 30 or np.nanmean(np.abs(y1[s1:e1] - y2[s2:e2])) > 6:
    #     return 1e6

        
    # tt2 = time.time()
    
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
    
    # tt3 = time.time()
    # print("id1: {}, id2: {}, cost:{:.4f}".format(str(track1['_id']), str(track2['_id']), nll))
    # print("")
    # print("other: {:.4f}, cost: {:.4f}, shape: {}".format(tt2-tt1, tt3-tt2, mu.shape))
    # print(nll)
    return nll


def merge_cost_simple_distance(track1, track2):
    """
    track1 and 2 have to be resmplaed first
    only deals with two tracks that have time AND space(x) overlap (in the general case by separation of axis)
    speeded up bhartt cost
    simple distance metric 1/(t_e^i-t_s^j) sum_t in[t_s^j, t_e^i]||pi(t)-pj(t)||_2
    """
    t1 = track1["timestamp"]#[filter1]
    t2 = track2["timestamp"]#[filter2]
    x1,x2,y1,y2 = track1["x_position"], track2["x_position"], track1["y_position"], track2["y_position"]
    
    # Only proceed if both time and space have overlaps
    sx1, ex1 = min(x1[0], x1[-1]), max(x1[0], x1[-1])
    sx2, ex2 = min(x2[0], x2[-1]), max(x2[0], x2[-1])
    
    # Adjust based on length
    l1, l2 = np.nanmean(track1["length"]), np.nanmean(track2["length"])
    if track1["direction"] == 1:
        ex1, ex2 = ex1 + l1, ex2 + l2
    else:
        sx1, sx2 = sx1 - l1, sx2 - l2
    
    if t1[-1]<=t2[0] or t2[-1]<=t1[0] or sx1>ex2 or sx2>ex1: # if no time&space overlap, don't merge -> stitcher's job
        # print("no time space overlap")
        return 1e5

    s1, e1, s2, e2 = find_overlap_idx(t1, t2)
    
    # try to vectorize
    pi = np.array([x1[s1:e1], y1[s1:e1]]) # 2xK
    pj = np.array([x2[s2:e2], y2[s2:e2]]) # 2xK
    diff = pi-pj
    distance = np.linalg.norm(diff)/(e1-s1)


    return distance



def combine_merged_dict(unmerged):
    '''
    unmerged: a list of fragment-dicts
    '''
    
    if len(unmerged) == 1:
        traj = unmerged[0]
        if "merged_ids" not in traj:
            traj["merged_ids"] = [traj["_id"]]
        
        return traj
    
    last_timestamps = [traj["last_timestamp"] for traj in unmerged]
    smallest_index = last_timestamps.index(min(last_timestamps))
    first_traj = unmerged[smallest_index]
    
    time_series_field = ["timestamp", "x_position", "y_position", "length", "width", "height"]
    dfs = []
    merged_ids = []
    for traj in unmerged:
        if "merged_ids" not in traj:
            merged_ids.append(traj["_id"])
        else:
            merged_ids.extend(traj["merged_ids"])
        data = {key: traj[key] for key in time_series_field} 
        df = pd.DataFrame(data, columns=data.keys()) 
        index = pd.to_timedelta(df["timestamp"], unit='s')
        df = df.set_index(index)
        df = df.drop(columns = "timestamp")
        dfs.append(df)

    df_merged = pd.concat(dfs).groupby(level=0, as_index=True, sort=True).mean()
    # merged = df_merged.to_dict('list')
    # overwrite first_traj with merged values
    for key in df_merged.columns:
        first_traj[key] = df_merged[key].values

    first_traj["merged_ids"] = merged_ids
    first_traj["timestamp"] = df_merged.index.values.astype('datetime64[ns]').astype('int64')*1e-9
    first_traj["first_timestamp"] = first_traj["timestamp"][0]
    first_traj["last_timestamp"] = first_traj["timestamp"][-1]
    first_traj["starting_x"] = first_traj["x_position"][0]
    first_traj["ending_x"] = first_traj["x_position"][-1]

    return first_traj



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
        if "merged_ids" not in traj:
            traj["merged_ids"] = [traj["_id"]]
        
        return traj
    
    dfs = []
    merged_ids = []
    for traj, df in unmerged:
        if "merged_ids" not in traj:
            merged_ids.append(traj["_id"])
        else:
            merged_ids.extend(traj["merged_ids"])
        dfs.append(df)

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
    
    return traj




def merge_fragments(direction, fragment_queue, merged_queue, parameters, name=None):
    '''
    graph structure
    if two nodes should be merged (by bhattar distance measure), then they are connected in graph
    node:
        node_id: fragment["_id"]
        raw: fragment dict
        data: resampled fragment df
    '''
    signal.signal(signal.SIGINT, soft_stop_hdlr)
    merge_logger = log_writer.logger
    if name:
        merge_logger.set_name(name)
    else:
        merge_logger.set_name("merger_"+direction)
        
    merge_logger.info("Process started")
    
    DIST_THRESH = parameters["merge_thresh"] # bhattar distance distance threshold
    TIMEWIN = parameters["time_win"]

    CONF_THRESH = parameters["conf_threshold"]
    TIMEOUT = parameters["merger_timeout"]
    
    G = nx.Graph() # merge graph, two nodes are connected if they can be merged
    sdll = SortedDLL() # a data structure to ensure trajectories are ordered in last_timestamp
    
    HB = parameters["log_heartbeat"]
    begin = time.time() # to time for log messages
    start = begin
    cntr, ct1, ct2, ct3, input_obj, output_obj, low_conf_cnt = 0,0,0,0,0,0,0
    
    while True:
        try:
            try:
                fragment = fragment_queue.get(timeout = TIMEOUT) # fragments are ordered in last_timestamp
                cntr += 1 # TODO: could over flow
            except queue.Empty:
                merge_logger.warning("merger timed out after {} sec.".format(TIMEOUT))
                comps = nx.connected_components(G)
                
                for comp in comps:
                    input_obj += len(comp)
                    output_obj += 1
                    unmerged = [ G.nodes[v]["data"] for v in list(comp)]
                    merged = combine_merged_dict(unmerged)
                    merged_queue.put(merged) 
                merge_logger.info("Final flushing {} raw fragments --> {} merged fragments".format(input_obj, output_obj),extra = None)
                break
            
            except SIGINTException:
                merge_logger.warning("SIGINT detected when trying to get from queue. Exit")
                break
            
            t1 = time.time()
            resampled = merge_resample(fragment, CONF_THRESH) # convert to df ! last_timestamp might be changed!!!
            t2 = time.time()
            ct1 += t2-t1
            
            if resampled is None:
                low_conf_cnt += 1 
                continue
            
            
            curr_time = resampled["last_timestamp"]
            curr_id = resampled["_id"]
        
            sdll.append({"id": curr_id, "tail_time": curr_time})
            
            curr_nodes = list(G.nodes(data=True)) # current nodes in G
            G.add_node(curr_id, data=resampled) 
            
                
            t1 = time.time()
            for node_id, node in curr_nodes:
                # if they have time overlaps
                dist = merge_cost(node["data"], resampled) # TODO: these two are not ordered in time,check time overlap within
                # merge_logger.info("{} and {}, cost={:.4f}".format(node_id, fragment["_id"], dist))
        
                if dist <= DIST_THRESH:
                    G.add_edge(node_id, curr_id, weight = dist)
                    sdll.update(key=curr_id, attr_val=curr_time)
                    sdll.update(key=node_id, attr_val=curr_time)
                    
                    # merge_logger.info("Merged {} and {}, cost={:.4f}".format(node_id, fragment["_id"], dist))
            
            t2 = time.time()
            ct2 += t2-t1
            
            t1 = time.time()
            to_remove = set()
            # check if the first in lru is timed out
            while True:
                # node_id, latest_time = next(iter(lru.items()))
                first = sdll.first_node()
                first_id = first.id
                first_tail = first.tail_time
                if first_tail < curr_time - TIMEWIN:                
                    comp = nx.node_connected_component(G, first_id)
                    # if aa in comp or bb in comp:
                    # merge_logger.info("Time out Merged {}, {:.2f}, {:.2f}, {:.2f}".format(comp, first_tail, curr_time, TIMEWIN), extra=None)
                    input_obj += len(comp)
                    output_obj += 1
                    unmerged = [G.nodes[v]["data"] for v in list(comp)]
                    merged = combine_merged_dict(unmerged)
                    merged_queue.put(merged) 
                    to_remove = to_remove.union(comp)
                    # [lru.pop(v) for v in comp]
                    [sdll.delete(v) for v in comp]

                else:
                    break # no need to check lru further
            # clean up graph and lru
            # if len(to_remove)>0:
            #     print(f"remove {len(to_remove)}")
            
            G.remove_nodes_from(to_remove)
            t2 = time.time()
            ct3 += t2-t1
            
            # heartbeat log
            now = time.time()
            if now - begin > HB:
                merge_logger.info("Graph nodes : {}, Graph edges: {}, cache: {}".format(G.number_of_nodes(), G.number_of_edges(), sdll.count()),extra = None)
                # merge_logger.info("Time elapsed for resample: {:.2f}, adding edge: {:.2f}, remove: {:.2f}, total run time: {:.2f}".format(ct1, ct2, ct3, now-start))
                merge_logger.info("{} raw fragments --> {} merged fragments, skipped {} low_conf.".format(input_obj, output_obj, low_conf_cnt),extra = None)
                begin = time.time()

        except SIGINTException:  # SIGINT detected
            merge_logger.info("SIGINT detected. Exit.")
            break
        
        except (ConnectionResetError, BrokenPipeError, EOFError) as e:   
            merge_logger.warning("Connection error: {}".format(e))
            break
        
        # added 6/14/2023
        except Exception as e: # other unknown exceptions are handled as error TODO UNTESTED CODE!
            merge_logger.error("Other error: {}, push all merged trajs to queue".format(e))
            
            comps = nx.connected_components(G)
            for comp in comps:
                input_obj += len(comp)
                output_obj += 1
                unmerged = [ G.nodes[v]["data"] for v in list(comp)]
                merged = combine_merged_dict(unmerged)
                merged_queue.put(merged) 
            merge_logger.info("Final flushing {} raw fragments --> {} merged fragments".format(input_obj, output_obj),extra = None)
            break
            
    # sys.exit()
    
        

        
        
if __name__ == '__main__':

    
    import json
    import os
    # from _evaluation.eval_stitcher import plot_traj, plot_stitched
    # from merge import merge_fragments
    from i24_database_api import DBClient
    from bson.objectid import ObjectId
    from itertools import combinations

    with open("config/parameters.json") as f:
        parameters = json.load(f)
    parameters["raw_trajectory_queue_get_timeout"] = 0.1
    with open(os.path.join(os.environ["USER_CONFIG_DIRECTORY"], "db_param.json")) as f:
        db_param = json.load(f)  
        
    raw_collection = "wednesday_2d"
    rec_collection = "wednesday_2d__2"
    
    dbc = DBClient(**db_param)
    raw = dbc.client["trajectories"][raw_collection]
    rec = dbc.client["reconciled"][rec_collection]
    temp = dbc.client["temp"][rec_collection]
    
    # RES_THRESH_X = parameters["stitcher_args"]["residual_threshold_x"]
    # RES_THRESH_Y = parameters["residual_threshold_y"]
    # TIME_WIN = parameters["time_win"]
    # CONF_THRESH = parameters["conf_threshold"]
    parameters["merger_timeout"] = 0.1
    
    f_ids = [ObjectId("644c6f95c2c8fad0b2cf4232"),
    # ObjectId("644c64cbae8899e271b5f9eb"),
    ObjectId("644c6f58c2c8fad0b2cf4192")]
    
    fragment_queue = queue.Queue()
    merged_queue = queue.Queue()
    fragments = [fragment_queue.put(traj) for traj in temp.find({"_id": {"$in": f_ids}}).sort( "last_timestamp", 1 )]
    
        
    # for fgmt1, fgmt2 in combinations(fragments, 2):
    #     fgmt1 = merge_resample(fgmt1, CONF_THRESH)
    #     fgmt2 = merge_resample(fgmt2, CONF_THRESH)
    #     cost = merge_cost( fgmt1, fgmt2)
    #     print(fgmt1["_id"], fgmt2["_id"], cost)
    

    
    merge_fragments("west", fragment_queue, merged_queue, parameters)
    print("merged to ", merged_queue.qsize())
    # dummy_merge("west", fragment_queue, merged_queue, parameters)
    # from multi_opt import plot_track
    # plot_track(trajs)
    
    