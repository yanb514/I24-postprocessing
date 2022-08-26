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
from utils.utils_stitcher_cost import bhattacharyya_distance
from collections import deque
import queue

# from i24_database_api import DBClient
import i24_logger.log_writer as log_writer


def cost_3(track1, track2):
    '''
    TODO: modify so that it calculates only the time overlapped part
    resample in here
    use bhattacharyya_distance
    '''
    # no time overlap, return
    if track1["last_timestamp"] < track2["first_timestamp"]:
        return 1e6
    
    cost_offset = 0

    filter1 = np.array(track1["filter"], dtype=bool) # convert fomr [1,0] to [True, False]
    filter2 = np.array(track2["filter"], dtype=bool)
    
    t1 = np.array(track1["timestamp"])[filter1]
    t2 = np.array(track2["timestamp"])[filter2]
    
    x1 = np.array(track1["x_position"])[filter1]
    x2 = np.array(track2["x_position"])[filter2]
    
    y1 = np.array(track1["y_position"])[filter1]
    y2 = np.array(track2["y_position"])[filter2]

    
    # if time_gap > TIME_WIN, don't stitch
    gap = t2[0] - t1[-1] 
            
       
    if len(t1) >= len(t2):
        anchor = 1
        fitx, fity = track1["fitx"], track1["fity"]
        meast = t2
        measx = x2
        measy = y2
        pt = t1[-1]
        # if gap < -2: # if overlap in tiem for more than 2 sec, get all the overlaped range
        #     n = 0
        #     while meast[n] <= pt:
        #         n+= 1
        # else:
        n = min(len(meast), 30) # consider n measurements
        meast = meast[:n]
        measx = measx[:n]
        measy = measy[:n]
        dir = 1
        
        
    else:
        anchor = 2
        fitx, fity = track2["fitx"], track2["fity"]
        meast = t1
        measx = x1
        measy = y1
        pt = t2[0]
        # if gap < -2 or t1[0] > t2[0]: # if overlap in time is more than 2 sec, or t1 completely overlaps with t2, get all the overlaped range
        #     i = 0
        #     while meast[i] < pt:
        #         i += 1
        #     n = len(meast)-i
        # else:
        n = min(len(meast), 30) # consider n measurements
        meast = meast[-n:]
        measx = measx[-n:]
        measy = measy[-n:]
        dir = -1
        
        
    
    
    
    # find where to start the cone
    # 
    if anchor==2 and t1[0] > t2[0]: # t1 is completely overlap with t2
        pt = t1[-1]
        tdiff = meast * 0 # all zeros
    else:
        tdiff = (meast - pt) * dir


    # tdiff = meast - pt
    tdiff[tdiff<0] = 0 # cap non-negative

    
    
    slope, intercept = fitx
    targetx = slope * meast + intercept
    slope, intercept = fity
    targety = slope * meast + intercept
    
    sigmax = (0.05 + tdiff * 0.01) * fitx[0] #0.1,0.1, sigma in unit ft
    varx = sigmax**2
    # vary_pred = np.var(y1) if anchor == 1 else np.var(y2)
    sigmay = 1.5 + tdiff* 2 * fity[0]
    vary_pred = sigmay**2
    # vary_pred = max(vary_pred, 2) # lower bound
    vary_meas = np.var(measy)
    vary_meas = max(vary_meas, 2) # lower bound 
    
    
    
    bd = []
    for i, t in enumerate(tdiff):
        mu1 = np.array([targetx[i], targety[i]]) # predicted state
        mu2 = np.array([measx[i], measy[i]]) # measured state
        cov1 = np.diag([varx[i], vary_pred[i]]) # prediction variance - grows as tdiff
        cov2 = np.diag([varx[0], vary_meas])  # measurement variance - does not grow as tdiff
        # mu1 = np.array([targetx[i]]) # predicted state
        # mu2 = np.array([measx[i]]) # measured state
        # cov1 = np.diag([varx[i]]) # prediction variance - grows as tdiff
        # cov2 = np.diag([varx[0]])  # measurement variance - does not grow as tdiff
        bd.append(bhattacharyya_distance(mu1, mu2, cov1, cov2))
    
    nll = np.mean(bd)
    
    # print("id1: {}, id2: {}, cost:{:.2f}".format(str(track1['_id'])[-4:], str(track2['_id'])[-4:], nll))
    # print("")
    
    return nll + cost_offset

    
    




def merge_fragments(direction, fragment_queue, merged_queue, parameters):
    '''
    '''
    THRESH = parameters["merge_thresh"] # bhattar distance distance threshold
    TIMEWIN = parameters["time_win"]
    
    stitcher_logger = log_writer.logger
    stitcher_logger.set_name("merger_"+direction)
    
    G = nx.Graph() # merge graph, two nodes are connected if they can be merged
    deq = deque()
    cntr = 0
    
    # while parameters["stitched_collection"]=="":
    #     time.sleep(1)
    
    while True:
        try:
            fragment = fragment_queue.get(timeout = 1) # fragments are ordered in last_timestamp
        except queue.Empty:
            # TODO: final clean up
            comps = nx.connected_components(G)
            for comp in comps:
                stitcher_logger.debug("Merged {} fragments".format(len(comp)))
                merged_queue.put(list(comp))
            break
        
        cntr += 1
        curr_time = fragment["last_timestamp"]
        deq.append((fragment["_id"], curr_time))
        
        curr_nodes = G.nodes(data=True) # current nodes in G
        
        G.add_node(fragment["_id"], data=fragment)
        
        for node in curr_nodes:
            # if they have time overlaps
            if node["data"]["last_timestamp"] > fragment["first_timestamp"]: # TODO: check time overlap in the cost
                dist = cost_3(node["data"], fragment)
                if dist <= THRESH:
                    G.add_edge(node["_id"], fragment["_id"], weight = dist)
            
            
        # find connected components in G and check for timeout
        # TODO: use better data structure to make the following more efficient
        if cntr % 100 == 0:
            stitcher_logger.info("Graph nodes : {}, Graph edges: {}".format(G.number_of_nodes(), G.number_of_edges()),extra = None)
            
            to_remove = set()
            comps = nx.connected_components(G)
            for comp in comps:
                latest_time = max([G.nodes[v]["data"]["last_timestamp"] for v in list(comp)])
                if latest_time < curr_time - THRESH:
                    stitcher_logger.debug("Merged {} fragments".format(len(comp)))
                    merged_queue.put(list(comp))
                    to_remove = to_remove.union(comp)
                    

            G.remove_nodes_from(to_remove)
            
            
                    
        
                
        
        
    
    