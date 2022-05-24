#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 15:17:59 2022

@author: yanbing_wang
"""
import queue
from collections import deque 
from i24_database_api.db_reader import DBReader
from i24_database_api.db_writer import DBWriter
import os
from i24_configparse.parse import parse_cfg
from utils.data_structures import Fragment, PathCache, MOT_Graph
import time
from i24_logger.log_writer import logger 


# TODO:
    # 1. check if the answers agree with nx.edmond_karp
    # 2. add more intelligent enter/exiting cost based on the direction and where they are relative to the road
    # 3. collapse path into human-readable ones, e.g., [1001, 1002, etc]
    
    
def read_to_queue(parameters):
    '''
    construct MOT graph from fragment list based on the specified loss function
    '''
    # connect to database
    raw = DBReader(host=parameters.default_host, port=parameters.default_port, username=parameters.readonly_user,   
                   password=parameters.default_password,
                   database_name=parameters.db_name, collection_name=parameters.raw_collection)
    print("connected to raw collection")
    gt = DBReader(host=parameters.default_host, port=parameters.default_port, username=parameters.readonly_user,   
                    password=parameters.default_password,
                    database_name=parameters.db_name, collection_name=parameters.gt_collection)
    print("connected to gt collection")
    stitched = DBWriter(host=parameters.default_host, port=parameters.default_port, 
            username=parameters.default_username, password=parameters.default_password,
            database_name=parameters.db_name, collection_name=parameters.stitched_collection,
            server_id=1, process_name=1, process_id=1, session_config_id=1, schema_file=None)
    stitched.collection.drop()
    
    # stitched_reader = DBReader(host=parameters.default_host, port=parameters.default_port, 
    #                            username=parameters.readonly_user, password=parameters.default_password,
    #                            database_name=parameters.db_name, collection_name=parameters.stitched_collection)
    print("connected to stitched collection")
    
    # specify ground truth ids and the corresponding fragment ids
    gt_ids = [i for i in range(150,220)]
    fragment_ids = []
    gt_res = gt.read_query(query_filter = {"ID": {"$in": gt_ids}},
                            limit = 0)
    
    for gt_doc in gt_res:
        fragment_ids.extend(gt_doc["fragment_ids"])
    
    # raw_res = raw.read_query(query_filter = {"_id": {"$in": fragment_ids}},
    #                           query_sort = [("last_timestamp", "ASC")])
    raw_res = raw.read_query(query_filter = {"$and":[ {"last_timestamp": {"$gt": 300}}, 
                                                      {"last_timestamp": {"$lt": 500}},
                                                      {"_id": {"$in": fragment_ids}}]},
                                query_sort = [("last_timestamp", "ASC")])
    
    # write fragments to queue
    fragment_queue = queue.Queue()
    # fragment_set = set()
    for doc in raw_res:
        fragment_queue.put(doc)
        # fragment_set.add(doc["_id"])
        
    fragment_size = fragment_queue.qsize()
    print("Queue size: ", fragment_size)

    return fragment_queue
    

            


def min_cost_flow_online(fragment_queue, stitched_trajectory_queue, parameters):
    '''
    iteratively solving a MCF problem on a smaller graph
    '''
    TIME_WIN = parameters.time_win
    
    # Get parameters
    TIME_WIN = parameters.time_win
    IDLE_TIME = parameters.idle_time
    TIMEOUT = parameters.raw_trajectory_queue_get_timeout
    ATTR_NAME = parameters.fragment_attr_name
    
    # Initialize some data structures
    curr_fragments = deque()  # fragments that are in current window (left, right), sorted by last_timestamp
    # past_fragments = dict()  # set of ids indicate end of fragment ready to be matched, insertion ordered
    P = PathCache(attr_name = ATTR_NAME) # an LRU cache of Fragment object (see utils.data_structures)
    m = MOT_Graph(ATTR_NAME, parameters)
    
    # Make a database connection for writing
    dbw = DBWriter(host=parameters.default_host, port=parameters.default_port, 
               username=parameters.default_username, password=parameters.default_password,
               database_name=parameters.db_name, collection_name = parameters.stitched_collection, 
               server_id=1, process_name=1, process_id=1, session_config_id=1, max_idle_time_ms = None,
               schema_file=parameters.stitched_schema_path)
    
    count = 0
    # fgmt_count = 0
    cache_size = 0
    cache_arr = []
    t0 = time.time()
    time_arr = []
    nodes_arr = []

    while True:
        
        try: # grab a fragment from the queue if queue is not empty
            fragment = Fragment(fragment_queue.get(block = True, timeout = TIMEOUT)) # make object
            P.add_node(fragment)
            
            # specify time window for curr_fragments
            curr_fragments.append(getattr(fragment, ATTR_NAME)) 
            left = fragment.first_timestamp - TIME_WIN   
            
            # compute fragment statistics (1d motion model)
            fragment.compute_stats()
            # fgmt_count += 1
            cache_size = len(P.path)
            cache_arr.append(cache_size)
            t1 = time.time()
            time_arr.append(t1-t0)
            nodes_arr.append(len(m.G.nodes()))
            
            
            
        except: # if queue is empty, process the remaining
            if not curr_fragments: # if all remaining fragments are processed
                left += IDLE_TIME + 1e-6 # to make all fragments "ready" for tail matching
                if len(P.path) == 0:# exit the stitcher if all fragments are written to database
                    break
            else: 
                # specify time window for curr_fragments
                left = P.get_fragment(curr_fragments[0]).last_timestamp + 1e-6
            fragment = None
            
        
            
            
        # Add fragments to G if buffer time is reached
        while curr_fragments and P.get_fragment(curr_fragments[0]).last_timestamp < left: 
            
            past_id = curr_fragments.popleft()
            past_fragment = P.get_fragment(past_id)
            m.add_node(past_fragment, P.path) # update G with the new fragment
            count += 1
 
        
        # Start min-cost-flow
        if count >= 20 or fragment is None:
            
            count = 0
            m.min_cost_flow("s", "t")
            

            # Collapse paths
            m.find_all_post_paths(m.G, "t", "s") 
            
            for path, cost in m.all_paths:
                path = m.pretty_path(path)
                head = path[0]
                tail = path[-1]
                
                # Update P by unioning nodes along the path
                # forward union
                # for i in range(len(path)-1):
                #     try: # if attr = "ID"
                #         id1, id2 = float(path[i].partition("-")[0]), float(path[i+1].partition("-")[0])
                #     except: # if attr = "id"
                #         id1, id2 = path[i].partition("-")[0], path[i+1].partition("-")[0]
                #     if id1 in P.path and id2 in P.path and id1 != id2:
                #         # print("union: {} and {}".format(id1, id2))
                #         P.union(id1, id2)
                # backward union
                
                for i in range(1, len(path)):
                    id1, id2 = path[-i], path[-i-1]
                    
                    if id1 in P.path and id2 in P.path:
                        # print("union: {} and {}".format(id1, id2))
                        P.union(id2, id1)
                
                        
                # Delete nodes from G, only keep heads' pre and tail's post 
                for node_id in path:
                    try: m.G.remove_node(node_id +"-pre")
                    except: pass
                    try: m.G.remove_node(node_id +"-post")
                    except: pass
                
                
                # Flip the edge directions to s-t, aggregate edge cost
                m.G.add_edge("s", head + "-pre", weight = 0, flipped = False)
                m.G.add_edge(tail+"-post", "t", weight = 0, flipped = False)
                m.G.add_edge(head+ "-pre", tail+"-post", weight = -cost, flipped = False)   
                # print("path: ", path)
                # m.fragment_dict[tail] = P.get_fragment(tail)
                
                
            
            
        # Retrive paths, write to DB
        while True:
            
            try:  
                root = P.first_node()
                if not root:
                    break
                # print(root.ID, root.tail_time, left)
                
                if root.tail_time < left - IDLE_TIME:
                    # print("root's tail time: {:.2f}, current time window: {:.2f}-{:.2f}".format(root.tail_time, left, right))
                    path = P.pop_first_path() 
                    # print(path)
                    print("** write to db: root {}, last_modified {:.2f}, path length: {}".format(root.ID, root.tail_time,len(path)))
                    stitched_trajectory_queue.put(path, timeout = parameters.stitched_trajectory_queue_put_timeout)
                    dbw.write_one_trajectory(thread = True, fragment_ids = path)
                else: # break if first in cache is not timed out yet
                    break
            except StopIteration: # break if nothing in cache
                break
                
    return time_arr, cache_arr, nodes_arr
        
        
    
if __name__ == '__main__':
    
    # get parameters
    cwd = os.getcwd()
    cfg = "./config"
    config_path = os.path.join(cwd,cfg)
    os.environ["user_config_directory"] = config_path
    parameters = parse_cfg("DEBUG", cfg_name = "test_param.config")
    
    fragment_queue = read_to_queue(parameters)
    s1 = fragment_queue.qsize()
    stitched_trajectory_queue = queue.Queue()
    t1 = time.time()
    time_arr, cache_arr, nodes_arr = min_cost_flow_online(fragment_queue, stitched_trajectory_queue, parameters)
    s2 = stitched_trajectory_queue.qsize()
    t2 = time.time()
    print("{} fragment stitched to {} trajectories, taking {:.2f} sec".format(s1, s2, t2-t1))
    
    # plot runtime
    # fgmt = [154, 292, 431, 585]
    # time = [1.05, 1.62, 2.77, 4.63]
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure()
    plt.scatter(np.arange(s1), time_arr, label="run time (sec)")
    plt.xlabel("# fragments")
    plt.ylabel("cumulative run time (sec)")
    
    plt.figure()
    plt.scatter(np.arange(s1), cache_arr, label = "cache size")
    plt.xlabel("# fragments")
    plt.ylabel("memory size (# fragments)")
    
    plt.figure()
    plt.scatter(np.arange(s1), nodes_arr, label = "cache size")
    plt.xlabel("# fragments")
    plt.ylabel("# nodes in G")
    
    
    