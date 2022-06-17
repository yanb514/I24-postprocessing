#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 15:17:59 2022

@author: yanbing_wang
"""
import queue
from collections import deque 
import os

from i24_database_api.db_reader import DBReader
from i24_database_api.db_writer import DBWriter
import i24_logger.log_writer as log_writer
from i24_configparse import parse_cfg
from utils.data_structures import Fragment, PathCache, MOT_Graph, MOTGraphSingle, SortedDLL
import time
import heapq
from collections import defaultdict
import sys
sys.path.append('../')


# config_path = os.path.join(os.getcwd(),"config")
# os.environ["user_config_directory"] = config_path
# parameters = parse_cfg("DEBUG", cfg_name = "test_param.config")


# TODO:
    # 1. check if the answers agree with nx.edmond_karp
    # 2. add more intelligent enter/exiting cost based on the direction and where they are relative to the road
    
    
# @catch_critical(errors = (Exception))  
def read_to_queue(gt_ids, gt_val, lt_val, parameters):
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
    gt_ids = gt_ids
    fragment_ids = []
    gt_res = gt.read_query(query_filter = {"ID": {"$in": gt_ids}},
                            limit = 0)
    actual_gt_ids = []
    for gt_doc in gt_res:
        try:
            fragment_ids.extend(gt_doc["fragment_ids"])
            actual_gt_ids.append(gt_doc["ID"])
        except:
            pass
    
    # raw_res = raw.read_query(query_filter = {"_id": {"$in": fragment_ids}},
    #                           query_sort = [("last_timestamp", "ASC")])
    raw_res = raw.read_query(query_filter = {"$and":[ {"last_timestamp": {"$gt": gt_val}}, 
                                                      {"last_timestamp": {"$lt": lt_val}},
                                                        {"_id": {"$in": fragment_ids}}
                                                      ]},
                                query_sort = [("last_timestamp", "ASC")])
    
    # write fragments to queue
    fragment_queue = queue.Queue()
    for doc in raw_res:
        fragment_queue.put(doc)
        
    fragment_size = fragment_queue.qsize()
    print("Queue size: ", fragment_size)

    return fragment_queue, actual_gt_ids,set(fragment_ids)  



def min_cost_flow_batch(direction, fragment_queue, stitched_trajectory_queue, parameters):
    '''
    solve min cost flow problem on a given graph using successive shortest path 
    - derived from Edmonds-Karp for max flow
    '''
    # Initiate a logger
    stitcher_logger = log_writer.logger
    stitcher_logger.set_name("stitcher_"+direction)
    stitcher_logger.info("** min_cost_flow_batch starts. fragment_queue size: {}".format(fragment_queue.qsize()),extra = None)

    # Make a database connection for writing
    schema_path = os.path.join(os.environ["user_config_directory"],parameters.stitched_schema_path)
    dbw = DBWriter(host=parameters.default_host, port=parameters.default_port, 
                username=parameters.default_username, password=parameters.default_password,
                database_name=parameters.db_name, collection_name = parameters.stitched_collection, 
                server_id=1, process_name=1, process_id=1, session_config_id=1, max_idle_time_ms = None,
                schema_file=schema_path)
    
    # Get parameters
    ATTR_NAME = parameters.fragment_attr_name
    
    # Initialize some data structures
    m = MOT_Graph(ATTR_NAME, parameters)
    m.construct_graph_from_fragments(fragment_queue)
    # all_paths = []
    
    
            
    # count = 0
    m.min_cost_flow("s", "t")
    
    # Collapse paths
    m.find_all_post_paths(m.G, "t", "s") 
    
    for path, cost in m.all_paths:
        path = m.pretty_path(path)
        # all_paths.append(path)
        # print(path)
        # print("** write to db: root {}, path length: {}".format(path[0], len(path)))
        stitched_trajectory_queue.put(path, timeout = parameters.stitched_trajectory_queue_put_timeout)
        # dbw.write_one_trajectory(thread = True, fragment_ids = path)
        
    # return all_paths
    return
    
        
    
    
    
    
    
                


def min_cost_flow_online(direction, fragment_queue, stitched_trajectory_queue, parameters):
    '''
    this version is bad. don't use
    iteratively solving a MCF problem on a smaller graph
    '''
    # Initiate a logger
    stitcher_logger = log_writer.logger
    stitcher_logger.set_name("stitcher_"+direction)
    stitcher_logger.info("** min_cost_flow_online starts. fragment_queue size: {}".format(fragment_queue.qsize()),extra = None)

    # Make a database connection for writing
    schema_path = os.path.join(os.environ["user_config_directory"],parameters.stitched_schema_path)
    dbw = DBWriter(host=parameters.default_host, port=parameters.default_port, 
                username=parameters.default_username, password=parameters.default_password,
                database_name=parameters.db_name, collection_name = parameters.stitched_collection, 
                server_id=1, process_name=1, process_id=1, session_config_id=1, max_idle_time_ms = None,
                schema_file=schema_path)
    
    
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
    
    # book keeping stuff
    # count = 0
    # fgmt_count = 0
    cache_size = 0
    cache_arr = []
    t0 = time.time()
    time_arr = []
    nodes_arr = []
    all_paths = []
    
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
            # t1 = time.time()
            # time_arr.append(t1-t0)
            # nodes_arr.append(len(m.G.nodes()))
            
            
            
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
            # count += 1
 
        
        # Start min-cost-flow
        # if count >= 20 or fragment is None:
            
        # count = 0
        m.min_cost_flow("s", "t")
        
        # Collapse paths
        m.find_all_post_paths(m.G, "t", "s") 
        
        for path, cost in m.all_paths:
            path = m.pretty_path(path)
            head = path[0]
            tail = path[-1]
            
            
            for i in range(0, len(path)-1):
                # id1, id2 = path[-i], path[-i-1]
                id1, id2 = path[i], path[i+1]
                if id1 in P.path and id2 in P.path:
                    # print("union: {} and {}".format(id1, id2))
                    P.union(id1, id2)
            
                    
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
                    path = P.pop_first_path() # remove nodes from P
                    # all_paths.append(path)
                    # remove nodes from G
                    for node_id in path:
                        try: m.G.remove_node(node_id +"-pre")
                        except: pass
                        try: m.G.remove_node(node_id +"-post")
                        except: pass
                    # print(path)
                    # print("** write to db: root {}, last_modified {:.2f}, path length: {}".format(root.ID, root.tail_time,len(path)))
                    stitched_trajectory_queue.put(path, timeout = parameters.stitched_trajectory_queue_put_timeout)
                    # dbw.write_one_trajectory(thread = True, fragment_ids = path)
                else: # break if first in cache is not timed out yet
                    break
            except StopIteration: # break if nothing in cache
                break
                
    # return time_arr, cache_arr, nodes_arr
    # return all_paths
    return




# @catch_critical(errors = (Exception))
def min_cost_flow_online_neg_cycle(direction, fragment_queue, stitched_trajectory_queue, parameters):
    '''
    run MCF on the entire (growing) graph as a new fragment is added in
    this online version is an approximation of batch MCF
    '''
    
    # Initiate a logger
    stitcher_logger = log_writer.logger
    stitcher_logger.set_name("stitcher_"+direction)
    stitcher_logger.info("** min_cost_flow_online_neg_cycle starts. fragment_queue size: {}".format(fragment_queue.qsize()),extra = None)

    # Make a database connection for writing
    schema_path = os.path.join(os.environ["user_config_directory"],parameters.stitched_schema_path)
    dbw = DBWriter(host=parameters.default_host, port=parameters.default_port, 
               username=parameters.default_username, password=parameters.default_password,
               database_name=parameters.db_name, collection_name = parameters.stitched_collection, 
               server_id=1, process_name=1, process_id=1, session_config_id=1, max_idle_time_ms = None,
               schema_file=schema_path)
        
    
    
    ATTR_NAME = parameters.fragment_attr_name
    m = MOT_Graph(ATTR_NAME, parameters)
    cache = {}
    # curr_fragments = deque()
    IDLE_TIME = parameters.idle_time
    
    # cum_time = []
    # t0 = time.time()
    # cum_mem = []
    
    while True:
        
        # stitcher_logger.info("fragment qsize: {}".format(fragment_queue.qsize()))
        
        # t1 = time.time()
        # cum_time.append(t1-t0)
        # cum_mem.append(len(cache))
        
        try:
            fgmt = Fragment(fragment_queue.get(timeout = 2))
            fgmt.compute_stats()
            m.add_node(fgmt, cache)
            fgmt_id = getattr(fgmt, ATTR_NAME)
            cache[fgmt_id] = fgmt
            left = fgmt.first_timestamp
            # stitcher_logger.info("fgmt_id: {}, first timestamp: {:.2f}".format(fgmt_id, left))
            
        except:
            # process the remaining in G
            m.find_all_post_paths(m.G, "t", "s")
            all_paths = m.all_paths
            for path,_ in all_paths:
                path = m.pretty_path(path)
                stitched_trajectory_queue.put(path)
                for p in path: # should not have repeats in nodes_to_remove
                    m.G.remove_node(p+"-pre")
                    m.G.remove_node(p+"-post")
                    cache.pop(p)
                stitcher_logger.info("** no new fgmt, write to queue. path length: {}, head id: {}, graph size: {}".format(len(path), path[0], len(cache)))
                # print("final flush head tail: ", path[0], path[-1])
            break
        
        # Finding all pivots (predecessor nodes of new_fgmt-pre such that the path cost pivot-new_fgmt-t is negative)
        pivot_heap = []
        for pred, data in m.G.pred[fgmt_id + "-pre"].items():
            cost_new_path = data["weight"] + parameters.inclusion
            if cost_new_path < 0: # favorable to attach fgmt after pred
                heapq.heappush(pivot_heap, (cost_new_path, pred))
           

        # check the cost of old path from t->pivot along "flipped" edges
        while pivot_heap:
            cost_new_path, pred = heapq.heappop(pivot_heap) 
            if pred == "s": # create a new trajectory
                m.flip_edge_along_path([pred, fgmt_id+"-pre", fgmt_id+"-post", "t"], flipped = True)
                break

            m.find_all_post_paths(m.G, "t", pred)
            old_path, cost_old_path = m.all_paths[0] # should have one path only
            
            # the new path that starts from pivot and includes the new fragment is better than the path starting from pivot and not including the new fgmt
            # therefore, update the new path
            
            if cost_new_path < -cost_old_path:
                # flip edges in path from pivot -> fgmt -> t
                m.flip_edge_along_path([pred, fgmt_id+"-pre", fgmt_id+"-post", "t"], flipped = True)     
                
                succ = old_path[1]
                m.flip_edge_along_path([succ, pred], flipped = False)# back to the original state
                
                # if sucessor of pivot in old path is not t, connect sucessor to s
                if succ != "t":
                    m.flip_edge_along_path(["s", succ], flipped = True)
                    
                break # no need to check further in heap
        
        
        # look at all neighbors of "t", which are the tail nodes
        # if the tail is time-out, then pop the entire path
        m.find_all_post_paths(m.G, "t", "s")
        # print(m.all_paths)
        
        nodes_to_remove = []
        for node in m.G.adj["t"]: # for each tail node
            if m.G["t"][node]["flipped"]:
                tail_id = node.partition("-")[0]
                if cache[tail_id].last_timestamp + IDLE_TIME < left:
                    m.find_all_post_paths(m.G, node, "s")
                    path, cost = m.all_paths[0]
                    path = m.pretty_path(path)
                    # print("remove: ", path)
                    stitcher_logger.info("** write to queue. path length: {}, head id: {}, graph size: {}".format(len(path), path[0], len(cache)))
                    stitched_trajectory_queue.put(path)
                    # dbw.write_one_trajectory(thread = True, fragment_ids = path)
                    # remove path from G and cache
                    nodes_to_remove.extend(path)

                        
        for p in nodes_to_remove: # should not have repeats in nodes_to_remove
            m.G.remove_node(p+"-pre")
            m.G.remove_node(p+"-post")
            cache.pop(p)
            
        # stitcher_logger.info("nodes to remove: {}, cache size: {}".format(len(nodes_to_remove), len(cache)))
                
        
    # return cum_time, cum_mem
    return







def min_cost_flow_online_slow(direction, fragment_queue, stitched_trajectory_queue, parameters):
    '''
    run MCF on the entire (growing) graph as a new fragment is added in
    print out the paths, just to see how paths change
    how does the new fgmt modify the previous MCF solution?
        1. the new fgmt creates a new trajectory (Y)
        2. the new fgmt addes to the tail of an existing trajectory (Y)
        3. the new fgmt breaks an existing trajectory (in theory should exist, but not observed)
        4 ...?
    '''
    # Initiate a logger
    stitcher_logger = log_writer.logger
    stitcher_logger.set_name("stitcher_"+direction)

    # Make a database connection for writing
    schema_path = os.path.join(os.environ["user_config_directory"],parameters.stitched_schema_path)
    dbw = DBWriter(host=parameters.default_host, port=parameters.default_port, 
               username=parameters.default_username, password=parameters.default_password,
               database_name=parameters.db_name, collection_name = parameters.stitched_collection, 
               server_id=1, process_name=1, process_id=1, session_config_id=1, max_idle_time_ms = None,
               schema_file=schema_path)
        
    stitcher_logger.info("** min_cost_flow_online_neg_cycle starts. fragment_queue size: {}".format(fragment_queue.qsize()),extra = None)
    
    ATTR_NAME = parameters.fragment_attr_name
    m = MOT_Graph(ATTR_NAME, parameters)
    cache = {}
    
    while True:
        try:
            fgmt = Fragment(fragment_queue.get(timeout=2))
        except:
            all_paths = m.all_paths
            for path,_ in all_paths:
                path = m.pretty_path(path)
                stitched_trajectory_queue.put(path)
            break
        fgmt.compute_stats()
        m.add_node(fgmt, cache)
        fgmt_id = getattr(fgmt, ATTR_NAME)
        cache[fgmt_id] = fgmt
        print("new fgmt: ", fgmt_id)
        
        # run MCF
        m.min_cost_flow("s", "t")
        # m.find_all_post_paths(m.G, "t", "s") 
        # print(m.all_paths)
        
        # flip edges back
        edge_list = list(m.G.edges)
        for u,v in edge_list:
            if m.G.edges[u,v]["flipped"]:
                cost = m.G.edges[u,v]["weight"]
                m.G.remove_edge(u,v)
                m.G.add_edge(v,u, weight = -cost, flipped = False)
                
                
        # check if the tail of any paths is timed out. if so remove the entire path from the graph
        # TODO
    return

    

def min_cost_flow_online_alt_path(direction, fragment_queue, stitched_trajectory_queue, parameters):
    '''
    incrementally fixing the matching
    '''
    # Initiate a logger
    stitcher_logger = log_writer.logger
    stitcher_logger.set_name("stitcher_"+direction)

    # Make a database connection for writing
    schema_path = os.path.join(os.environ["user_config_directory"],parameters.stitched_schema_path)
    dbw = DBWriter(host=parameters.default_host, port=parameters.default_port, 
               username=parameters.default_username, password=parameters.default_password,
               database_name=parameters.db_name, collection_name = parameters.stitched_collection, 
               server_id=1, process_name=1, process_id=1, session_config_id=1, max_idle_time_ms = None,
               schema_file=schema_path)
        
    stitcher_logger.info("** min_cost_flow_online_alt_path starts. fragment_queue size: {}".format(fragment_queue.qsize()),extra = None)
    
    ATTR_NAME = parameters.fragment_attr_name
    TIME_WIN = parameters.time_win
    
    m = MOTGraphSingle(ATTR_NAME, parameters)
    
    counter = 0 # for log
    
    while True:
        try:
            fgmt = Fragment(fragment_queue.get(timeout = 3))
        except:
            all_paths = m.get_all_traj()
            for path in all_paths:
                stitched_trajectory_queue.put(path[::-1])
            break
        
        fgmt.compute_stats()
        m.add_node(fgmt)
        
        # run MCF
        fgmt_id = getattr(fgmt, ATTR_NAME)
        m.augment_path(fgmt_id)

        # pop path if a path is ready
        all_paths = m.pop_path(time_thresh = fgmt.first_timestamp - TIME_WIN)
        for path in all_paths:
            stitched_trajectory_queue.put(path[::-1])
            m.clean_graph(path)
            stitcher_logger.info("** stitched {} fragments into one trajectory".format(len(path)),extra = None)
         
        
        if counter % 100 == 0:
            stitcher_logger.info("graph size: {}, deque size: {}".format(len(m.G), len(m.in_graph_deque)),extra = None)
            counter = 0
        counter += 1
    return   




def test_fragments(gt_ids, paths):
    '''
    Count the number of fragments (under-stitch) from the output of the stitcher
    '''   
    base = 100000     
    gt_id_st_fgm_ids = defaultdict(set) # key: (int) gt_id, val: (set) corresponding stitcher fragment_ids
    IDS = 0

    for i,path in enumerate(paths):
        corr_gt_ids = set()
        for node in path:
            node = float(node)
            corr_gt_ids.add(node//base)
            gt_id_st_fgm_ids[node//base].add(i)
            
        if len(corr_gt_ids) > 1:
            print("ID switches: ", corr_gt_ids)
            IDS += len(corr_gt_ids) - 1
    
    FGMT = 0
    for key,val in gt_id_st_fgm_ids.items():
        if len(val) > 1:
            print("fragments: ", [paths[i] for i in val])
            FGMT += len(val)-1
                

    return FGMT, IDS
    
if __name__ == '__main__':
    
    # get parameters
    cwd = os.getcwd()
    cfg = "config"
    config_path = os.path.join(cwd,cfg)
    os.environ["user_config_directory"] = config_path
    os.environ["my_config_section"] = "DEBUG"
    parameters = parse_cfg("my_config_section", cfg_name = "test_param.config")
    
    # read to queue
    # gt_ids = [i for i in range(150, 180)]
    gt_ids = [150]
    gt_val = 30
    lt_val = 100
    
    fragment_queue,actual_gt_ids,_ = read_to_queue(gt_ids=gt_ids, gt_val=gt_val, lt_val=lt_val, parameters=parameters)
    stitched_trajectory_queue = queue.Queue()
    print("actual_gt_ids: ", len(actual_gt_ids))
    s1 = fragment_queue.qsize()
    

    # --------- start batch stitching --------- 
    # print("MCF Batch...")
    # t1 = time.time()
    # min_cost_flow_batch("west", fragment_queue, stitched_trajectory_queue, parameters)
    # # stitch_raw_trajectory_fragments("west", fragment_queue,stitched_trajectory_queue, parameters)
    # batch = list(stitched_trajectory_queue.queue)
    # s2 = stitched_trajectory_queue.qsize()
    # t2 = time.time()
    # print("{} fragment stitched to {} trajectories, taking {:.2f} sec".format(s1, s2, t2-t1))
    # # test
    # FGMT, IDS = test_fragments(gt_ids, batch)
    # print("FGMT: {}, IDS: {}".format(FGMT, IDS))
    
    

    # --------- start online stitching --------- 
    fragment_queue,actual_gt_ids,_ = read_to_queue(gt_ids=gt_ids, gt_val=gt_val, lt_val=lt_val, parameters=parameters)
    stitched_trajectory_queue = queue.Queue()
    t1 = time.time()
    min_cost_flow_online_alt_path("west", fragment_queue, stitched_trajectory_queue, parameters)
    online = list(stitched_trajectory_queue.queue)
    s2 = stitched_trajectory_queue.qsize()
    t2 = time.time()
    print("{} fragment stitched to {} trajectories, taking {:.2f} sec".format(s1, s2, t2-t1))
    
    # test
    FGMT, IDS = test_fragments(gt_ids, online)
    print("FGMT: {}, IDS: {}".format(FGMT, IDS))
    
    
    
    # for path_o in online:
    #     if path_o not in batch:
    #         print("difference: ", path_o)
    
    
    # plot runtime
    # import matplotlib.pyplot as plt
    # import numpy as np
    # plt.figure()
    # plt.scatter(np.arange(s1), time_arr, label="run time (sec)")
    # plt.xlabel("# fragments")
    # plt.ylabel("cumulative run time (sec)")
    
    # plt.figure()
    # plt.scatter(np.arange(s1), cache_arr, label = "cache size")
    # plt.xlabel("# fragments")
    # plt.ylabel("memory size (# fragments)")
    
    # plt.figure()
    # plt.scatter(np.arange(s1), nodes_arr, label = "cache size")
    # plt.xlabel("# fragments")
    # plt.ylabel("# nodes in G")
    
    
    