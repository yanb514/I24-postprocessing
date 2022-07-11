#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 15:17:59 2022

@author: yanbing_wang
"""
import queue
from collections import deque 
import os
import signal
import time
import heapq
from collections import defaultdict
import sys
# sys.path.append('../')

from i24_database_api.db_reader import DBReader
from i24_database_api.db_writer import DBWriter
import i24_logger.log_writer as log_writer
from i24_configparse import parse_cfg
from utils.utils_mcf import Fragment, PathCache, MOT_Graph, MOTGraphSingle, SortedDLL



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

    IDLE_TIME = parameters.idle_time

    
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
            fgmt = Fragment(fragment_queue.get(block=True))
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

    

def dummy_stitcher(old_q, new_q):
    # Initiate a logger
    stitcher_logger = log_writer.logger
    stitcher_logger.set_name("dummy")
    stitcher_logger.info("** min_cost_flow_online_alt_path starts", extra = None)

    # Signal handling    
    signal.signal(signal.SIGINT, signal.SIG_IGN)    
    signal.signal(signal.SIGPIPE,signal.SIG_DFL) # reset SIGPIPE so that no BrokePipeError when SIGINT is received
    
    while True:
        try:
            x = old_q.get(timeout = 3)
        except:
            stitcher_logger.info("old_q is empty, exit")
            sys.exit(2)
        
        time.sleep(0.1)
        
        new_q.put([x["_id"]])
        stitcher_logger.info("old_q size: {}, new_q size:{}".format(old_q.qsize(),new_q.qsize()))
        
        
    stitcher_logger.info("Exiting dummy stitcher while loop")
    sys.exit(2)
        
        

    
def min_cost_flow_online_alt_path(direction, fragment_queue, stitched_trajectory_queue, parameters):
    '''
    incrementally fixing the matching
    '''
 
    # Initiate a logger
    stitcher_logger = log_writer.logger
    stitcher_logger.set_name("stitcher_"+direction)
    stitcher_logger.info("** min_cost_flow_online_alt_path starts", extra = None)

    # Signal handling: 
    # SIGINT (sent from parent process) raises KeyboardInterrupt,  close dbw and soft exit
    # SIGUSR1 is ignored. The process terminates when queue is empty
    def sigusr_handler(sigusr, frame):
        stitcher_logger.warning("SIGUSR1 detected in stitcher. Finish processing current queues.")
        signal.signal(sigusr, signal.SIG_IGN)    
        signal.signal(signal.SIGPIPE,signal.SIG_DFL) # reset SIGPIPE so that no BrokePipeError when SIGINT is received
        
    # signal.signal(signal.SIGUSR1, handler)
    signal.signal(signal.SIGINT, signal.SIG_IGN) #ignore signal sent from keyboard.
    signal.signal(signal.SIGUSR1, sigusr_handler)
    
    # Make a database connection for writing
    schema_path = os.path.join(os.environ["user_config_directory"],parameters.stitched_schema_path)
    dbw = DBWriter(parameters, collection_name = parameters.stitched_collection, schema_file=schema_path)

    # Get parameters
    ATTR_NAME = parameters.fragment_attr_name
    TIME_WIN = parameters.time_win
    
    # Initialize tracking graph
    m = MOTGraphSingle(ATTR_NAME, parameters)
    counter = 0 # iterations for log
    
    while True:
        try:
            try:
                raw_fgmt = fragment_queue.get(timeout = parameters.raw_trajectory_queue_get_timeout)
                # stitcher_logger.debug("get fragment id: {}".format(raw_fgmt["_id"]))
                fgmt = Fragment(raw_fgmt)
                
            except queue.Empty: # queue is empty
                all_paths = m.get_all_traj()
                
                for path in all_paths:
                    # stitcher_logger.info("Flushing out final trajectories in graph")
                    stitcher_logger.info("** Flushing out {} fragments into one trajectory".format(len(path)),extra = None)
                    stitched_trajectory_queue.put(path[::-1])
                    dbw.write_one_trajectory(thread=True, fragment_ids = path[::-1])
                
                stitcher_logger.info("fragment_queue is empty, exit.")
                break
            
            fgmt.compute_stats()
            m.add_node(fgmt)
            # print(m.G.edges(data=True))
            
            # run MCF
            fgmt_id = getattr(fgmt, ATTR_NAME)
            m.augment_path(fgmt_id)
    
            # pop path if a path is ready
            all_paths = m.pop_path(time_thresh = fgmt.first_timestamp - TIME_WIN)
            
            for path in all_paths:
                # stitcher_logger.debug("path: {}".format(path))
                stitched_trajectory_queue.put(path[::-1])
                dbw.write_one_trajectory(thread=True, fragment_ids = path[::-1])
                m.clean_graph(path)
                if len(path)>1:
                    stitcher_logger.info("** stitched {} fragments into one trajectory".format(len(path)),extra = None)
             
            if counter % 100 == 0:
                stitcher_logger.debug("Graph nodes : {}, Graph edges: {}".format(m.G.number_of_nodes(), m.G.number_of_edges()),extra = None)
                counter = 0
            counter += 1
        
        except (KeyboardInterrupt, BrokenPipeError, EOFError, AttributeError):
            # handle SIGINT here
            del dbw
            stitcher_logger.info("DBWriter closed. {}")
            stitcher_logger.warning("SIGINT detected. Exiting stitcher")
            break
            
    stitcher_logger.info("Exiting stitcher while loop")
    # del stitcher_logger
    # os.kill(os.getppid(), signal.SIGTERM) # for mac, really kill this process so that p.is_alive = False -> permissionError
    sys.exit()
        
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
    os.environ["my_config_section"] = "TEST"
    parameters = parse_cfg("my_config_section", cfg_name = "test_param.config")
    
    # read to queue
    # gt_ids = [i for i in range(150, 180)]
    # gt_ids = [150]
    # gt_val = 30
    # lt_val = 100
    
    # fragment_queue,actual_gt_ids,_ = read_to_queue(gt_ids=gt_ids, gt_val=gt_val, lt_val=lt_val, parameters=parameters)
    # stitched_trajectory_queue = queue.Queue()
    # print("actual_gt_ids: ", len(actual_gt_ids))
    # s1 = fragment_queue.qsize()
    

    from bson.objectid import ObjectId
    fragment_queue = queue.Queue()
    f_ids = [ObjectId('62c713dfc77930b8d9533454'), ObjectId('62c713fbc77930b8d9533462')]
    raw = DBReader(parameters, collection_name="batch_5_07072022")
    for f_id in f_ids:
        f = raw.find_one("_id", f_id)
        fragment_queue.put(f)
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
    # fragment_queue,actual_gt_ids,_ = read_to_queue(gt_ids=gt_ids, gt_val=gt_val, lt_val=lt_val, parameters=parameters)
    stitched_trajectory_queue = queue.Queue()
    t1 = time.time()
    min_cost_flow_online_alt_path("west", fragment_queue, stitched_trajectory_queue, parameters)
    online = list(stitched_trajectory_queue.queue)
    s2 = stitched_trajectory_queue.qsize()
    t2 = time.time()
    print("{} fragment stitched to {} trajectories, taking {:.2f} sec".format(s1, s2, t2-t1))
    
    # test
    # FGMT, IDS = test_fragments(gt_ids, online)
    # print("FGMT: {}, IDS: {}".format(FGMT, IDS))
    
    
    
    # for path_o in online:
    #     if path_o not in batch:
    #         print("difference: ", path_o)
    #%%
    import matplotlib.pyplot as plt
    plt.figure()

    # d = stitched_trajectory_queue.get(block=False)
    # plt.scatter(d["timestamp"], d["x_position"], c="r", s=0.2, label="reconciled")
    for f_id in f_ids:
        f = raw.find_one("_id", f_id)
        plt.scatter(f["timestamp"], f["x_position"], c="b", s=0.5, label="raw")
    plt.legend()
    
    # plot runtime
    
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
    
    
    