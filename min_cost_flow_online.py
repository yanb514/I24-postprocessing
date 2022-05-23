#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 15:17:59 2022

@author: yanbing_wang
"""
import networkx as nx
import queue
from collections import deque 
from i24_database_api.db_reader import DBReader
from i24_database_api.db_writer import DBWriter
import os
from i24_configparse.parse import parse_cfg
from utils.data_structures import Fragment, PathCache, MOT_Graph

# TODO:
    # 1. check if the answers agree with nx.edmond_karp
    # 2. add more intelligent enter/exiting cost based on the direction and where they are relative to the road
    
    
    
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
    gt_ids = [7]
    fragment_ids = []
    gt_res = gt.read_query(query_filter = {"ID": {"$in": gt_ids}},
                            limit = 0)
    
    for gt_doc in gt_res:
        fragment_ids.extend(gt_doc["fragment_ids"])
    
    raw_res = raw.read_query(query_filter = {"_id": {"$in": fragment_ids}},
                              query_sort = [("last_timestamp", "ASC")])
    # raw_res = raw.read_query(query_filter = {"$and":[ {"last_timestamp": {"$gt": 545}}, 
    #                                                   {"last_timestamp": {"$lt": 580}},
    #                                                   {"_id": {"$in": fragment_ids}}]},
                               # query_sort = [("last_timestamp", "ASC")])
    
    # write fragments to queue
    fragment_queue = queue.Queue()
    # fragment_set = set()
    for doc in raw_res:
        fragment_queue.put(doc)
        # fragment_set.add(doc["_id"])
        
    fragment_size = fragment_queue.qsize()
    print("Queue size: ", fragment_size)

    return fragment_queue
    

            


def min_cost_flow_online(fragment_queue, parameters):
    '''
    iteratively solving a MCF problem on a smaller graph
    '''
    TIME_WIN = parameters.time_win
    
    # Get parameters
    TIME_WIN = parameters.time_win
    VARX = parameters.varx
    VARY = parameters.vary
    THRESH = parameters.thresh
    IDLE_TIME = parameters.idle_time
    TIMEOUT = parameters.raw_trajectory_queue_get_timeout
    
    # Initialize some data structures
    curr_fragments = deque()  # fragments that are in current window (left, right), sorted by last_timestamp
    # past_fragments = dict()  # set of ids indicate end of fragment ready to be matched, insertion ordered
    P = PathCache() # an LRU cache of Fragment object (see utils.data_structures)
    m = MOT_Graph(parameters)
    
    while not fragment_queue.empty():
        fragment = Fragment(fragment_queue.get(block = False))
        curr_fragments.append(fragment.ID) 
        left = fragment.first_timestamp - TIME_WIN   
        
        # compute fragment statistics (1d motion model)
        fragment.compute_stats()      
        P.add_node(fragment, "ID")
        
        # Add fragments to G if buffer time is reached
        while curr_fragments and P.get_fragment(curr_fragments[0]).last_timestamp < left: 
            past_id = curr_fragments.popleft()
            past_fragment = P.get_fragment(past_id)
            m.add_node(past_fragment) # update G with the new fragment

        # Start min-cost-flow
        m.min_cost_flow("s", "t")
        
        # Collapse paths
        m.find_all_post_paths(m.G, "t", "s")
        
        # P.union the chain
        # Delete nodes from G, only keep heads' pre and tail's post
        # aggregate edge cost
        # flip the edges back!
        
        for path, cost in m.all_paths:
            path = path[1:-1] # exclude s, head-pre, tail-post and t, everything in the middle should be deleted
            head = path[0]
            tail = path[-1]
            # Update P
            for i, node in enumerate(path):
                m.G.remove_node(node)
            for node in path:
                m.G.remove_node(node)
            print(path, cost)
            
        
        # Retrive paths
        
        
        
    
if __name__ == '__main__':
    
    # get parameters
    cwd = os.getcwd()
    cfg = "./config"
    config_path = os.path.join(cwd,cfg)
    os.environ["user_config_directory"] = config_path
    parameters = parse_cfg("TEST", cfg_name = "test_param.config")
    
    fragment_queue = read_to_queue(parameters)
    min_cost_flow_online(fragment_queue, parameters)
    