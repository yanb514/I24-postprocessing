#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 15:17:59 2022

@author: yanbing_wang
"""
import queue
import heapq
import signal
import time
from collections import defaultdict
from bson.objectid import ObjectId
import sys

from i24_database_api import DBClient
import i24_logger.log_writer as log_writer
from utils.utils_mcf import MOTGraphSingle
from utils.misc import calc_fit, find_overlap_idx
from utils.utils_opt import combine_fragments, resample
import multiprocessing

lock = multiprocessing.Lock()


class SIGINTException(SystemExit):
    pass

def soft_stop_hdlr(sig, action):
    '''
    Signal handling for SIGINT
    Soft terminate current process. Close ports and exit.
    '''
    raise SIGINTException # so to exit the while true loop
           
    
# @catch_critical(errors = (Exception))  
def read_to_queue(gt_ids, gt_val, lt_val, parameters):
    '''
    construct MOT graph from fragment list based on the specified loss function
    '''
    # connect to database
    raw = DBClient(**parameters, collection_name = "")
    print("connected to raw collection")
    gt = DBClient(host=parameters.default_host, port=parameters.default_port, username=parameters.readonly_user,   
                    password=parameters.default_password,
                    database_name=parameters.db_name, collection_name=parameters.gt_collection)
    print("connected to gt collection")
    stitched = DBClient(host=parameters.default_host, port=parameters.default_port, 
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



    
def min_cost_flow_online_alt_path(direction, fragment_queue, stitched_trajectory_queue, parameters, name=None):
    '''
    incrementally fixing the matching
    '''
    signal.signal(signal.SIGINT, soft_stop_hdlr)
    
    # Initiate a logger
    stitcher_logger = log_writer.logger
    if name:
        stitcher_logger.set_name(name)
    else:
        stitcher_logger.set_name("stitcher_"+direction)
    stitcher_logger.info("** min_cost_flow_online_alt_path starts", extra = None)
    # setattr(stitcher_logger, "_default_logger_extra",  {})

    # Get parameters
    ATTR_NAME = parameters["fragment_attr_name"]
    TIME_WIN = parameters["time_win"]
    DIST_WIN = parameters["dist_win"]
    
    # Initialize tracking graph
    m = MOTGraphSingle(direction=direction, attr=ATTR_NAME, parameters=parameters)
    
    # counter = 0 # iterations for log
    cum_t1 = 0
    cum_t2 = 0
    cum_t3 = 0
    
    GET_TIMEOUT = parameters["stitcher_timeout"]
    HB = parameters["log_heartbeat"]
    begin = time.time()
    input_obj = 0
    output_obj = 0
    
    while True:
        try:
            try:
                fgmt = fragment_queue.get(timeout = GET_TIMEOUT) # a merged dictionary
                
            except queue.Empty: # queue is empty
                stitcher_logger.info("Getting from fragment_queue timed out after {} sec.".format(GET_TIMEOUT))
                all_paths = m.get_all_traj()
                
                for path in all_paths:
                    trajs = m.get_traj_dicts(path)
                    stitched_trajectory_queue.put(trajs[::-1])

                    input_obj += len(path)
                    output_obj += 1
                
                # stitcher_logger.info("fragment_queue is empty, exit.")
                stitcher_logger.info("Final flushing {} raw fragments --> {} stitched fragments".format(input_obj, output_obj),extra = None)
                break
            
            except SIGINTException:
                stitcher_logger.warning("SIGINT detected when trying to get from queue. Exit")
                break

            fgmt_id = fgmt[ATTR_NAME]

            t1 = time.time()
            m.add_node(fgmt)
            cum_t1 += time.time()-t1
            
            # Path augment
            t2 = time.time()
            m.augment_path(fgmt_id)
            cum_t2 += time.time()-t2
            
            # Pop path if a path is ready
            t3 = time.time()
            # TODO: double check direction when have wb data
            dist_thresh = fgmt["ending_x"]-DIST_WIN if direction=="eb" else fgmt["ending_x"]+DIST_WIN
            all_paths = m.pop_path(time_thresh = fgmt["first_timestamp"] - TIME_WIN,
                                   dist_thresh = dist_thresh)  
            
            for path in all_paths:
                   
                trajs = m.get_traj_dicts(path)
                stitched_trajectory_queue.put(trajs[::-1])
                m.clean_graph(path)
                # stitcher_logger.info("last timestamp: {:.2f}".format(trajs[0]["last_timestamp"]))
        
                input_obj += len(path)
                output_obj += 1
                # stitcher_logger.debug("** Stitched {} fragments".format(len(path)),extra = None)
                
                
             
            cum_t3 += time.time()-t3
            
            # heartbeat log
            now = time.time()
            if now - begin > HB:
                stitcher_logger.info("MCF graph # nodes: {}, # edges: {}, deque: {}, cache: {}".format(m.G.number_of_nodes(), m.G.number_of_edges(), len(m.in_graph_deque), len(m.cache)),extra = None)
                # stitcher_logger.info("Elapsed add:{:.2f}, augment:{:.2f}, pop:{:.2f}, total:{:.2f}".format(cum_t1, cum_t2, cum_t3, now-start), extra=None)
                stitcher_logger.info("{} raw fragments --> {} stitched fragments".format(input_obj, output_obj),extra = None)
                begin = time.time()

        
        except SIGINTException:  # SIGINT detected
            stitcher_logger.warning("SIGINT detected. Exit")
            break
            
        except (ConnectionResetError, BrokenPipeError, EOFError) as e:   
            stitcher_logger.warning("Connection error: {}".format(e))
            break
                 
    stitcher_logger.info("Exit stitcher")
    sys.exit()
        
    return   
 

    
def stitch_rolling(direction, fragment_queue_prev, fragment_queue_curr, next_queue, stitched_trajectory_queue, parameters, node_idx):
    """
    same with online_alt_path, but for consecutively stitch two adjacent nodes
    :param direction
    :param fragment_queue_prev: from the last transition stitcher
    :param fragment_queue_curr: from local stitcher. this queue should be consumed first
    :param next_queue
    :param stitched_trajectory_queue
    :param parameters
    :param node_idx: 0-indexed
    
    1. reorder fragments from fragment_queue_prev, fragment_queue_curr in a heap
    2. pop from heap and build MCF if last_timestamp condition is met (*)
    3. update parameters["transition_last_timestamp"][direction][node_idx]
    
    """
    signal.signal(signal.SIGINT, soft_stop_hdlr)
    
    # Initiate a logger
    stitcher_logger = log_writer.logger
    name = "stitch_"+str(node_idx+1)+str(node_idx+2)+"_"+direction
    stitcher_logger.set_name(name)
    stitcher_logger.info("** stitch_rolling starts", extra = None)
    setattr(stitcher_logger, "_default_logger_extra",  {})

    # Get parameters
    ATTR_NAME = parameters["fragment_attr_name"]
    TIME_WIN = parameters["time_win"]
    
    # Initialize tracking graph
    m = MOTGraphSingle(ATTR_NAME, parameters)
    
    # counter = 0 # iterations for log
    cum_t1 = 0
    cum_t2 = 0
    cum_t3 = 0
    
    # GET_TIMEOUT = parameters["stitcher_timeout"]
    HB = parameters["log_heartbeat"]
    DELAY = parameters["delay"]
   
    # TODO: ASSUMPTION!!! local queue do not change size anymore, and should be consumed first
    h = []
    while not fragment_queue_curr.empty():
        traj_docs = fragment_queue_curr.get(block=False)
        combined_trajectory = combine_fragments(traj_docs)
        resampled_trajectory = resample(combined_trajectory, dt=0.1, fillnan=True)
        heapq.heappush(h, (resampled_trajectory["last_timestamp"],resampled_trajectory["_id"], resampled_trajectory)) # orderd by last_timestamp but it may not be unique
    stitcher_logger.info("Heap size after emptying fragment_queue_curr: {}".format(len(h)))
    
    begin = time.time()
    start = begin
    input_obj = 0
    output_obj = 0
    ready = True if node_idx == 0 else False
    
    
    while True:
        try:
            try:
                traj_doc = fragment_queue_prev.get(timeout = (node_idx+1)*DELAY)
                combined_trajectory = combine_fragments(traj_doc)
                resampled_trajectory = resample(combined_trajectory, dt=0.1, fillnan=True)
                heapq.heappush(h, (resampled_trajectory["last_timestamp"],resampled_trajectory["_id"], resampled_trajectory)) # orderd by last_timestamp but it may not be unique
                
            except queue.Empty: # queue is empty
                stitcher_logger.info("Getting from fragment_queue timed out after {} sec.".format((node_idx+1)*DELAY))
                while h:
                    _,fgmt_id, fgmt = heapq.heappop(h)        
                    m.add_node(fgmt)
                    m.augment_path(fgmt_id)  
                
                # Path augment
                all_paths = m.get_all_traj()
                
                for path in all_paths:
                    trajs = m.get_traj_dicts(path)
                    input_obj += len(path)
                    output_obj += 1
                    with lock:
                        parameters["transition_last_timestamp_"+direction][node_idx] = trajs[0]["last_timestamp"]
                        print(parameters["transition_last_timestamp_"+direction][node_idx] )
                    # determine where to put the result
                    last_compute_node = max([traj["compute_node_id"] for traj in trajs])
                    # stitcher_logger.info(last_compute_node)
                    
                    if int(last_compute_node[-1]) == node_idx+1:  
                        # print("put to curent queue")
                        stitched_trajectory_queue.put(trajs[::-1])
                    elif int(last_compute_node[-1]) == node_idx+2:
                        # print("put to the next transition queue")
                        next_queue.put(trajs[::-1])
                    else:
                        print(last_compute_node, " somthing's wrong")

                
                # stitcher_logger.info("fragment_queue is empty, exit.")
                stitcher_logger.info("Flushing {} raw fragments --> {} stitched fragments".format(input_obj, output_obj),extra = None)
                break
            
            except SIGINTException:
                stitcher_logger.warning("SIGINT detected when trying to get from queue. Exit")
                break

            # check if the earliest of h is ready to be processed
            while ready or (len(h)>0 and h[0][2]["last_timestamp"] < parameters["transition_last_timestamp_"+direction][node_idx-1]-TIME_WIN):
                _,fgmt_id, fgmt = heapq.heappop(h)        
                m.add_node(fgmt)
                
                # Path augment
                m.augment_path(fgmt_id)  
                
                # Pop path if a path is ready
                all_paths = m.pop_path(time_thresh = fgmt["first_timestamp"] - TIME_WIN)  
                
                for path in all_paths:
                       
                    trajs = m.get_traj_dicts(path)
                    m.clean_graph(path)
                    # stitcher_logger.info("last timestamp: {:.2f}".format(trajs[0]["last_timestamp"]))
            
                    input_obj += len(path)
                    output_obj += 1
                    with lock:
                        parameters["transition_last_timestamp_"+direction][node_idx] = trajs[0]["last_timestamp"]
                    
                    # determine where to put the result
                    last_compute_node = max([traj["compute_node_id"] for traj in trajs])
                    if int(last_compute_node[-1]) == node_idx+1:  
                        print("put to curent queue")
                        stitched_trajectory_queue.put(trajs[::-1])
                    elif int(last_compute_node[-1]) == node_idx+2:
                        # print("put to the next transition queue")
                        next_queue.put(trajs[::-1])
                    else:
                        print(last_compute_node, " somthing's wrong")
                
                # heartbeat log
                now = time.time()
                if now - start > HB:
                    stitcher_logger.info("MCF graph # nodes: {}, # edges: {}, deque: {}, cache: {}".format(m.G.number_of_nodes(), m.G.number_of_edges(), len(m.in_graph_deque), len(m.cache)),extra = None)
                    # stitcher_logger.info("Elapsed add:{:.2f}, augment:{:.2f}, pop:{:.2f}, total:{:.2f}".format(cum_t1, cum_t2, cum_t3, now-start), extra=None)
                    stitcher_logger.info("{} raw fragments --> {} stitched fragments".format(input_obj, output_obj),extra = None)
                    begin = time.time()
    
            
        except SIGINTException:  # SIGINT detected
            stitcher_logger.warning("SIGINT detected. Exit")
            break
            
        except (ConnectionResetError, BrokenPipeError, EOFError) as e:   
            stitcher_logger.warning("Connection error: {}".format(e))
            break
                 
    stitcher_logger.info("Exit stitcher")
    sys.exit()
        
    return   
    

if __name__ == '__main__':

    
    import json
    import os
    # from multi_opt import plot_track
    from merge import merge_fragments
    
    with open("config/parameters.json") as f:
        parameters = json.load(f)
    parameters["raw_trajectory_queue_get_timeout"] = 0.1
    with open(os.path.join(os.environ["USER_CONFIG_DIRECTORY"], "db_param.json")) as f:
        db_param = json.load(f)  
    # raw_collection = "morose_caribou--RAW_GT1" # collection name is the same in both databases
    raw_collection = "tm_200_raw_v4.3" # collection name is the same in both databases
    rec_collection = "tm_200_raw_v4.3__0"
    
    
    dbc = DBClient(**db_param)
    raw = dbc.client["transmodeler"][raw_collection]
    rec = dbc.client["reconciled"][rec_collection]
    # gt = DBClient(**parameters["db_param"], database_name = trajectory_database, collection_name="groundtruth_scene_1")
    

    fragment_queue = queue.Queue()
    merged_queue = queue.Queue() 
    
    
    f_ids = [ObjectId("64173679e7de4a0b772b8299"), 
             ObjectId("64173679e7de4a0b772b8298"),
            ]
    # get parameters for fitting
    RES_THRESH_X = parameters["residual_threshold_x"]
    RES_THRESH_Y = parameters["residual_threshold_y"]
    CONF_THRESH = parameters["conf_threshold"],
    REMAIN_THRESH = parameters["remain_threshold"]
    # from data_feed import add_filter
    
    docs = []
    for traj in raw.find({"_id": {"$in": f_ids}}).sort( "last_timestamp", 1 ):
        # print(traj["fragment_ids"])
        fragment_queue.put(traj)
        docs.append(traj)
    s1 = fragment_queue.qsize()
    # plot_track(docs)
    

    # --------- start online stitching --------- 
    parameters["stitcher_timeout"] = 0.1
    parameters["merger_timeout"] = 0.01
    
    # merge_fragments("eb", fragment_queue, merged_queue, parameters)
    
    # fragment_queue,actual_gt_ids,_ = read_to_queue(gt_ids=gt_ids, gt_val=gt_val, lt_val=lt_val, parameters=parameters)
    stitched_trajectory_queue = queue.Queue()
    t1 = time.time()
    min_cost_flow_online_alt_path("eb", fragment_queue, stitched_trajectory_queue, parameters)
    online = list(stitched_trajectory_queue.queue)
    s2 = stitched_trajectory_queue.qsize()
    t2 = time.time()
    print("{} fragment stitched to {} trajectories, taking {:.2f} sec".format(s1, s2, t2-t1))
    
    # # plot
    # 
    # docs = []
    # while not stitched_trajectory_queue.empty():
    #     d = stitched_trajectory_queue.get()[0]
    #     docs.append(d)
    # plot_track(docs)
    
    
    