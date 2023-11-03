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
import platform
import _pickle as pickle

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
    # DIST_WIN = parameters["dist_win"]
    
    # Initialize tracking graph
    m = MOTGraphSingle(direction=direction, attr=ATTR_NAME, parameters=parameters)
    
    # counter = 0 # iterations for log
    # cum_t1,cum_t2,cum_t3 = 0,0,0
#     cum_t1_arr,cum_t2_arr,cum_t3_arr = [cum_t1],[cum_t2],[cum_t3]
#     input_arr = [0]
#     edges_arr, nodes_arr = [0], [0]
    
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
#                     print("queue empty path", path)
                    trajs = m.get_traj_dicts(path)
                    stitched_trajectory_queue.put(trajs[::-1])
                    input_obj += len(path)
                    output_obj += 1
#                     stitcher_logger.info("final stitch together {}".format([trj for trj in path]))
                
                # stitcher_logger.info("fragment_queue is empty, exit.")
                stitcher_logger.info("Final flushing {} raw fragments --> {} stitched fragments".format(input_obj, output_obj),extra = None)
                break
            
            except SIGINTException:
                stitcher_logger.warning("SIGINT detected when trying to get from queue. Exit")
                break

            fgmt_id = fgmt[ATTR_NAME]
            
            # t1 = time.time()
            # ============ Add node ============
            m.add_node(fgmt)
            stitcher_logger.debug("add_node {}".format(fgmt_id))
            # cum_t1 += time.time()-t1
            
            # ============ Path augment ============
            # t2 = time.time()
            m.augment_path(fgmt_id)
            stitcher_logger.debug("augment_path {}".format(fgmt_id))
            # cum_t2 += time.time()-t2
            
            # ============ Pop path ============
            # t3 = time.time()
            all_paths = m.pop_path(time_thresh = fgmt["first_timestamp"] - TIME_WIN)  
            stitcher_logger.debug("all_paths {}".format(len(all_paths) if type(all_paths) is list else []))
            
            num_cache = len(m.cache)
            num_nodes = m.G.number_of_nodes()
            
            for path in all_paths:
                # print("pop path", path)
                trajs = m.get_traj_dicts(path)
                stitched_trajectory_queue.put(trajs[::-1])
                m.clean_graph(path)
#                 stitcher_logger.info("stitch together {}".format([trj for trj in path]))
                
                input_obj += len(path)
                output_obj += 1
                
            stitcher_logger.debug("clean_path cache {}->{}, nodes {}->{}".format(num_cache, len(m.cache),
                                                                                 num_nodes, m.G.number_of_nodes()))
            # cum_t3 += time.time()-t3
            
            # heartbeat log
            now = time.time()
            if now - begin > HB:
                stitcher_logger.info("MCF graph # nodes: {}, # edges: {}, deque: {}, cache: {}".format(m.G.number_of_nodes(), m.G.number_of_edges(), len(m.in_graph_deque), len(m.cache)),extra = None)
                # stitcher_logger.info("Elapsed add:{:.2f}, augment:{:.2f}, pop:{:.2f}, total:{:.2f}".format(cum_t1, cum_t2, cum_t3, now-start), extra=None)
                stitcher_logger.info("{} raw fragments --> {} stitched fragments".format(input_obj, output_obj),extra = None)
                begin = time.time()

            # write aux info
#             cum_t1_arr.append(cum_t1)
#             cum_t2_arr.append(cum_t2)
#             cum_t3_arr.append(cum_t3)
#             input_arr.append(input_obj)
#             nodes_arr.append(m.G.number_of_nodes())
#             edges_arr.append(m.G.number_of_edges())
        
        except SIGINTException:  # SIGINT detected
            stitcher_logger.warning("SIGINT detected. Exit")
            break
            
        except (ConnectionResetError, BrokenPipeError, EOFError) as e:   
            stitcher_logger.warning("Connection error: {}".format(e))
            break
            
        except Exception as e: # other unknown exceptions are handled as error TODO UNTESTED CODE!
            stitcher_logger.error("Other error: {}, push all processed trajs to queue".format(e))
            
            all_paths = m.get_all_traj()
            for path in all_paths:
                # print("exception path", path)
                trajs = m.get_traj_dicts(path)
                stitched_trajectory_queue.put(trajs[::-1])
                input_obj += len(path)
                output_obj += 1
            stitcher_logger.info("Final flushing {} raw fragments --> {} stitched fragments".format(input_obj, 
                                                                                                    output_obj),
                                 extra = None)
            break
               
    # write aux info to pickle
#     if len(cum_t1_arr) > 1:
#         aux_info = {"AddNode":cum_t1_arr, "PushFlow":cum_t2_arr, "CleanGraph":cum_t3_arr,
#                     "nodes":nodes_arr, "edges":edges_arr, "input":input_arr}
#         with open(f'{stitcher_logger._logger.name}.pkl', 'wb') as handle:
#             pickle.dump(aux_info, handle)
        
    stitcher_logger.info("Exit stitcher")
    if "darwin" not in platform.system().lower():
        sys.exit()
  
    return   
 
 

if __name__ == '__main__':

    
    import json
    import os
    
    from merge import merge_fragments
    import multiprocessing as mp
    
    with open("config/parameters.json") as f:
        parameters = json.load(f)
    parameters["raw_trajectory_queue_get_timeout"] = 0.1
    with open(os.path.join(os.environ["USER_CONFIG_DIRECTORY"], "db_param.json")) as f:
        db_param = json.load(f)  
    # raw_collection = "morose_caribou--RAW_GT1" # collection name is the same in both databases
    raw_collection = "64541c307ef4a8ecc60a258d" # collection name is the same in both databases
    rec_collection = "64541c307ef4a8ecc60a258d__v2post3"
	
    
    dbc = DBClient(**db_param)
    raw = dbc.client["trajectories"][raw_collection]
    rec = dbc.client["reconciled"][rec_collection]
    temp = dbc.client["temp"][rec_collection]
    # gt = DBClient(**parameters["db_param"], database_name = trajectory_database, collection_name="groundtruth_scene_1")
    

    # mp_manager = mp.Manager()
    fragment_queue = queue.Queue()
    merged_queue = queue.Queue() 
    
    
    f_ids = [ObjectId("6454368e12e55fcbb879d4c5"),
    ObjectId("645436f012e55fcbb879d5c0"),
    ]
    # get parameters for fitting
    # RES_THRESH_X = parameters["residual_threshold_x"]
    # RES_THRESH_Y = parameters["residual_threshold_y"]
    # CONF_THRESH = parameters["conf_threshold"],
    # REMAIN_THRESH = parameters["remain_threshold"]
    # from data_feed import add_filter
    
    docs = []
    for traj in temp.find({"_id": {"$in": f_ids}}).sort( "last_timestamp", 1 ):
        # print(traj["fragment_ids"])
        fragment_queue.put(traj)
        docs.append(traj)
    s1 = fragment_queue.qsize()
    # plot_track(docs)
    

    # --------- start online stitching --------- 
    parameters["stitcher_timeout"] = 0.1
    parameters["merger_timeout"] = 0.01
    parameters["compute_node_list"] = ["videonode1", "videonode2", "videonode3", "videonode4", "videonode5", 
                                       "videonode6", "videonode7", "videonode8", "videonode9", "devvideo1"]
    parameters["stitcher_mode"] = "master"
    parameters["time_win"] = parameters["master_time_win"]

    merge_fragments("west", fragment_queue, merged_queue, parameters)
    # fragment_queue,actual_gt_ids,_ = read_to_queue(gt_ids=gt_ids, gt_val=gt_val, lt_val=lt_val, parameters=parameters)
    stitched_trajectory_queue = queue.Queue()
    t1 = time.time()
    min_cost_flow_online_alt_path("wb", merged_queue, stitched_trajectory_queue, parameters)
    online = list(stitched_trajectory_queue.queue)
    s2 = stitched_trajectory_queue.qsize()
    t2 = time.time()
    print("{} fragment stitched to {} trajectories, taking {:.2f} sec".format(s1, s2, t2-t1))
    
    for stitched_trajs in online:
        print(len(stitched_trajs), [str(item["_id"])[-4:] for item in stitched_trajs])
    # # plot
    # 
    # docs = []
    # while not stitched_trajectory_queue.empty():
    #     d = stitched_trajectory_queue.get()[0]
    #     docs.append(d)
    # plot_track(docs)
    
    
    