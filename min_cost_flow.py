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
from utils.utils_mcf import Fragment, MOTGraphSingle



# config_path = os.path.join(os.getcwd(),"config")
# os.environ["user_config_directory"] = config_path
# parameters = parse_cfg("DEBUG", cfg_name = "test_param.config")


# TODO:
    # 1. check if the answers agree with nx.edmond_karp
    # 2. add more intelligent enter/exiting cost based on the direction and where they are relative to the road
    
# Signal handling: in live data read, SIGINT and SIGUSR1 are handled in the same way
class SignalHandler():

    run = True
    count_sigint = 0 # count the number of times a SIGINT is received
    count_sigusr = 0
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    def __init__(self):
        signal.signal(signal.SIGINT, self.soft_stop)
        signal.signal(signal.SIGUSR1, self.finish_processing)
    
    def soft_stop(self, *args):
        signal.signal(signal.SIGPIPE,signal.SIG_DFL) # reset SIGPIPE so that no BrokePipeError when SIGINT is received
        self.run = False
        self.count_sigint += 1
        # stitcher_logger.info("{} detected {} times".format(signal.Signals(args[0]).name, self.count_sigint))
        
    def finish_processing(self, *args):
        # do nothing
        self.count_sigusr += 1
        # stitcher_logger.info("{} detected {} times".format(signal.Signals(args[0]).name, self.count_sigusr))
        
        siginfo = signal.sigwaitinfo({signal.SIGUSR1})
        print("py: got %d from %d by user %d\n" % (siginfo.si_signo,
                                                 siginfo.si_pid,
                                                 siginfo.si_uid))
        
        
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



    
def min_cost_flow_online_alt_path(direction, fragment_queue, stitched_trajectory_queue, parameters):
    '''
    incrementally fixing the matching
    '''
 
    # Initiate a logger
    stitcher_logger = log_writer.logger
    stitcher_logger.set_name("stitcher_"+direction)
    stitcher_logger.info("** min_cost_flow_online_alt_path starts", extra = None)
    setattr(stitcher_logger, "_default_logger_extra",  {})

    sig_hdlr = SignalHandler()

    # Make a database connection for writing
    # schema_path = os.path.join(os.environ["USER_CONFIG_DIRECTORY"],parameters.stitched_schema_path)
    # dbw = DBWriter(parameters, collection_name = parameters.raw_collection+"_stitched", schema_file=schema_path)
    # dbw.collection.drop()
    # dbw = DBWriter(parameters, collection_name = parameters.raw_collection+"_stitched", schema_file=schema_path)
    
    # Get parameters
    ATTR_NAME = parameters.fragment_attr_name
    TIME_WIN = parameters.time_win
    
    # Initialize tracking graph
    m = MOTGraphSingle(ATTR_NAME, parameters)
    counter = 0 # iterations for log
    
    while sig_hdlr.run:
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
                    # dbw.write_one_trajectory(thread=True, fragment_ids = path[::-1])
                
                stitcher_logger.info("fragment_queue is empty, exit.")
                break
            
            # RANSAC fit to determine the fit coef if it's a good track, otherwise reject
            if not fgmt.ransac_fit():
                print('remove ',fgmt)
                continue # skip this fgmt
                
            m.add_node(fgmt)
            fgmt_id = getattr(fgmt, ATTR_NAME)
            # print("* add ", fgmt_id)
            # print("**", m.G.edges(data=True))
            
            # run MCF
            m.augment_path(fgmt_id)
    
            # pop path if a path is ready
            # print("**", m.G.edges(data=True))
            all_paths = m.pop_path(time_thresh = fgmt.first_timestamp - TIME_WIN)
            
            
            
            for path in all_paths:
                # stitcher_logger.debug("path: {}".format(path))
                stitched_trajectory_queue.put(path[::-1])
                # dbw.write_one_trajectory(thread=True, fragment_ids = path[::-1])
                m.clean_graph(path)
                if len(path)>1:
                    stitcher_logger.info("** stitched {} fragments into one trajectory".format(len(path)),extra = None)
             
            if counter % 100 == 0:
                stitcher_logger.debug("Graph nodes : {}, Graph edges: {}".format(m.G.number_of_nodes(), m.G.number_of_edges()),extra = None)
                stitcher_logger.debug(f"raw queue: {fragment_queue.qsize()}, stitched queue: {stitched_trajectory_queue.qsize()}")
                counter = 0
            counter += 1
        
        
        except Exception as e: 
            if sig_hdlr.run:
                stitcher_logger.warning("No signals received. Exception: {}".format(e))
            else:
                stitcher_logger.warning("SIGINT detected. Exception:{}".format(e))
            break
            
        
    stitcher_logger.info("Exit stitcher while loop")
    # del dbw
    # stitcher_logger.info("DBWriter closed. Exit.")
    # sys.exit()
        
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
    os.environ["USER_CONFIG_DIRECTORY"] = config_path
    os.environ["my_config_section"] = "TEST"
    parameters = parse_cfg("my_config_section", cfg_name = "test_param.config")
    parameters.raw_trajectory_queue_get_timeout = 0.1

    
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
    f_ids = [ObjectId('62e018c427b64c6330545fa6'), ObjectId('62e0190427b64c6330545fe6'), ObjectId('62e0190427b64c6330545fe7')] # they should stitch to two
    # f_ids = [ ObjectId('62e0193027b64c6330546003'), ObjectId('62e0194627b64c6330546016'), ObjectId('62e0195327b64c6330546026'),  ObjectId('62e0196227b64c6330546035')]
    # f_ids = [ ObjectId('62e0198927b64c6330546059'), ObjectId('62e0199527b64c6330546068'), ObjectId('62e019a827b64c633054607d'),  ObjectId('62e019be27b64c6330546096')]
    
    raw = DBReader(parameters, username = "i24-data", collection_name="morose_panda--RAW_GT1")
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
    # import matplotlib.pyplot as plt
    # ax1 = plt.subplot(131)
    # ax2 = plt.subplot(132)
    # ax3 = plt.subplot(133)

    # # d = stitched_trajectory_queue.get(block=False)
    # # plt.scatter(d["timestamp"], d["x_position"], c="r", s=0.2, label="reconciled")
    # for f_id in f_ids:
    #     f = raw.find_one("_id", f_id)
    #     ax1.scatter(f["timestamp"], f["x_position"], s=0.5, label=f_id)
    #     ax1.set_title("time vs. x")
    #     ax2.scatter(f["timestamp"], f["y_position"], s=0.5, label=f_id)
    #     ax2.set_title("time vs. y")
    #     ax3.scatter(f["x_position"], f["y_position"], s=0.5, label=f_id)
    #     ax3.set_title("x vs. y")
    # plt.legend()
    
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
    
    
    