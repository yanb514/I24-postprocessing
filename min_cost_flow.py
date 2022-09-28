#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 15:17:59 2022

@author: yanbing_wang
"""
import queue
import signal
import time
from collections import defaultdict
from bson.objectid import ObjectId

from i24_database_api import DBClient
import i24_logger.log_writer as log_writer
from utils.utils_mcf import MOTGraphSingle
from utils.misc import calc_fit, find_overlap_idx


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

    # Get parameters
    ATTR_NAME = parameters["fragment_attr_name"]
    TIME_WIN = parameters["time_win"]
    RES_THRESH_X = parameters["residual_threshold_x"]
    RES_THRESH_Y = parameters["residual_threshold_y"]
    
    # Initialize tracking graph
    m = MOTGraphSingle(ATTR_NAME, parameters)
    counter = 0 # iterations for log

    
    GET_TIMEOUT = parameters["stitcher_timeout"]
    while sig_hdlr.run:
        try:
            try:
                fgmt = fragment_queue.get(timeout = GET_TIMEOUT) # a merged dictionary
                # stitcher_logger.debug("get fragment id: {}".format(raw_fgmt["_id"]))
                # fgmt = Fragment(raw_fgmt)
                
            except queue.Empty: # queue is empty
                stitcher_logger.info("Stitcher timed out after {} sec.".format(GET_TIMEOUT))
                all_paths = m.get_all_traj()
                
                for path in all_paths:
                    # filters = m.get_filters(path)
                    trajs = m.get_traj_dicts(path)
                    # stitched_trajectory_queue.put(path[::-1])
                    stitched_trajectory_queue.put(trajs[::-1])
                    
                    # stitcher_logger.info("Flushing out final trajectories in graph")
                    stitcher_logger.debug("** Flushing out {} fragments".format(len(path)),extra = None)
                    # dbw.write_one_trajectory(thread=True, fragment_ids = [ObjectId(o) for o in path[::-1]])
                
                # stitcher_logger.info("fragment_queue is empty, exit.")
                break
            
            # fgmt_id = getattr(fgmt, ATTR_NAME)
            fgmt_id = fgmt[ATTR_NAME]
            # fgmt = calc_fit(fgmt, RES_THRESH_X, RES_THRESH_Y)
            
            # RANSAC fit to determine the fit coef if it's a good track, otherwise reject
            # try:
            #     if len(fgmt["filter"]) == 0:
            #         # stitched_trajectory_queue.put(([fgmt_id], []))
            #         stitcher_logger.info("* skip {} - LOW CONF".format(fgmt_id))
            #         continue # skip this fgmt
            # except:
            #     pass
                
            m.add_node(fgmt)
            
            # print("* add ", fgmt_id)
            # print("**", m.G.edges(data=True))
            
            # run MCF
            m.augment_path(fgmt_id)
    
            # pop path if a path is ready
            # print("**", m.G.edges(data=True))
            all_paths = m.pop_path(time_thresh = fgmt["first_timestamp"] - TIME_WIN)  
            
            
            for path in all_paths:
                # filters = m.get_filters(path)
                # if not m.verify_path(path[::-1], cost_thresh = 15):
                #     stitcher_logger.info("** Stitched result not verified. Proceed anyways.")
                   
                trajs = m.get_traj_dicts(path)
                # stitched_trajectory_queue.put(path[::-1])
                stitched_trajectory_queue.put(trajs[::-1])
                
                
                # dbw.write_one_trajectory(thread=True, fragment_ids = [ObjectId(o) for o in path[::-1]])
                m.clean_graph(path)
                stitcher_logger.debug("** Stitched {} fragments".format(len(path)),extra = None)
             
            if counter % 100 == 0:
                stitcher_logger.debug("Graph nodes : {}, Graph edges: {}, Cache: {}".format(m.G.number_of_nodes(), m.G.number_of_edges(), len(m.cache)),extra = None)
                stitcher_logger.debug(f"raw queue: {fragment_queue.qsize()}, stitched queue: {stitched_trajectory_queue.qsize()}")
                counter = 0
            counter += 1
        
        
        except Exception as e: 
            if sig_hdlr.run:
                raise e
                # stitcher_logger.error("Unexpected exception: {}".format(e))
            else:
                stitcher_logger.warning("SIGINT detected. Exception:{}".format(e))
            break
            
        
    stitcher_logger.info("Exit stitcher")
    # stitcher_logger.info("Final count in stitched collection {}: {}".format(dbw.collection_name, dbw.count()))
    # del dbw
    # stitcher_logger.info("DBWriter closed. Exit.")
    # sys.exit()
        
    return   
    
    
def dummy_stitcher(direction, fragment_queue, stitched_trajectory_queue, parameters):
    '''
    incrementally fixing the matching
    '''
 
    # Initiate a logger
    stitcher_logger = log_writer.logger
    stitcher_logger.set_name("stitcher_"+direction)
    stitcher_logger.info("** min_cost_flow_online_alt_path starts", extra = None)
    setattr(stitcher_logger, "_default_logger_extra",  {})

    sig_hdlr = SignalHandler()

    GET_TIMEOUT = parameters["stitcher_timeout"]
    while sig_hdlr.run:
        try:
            try:
                fgmt = fragment_queue.get(timeout = GET_TIMEOUT)
                # stitcher_logger.debug("get fragment id: {}".format(raw_fgmt["_id"]))
                # fgmt = Fragment(raw_fgmt)
                stitched_trajectory_queue.put([fgmt])
                
            except queue.Empty: # queue is empty
                stitcher_logger.info("Stitcher timed out after {} sec.".format(GET_TIMEOUT))
 

        except Exception as e: 
            if sig_hdlr.run:
                raise e
                # stitcher_logger.error("Unexpected exception: {}".format(e))
            else:
                stitcher_logger.warning("SIGINT detected. Exception:{}".format(e))
            break
            
        
    stitcher_logger.info("Exit stitcher")
        
    return   
 

if __name__ == '__main__':

    
    import json
    import os
    from multi_opt import plot_track
    from merge import merge_fragments
    
    with open("config/parameters.json") as f:
        parameters = json.load(f)
    parameters["raw_trajectory_queue_get_timeout"] = 0.1
    with open(os.path.join(os.environ["USER_CONFIG_DIRECTORY"], "db_param.json")) as f:
        db_param = json.load(f)  
    # raw_collection = "morose_caribou--RAW_GT1" # collection name is the same in both databases
    rec_collection = "funny_squirrel--RAW_GT2__giggles"
    raw_collection = "funny_squirrel--RAW_GT2" 
    
    dbc = DBClient(**db_param)
    raw = dbc.client["trajectories"][raw_collection]
    rec = dbc.client["reconciled"][rec_collection]
    # gt = DBClient(**parameters["db_param"], database_name = trajectory_database, collection_name="groundtruth_scene_1")
    

    fragment_queue = queue.Queue()
    merged_queue = queue.Queue() 
    
    #funny_squirrel--RAW_GT2
    # f_ids = [ObjectId('6320f56babd7d7253149373c'), ObjectId('6320f587abd7d725314937c7')]
    # f_ids = [ObjectId('6320f576abd7d72531493774'), ObjectId('6320f579abd7d7253149378a')]
    # f_ids = [ObjectId('6320f578abd7d72531493781'), ObjectId('6320f592abd7d725314937fe')]
    # f_ids = [ObjectId('6320f575abd7d72531493772'), ObjectId('6320f56fabd7d7253149375a')]
    
    # giggles
    f_ids = [ObjectId('6320f56cabd7d72531493747'), ObjectId('6320f5a3abd7d7253149383e'),
             ObjectId('6320f56babd7d72531493741'), ObjectId('6320f5a1abd7d72531493836'),
             ObjectId('6320f56fabd7d7253149375b'), ObjectId('6320f577abd7d72531493780'),
             ObjectId('6320f573abd7d72531493769'), ObjectId('6320f5a4abd7d72531493844'),
             ObjectId('6320f57aabd7d7253149378d'), ObjectId('6320f5aaabd7d72531493863'),
             ObjectId('6320f579abd7d72531493789'), ObjectId('6320f5a0abd7d72531493833')]  
    
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
    
    merge_fragments("west", fragment_queue, merged_queue, parameters)
    
    # fragment_queue,actual_gt_ids,_ = read_to_queue(gt_ids=gt_ids, gt_val=gt_val, lt_val=lt_val, parameters=parameters)
    stitched_trajectory_queue = queue.Queue()
    t1 = time.time()
    min_cost_flow_online_alt_path("west", merged_queue, stitched_trajectory_queue, parameters)
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
    
    
    