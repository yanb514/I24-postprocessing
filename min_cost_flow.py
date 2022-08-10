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
from utils.utils_mcf import Fragment, MOTGraphSingle


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
    
    # Initialize tracking graph
    m = MOTGraphSingle(ATTR_NAME, parameters)
    counter = 0 # iterations for log
    
    # wait to get stitched collection name
    # while parameters["stitched_collection"]=="":
    #     time.sleep(1)
        
    # dbw = DBClient(**parameters["db_param"], database_name = parameters["stitched_database"],
    #                collection_name = parameters["stitched_collection"])
    
    GET_TIMEOUT = parameters["raw_trajectory_queue_get_timeout"]
    while sig_hdlr.run:
        try:
            try:
                raw_fgmt = fragment_queue.get(timeout = GET_TIMEOUT)
                # stitcher_logger.debug("get fragment id: {}".format(raw_fgmt["_id"]))
                fgmt = Fragment(raw_fgmt)
                
            except queue.Empty: # queue is empty
                all_paths = m.get_all_traj()
                
                for path in all_paths:
                    # stitcher_logger.info("Flushing out final trajectories in graph")
                    stitcher_logger.info("** Flushing out {} fragments".format(len(path)),extra = None)
                    stitched_trajectory_queue.put(path[::-1])
                    # dbw.write_one_trajectory(thread=True, fragment_ids = [ObjectId(o) for o in path[::-1]])
                
                stitcher_logger.info("fragment_queue is empty, exit.")
                break
            
            # RANSAC fit to determine the fit coef if it's a good track, otherwise reject
            if len(raw_fgmt["filter"]) == 0:
                # print('remove ',fgmt)
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
                # dbw.write_one_trajectory(thread=True, fragment_ids = [ObjectId(o) for o in path[::-1]])
                m.clean_graph(path)
                if len(path)>1:
                    stitcher_logger.info("** stitched {} fragments".format(len(path)),extra = None)
             
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
    # stitcher_logger.info("Final count in stitched collection {}: {}".format(dbw.collection_name, dbw.count()))
    # del dbw
    stitcher_logger.info("DBWriter closed. Exit.")
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

    
    import json
    with open("config/parameters.json") as f:
        parameters = json.load(f)
    parameters["raw_trajectory_queue_get_timeout"] = 0.1

    raw_collection = "pristine_stork--RAW_GT1"
    rec_collection = "pristine_stork--RAW_GT1__initiates"
    
    dbc = DBClient(**parameters["db_param"])
    raw = dbc.client["trajectories"][raw_collection]
    rec = dbc.client["reconciled"][rec_collection]
    # gt = DBClient(**parameters["db_param"], database_name = trajectory_database, collection_name="groundtruth_scene_1")
    

    fragment_queue = queue.Queue()
    f_ids = [ObjectId('62e403fa1b6a12ef2b2ae143'),ObjectId('62e404221b6a12ef2b2ae15b'),ObjectId('62e4045d1b6a12ef2b2ae185'),ObjectId('62e4047b1b6a12ef2b2ae197'), ObjectId('62e404951b6a12ef2b2ae1a6'),ObjectId('62e404ae1b6a12ef2b2ae1b9')]
    # f_ids = [ ObjectId('62e0193027b64c6330546003'), ObjectId('62e0194627b64c6330546016'), ObjectId('62e0195327b64c6330546026'),  ObjectId('62e0196227b64c6330546035')]
    # f_ids = [ ObjectId('62e0198927b64c6330546059'), ObjectId('62e0199527b64c6330546068'), ObjectId('62e019a827b64c633054607d'),  ObjectId('62e019be27b64c6330546096')]
    
    for f_id in f_ids:
        f = raw.find_one({"_id": f_id})
        fragment_queue.put(f)
    s1 = fragment_queue.qsize()

    

    # --------- start online stitching --------- 
    # fragment_queue,actual_gt_ids,_ = read_to_queue(gt_ids=gt_ids, gt_val=gt_val, lt_val=lt_val, parameters=parameters)
    stitched_trajectory_queue = queue.Queue()
    t1 = time.time()
    min_cost_flow_online_alt_path("west", fragment_queue, stitched_trajectory_queue, parameters)
    online = list(stitched_trajectory_queue.queue)
    s2 = stitched_trajectory_queue.qsize()
    t2 = time.time()
    print("{} fragment stitched to {} trajectories, taking {:.2f} sec".format(s1, s2, t2-t1))
    

    
    
    