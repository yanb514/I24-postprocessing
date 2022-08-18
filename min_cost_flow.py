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
                fgmt = fragment_queue.get(timeout = GET_TIMEOUT)
                # stitcher_logger.debug("get fragment id: {}".format(raw_fgmt["_id"]))
                # fgmt = Fragment(raw_fgmt)
                
            except queue.Empty: # queue is empty
                stitcher_logger.info("Stitcher queue is empty. Flushing out remaining trajectories in graph.")
                all_paths = m.get_all_traj()
                
                for path in all_paths:
                    filters = m.get_filters(path)
                    stitched_trajectory_queue.put((path[::-1], filters[::-1]))
                    
                    # stitcher_logger.info("Flushing out final trajectories in graph")
                    stitcher_logger.debug("** Flushing out {} fragments".format(len(path)),extra = None)
                    # dbw.write_one_trajectory(thread=True, fragment_ids = [ObjectId(o) for o in path[::-1]])
                
                # stitcher_logger.info("fragment_queue is empty, exit.")
                break
            
            # fgmt_id = getattr(fgmt, ATTR_NAME)
            fgmt_id = fgmt[ATTR_NAME]
            # RANSAC fit to determine the fit coef if it's a good track, otherwise reject
            try:
                if len(fgmt["filter"]) == 0:
                    # stitched_trajectory_queue.put(([fgmt_id], []))
                    stitcher_logger.info("* skip {} - LOW CONF".format(fgmt_id))
                    continue # skip this fgmt
            except:
                pass
                
            m.add_node(fgmt)
            
            # print("* add ", fgmt_id)
            # print("**", m.G.edges(data=True))
            
            # run MCF
            m.augment_path(fgmt_id)
    
            # pop path if a path is ready
            # print("**", m.G.edges(data=True))
            all_paths = m.pop_path(time_thresh = fgmt["first_timestamp"] - TIME_WIN)  
            
            
            for path in all_paths:
                filters = m.get_filters(path)
                if not m.verify_path(path[::-1]):
                    stitcher_logger.info("** stitched result not verified")
                    
                stitched_trajectory_queue.put((path[::-1], filters[::-1]))
                # dbw.write_one_trajectory(thread=True, fragment_ids = [ObjectId(o) for o in path[::-1]])
                m.clean_graph(path)
                stitcher_logger.debug("** stitched {} fragments".format(len(path)),extra = None)
             
            if counter % 100 == 0:
                stitcher_logger.info("Graph nodes : {}, Graph edges: {}, Cache: {}".format(m.G.number_of_nodes(), m.G.number_of_edges(), len(m.cache)),extra = None)
                stitcher_logger.debug(f"raw queue: {fragment_queue.qsize()}, stitched queue: {stitched_trajectory_queue.qsize()}")
                counter = 0
            counter += 1
        
        
        except Exception as e: 
            if sig_hdlr.run:
                stitcher_logger.error("Unexpected exception: {}".format(e))
            else:
                stitcher_logger.warning("SIGINT detected. Exception:{}".format(e))
            break
            
        
    stitcher_logger.info("Exit stitcher")
    # stitcher_logger.info("Final count in stitched collection {}: {}".format(dbw.collection_name, dbw.count()))
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

    
    import json
    with open("config/parameters.json") as f:
        parameters = json.load(f)
    parameters["raw_trajectory_queue_get_timeout"] = 0.1

    raw_collection = "morose_caribou--RAW_GT1" # collection name is the same in both databases
    rec_collection = "morose_caribou--RAW_GT1__medicatess"
    
    dbc = DBClient(**parameters["db_param"])
    raw = dbc.client["trajectories"][raw_collection]
    rec = dbc.client["reconciled"][rec_collection]
    # gt = DBClient(**parameters["db_param"], database_name = trajectory_database, collection_name="groundtruth_scene_1")
    

    fragment_queue = queue.Queue()
    # morose panda
    # f_ids = [ObjectId('62e018c427b64c6330545fa6'),ObjectId('62e0190427b64c6330545fe6'),ObjectId('62e0190427b64c6330545fe7')]
    
    # f_ids = [ObjectId('62e403fe1b6a12ef2b2ae146'),ObjectId('62e404381b6a12ef2b2ae168'),ObjectId('62e404741b6a12ef2b2ae192'),ObjectId('62e4048c1b6a12ef2b2ae1a2')] # stitch to 1
    # f_ids = [ObjectId('62e4032b1b6a12ef2b2ae0cc'), ObjectId('62e4039f1b6a12ef2b2ae108')] # to 1
    # f_ids = [ ObjectId('62e403c41b6a12ef2b2ae126'), ObjectId('62e403e51b6a12ef2b2ae13c'), ObjectId('62e404251b6a12ef2b2ae15c'),  ObjectId('62e404461b6a12ef2b2ae173')] # to 1
    # f_ids = [ObjectId('62e403fa1b6a12ef2b2ae143'), ObjectId('62e404221b6a12ef2b2ae15b'), 
    #           ObjectId('62e4045d1b6a12ef2b2ae185'),ObjectId('62e4047b1b6a12ef2b2ae197'), 
    #           ObjectId('62e404951b6a12ef2b2ae1a6'), ObjectId('62e404ae1b6a12ef2b2ae1b9')]
    # f_ids = [ObjectId('62e403041b6a12ef2b2ae0bd'), ObjectId('62e4031f1b6a12ef2b2ae0c8'), ObjectId('62e403221b6a12ef2b2ae0c9')] # stitch to 2
    # f_ids = [ObjectId('62e403e61b6a12ef2b2ae13d'), ObjectId('62e4041b1b6a12ef2b2ae158'), ObjectId('62e4042f1b6a12ef2b2ae162'),
    #          ObjectId('62e404501b6a12ef2b2ae179'), ObjectId('62e4046e1b6a12ef2b2ae18e')] # to 1


    # initiates
    # f_ids = [ObjectId('62e402d51b6a12ef2b2ae0a0'), ObjectId('62e402df1b6a12ef2b2ae0a5')] # to2
    # f_ids = [ObjectId('62e402ce1b6a12ef2b2ae09a'), ObjectId('62e402e11b6a12ef2b2ae0a6'), ObjectId('62e402f11b6a12ef2b2ae0b0')] # to3
    # f_ids = [ObjectId('62e402251b6a12ef2b2ae036'), ObjectId('62e402631b6a12ef2b2ae058'), ObjectId('62e4026c1b6a12ef2b2ae062'), ObjectId('62e402731b6a12ef2b2ae066'), ObjectId('62e402881b6a12ef2b2ae076')] # to2
    # f_ids = [ObjectId('62e403041b6a12ef2b2ae0bd'), ObjectId('62e4031f1b6a12ef2b2ae0c8'), ObjectId('62e403221b6a12ef2b2ae0c9')]
    
    # juxtaposes
    # f_ids = [ObjectId('62e4037d1b6a12ef2b2ae0f3'), ObjectId('62e403891b6a12ef2b2ae0fa'), ObjectId('62e403961b6a12ef2b2ae102')]#3
    # f_ids = [ObjectId('62e402ef1b6a12ef2b2ae0ae'), ObjectId('62e403041b6a12ef2b2ae0bd'), ObjectId('62e4031f1b6a12ef2b2ae0c8')] # 2
    # f_ids = [ObjectId('62e403011b6a12ef2b2ae0bb'), ObjectId('62e403061b6a12ef2b2ae0c0')] # 2
    # f_ids = [ObjectId('62e402211b6a12ef2b2ae02f'), ObjectId('62e402631b6a12ef2b2ae05a'), ObjectId('62e4026c1b6a12ef2b2ae060')] #2
    # f_ids = [ObjectId('62e4021d1b6a12ef2b2ae02d'), ObjectId('62e402221b6a12ef2b2ae031')] # 2
    # f_ids = [ObjectId('62e4029a1b6a12ef2b2ae080'), ObjectId('62e402d41b6a12ef2b2ae09f')] # 2
    
    # ostentatious_hippo--RAW_GT1__disputes
    # f_ids = [ObjectId('62f6c95fba08cdedcca36fdd'), ObjectId('62f6c969ba08cdedcca36fef')] # 1
    # sweettalks
    # f_ids = [ObjectId('62f6c99cba08cdedcca3704a'), ObjectId('62f6c99dba08cdedcca3704b')] #2
    # f_ids = [ObjectId('62f6c97cba08cdedcca3700f'), ObjectId('62f6c97bba08cdedcca3700e')] # 1
    # f_ids = [ObjectId('62f6c96bba08cdedcca36ff4'), ObjectId('62f6c978ba08cdedcca37009')] #1
    # f_ids = [ObjectId('62f6c943ba08cdedcca36fb6'), ObjectId('62f6c94eba08cdedcca36fc9')] # 1
    # f_ids = [ObjectId('62f6c926ba08cdedcca36f7f'), ObjectId('62f6c933ba08cdedcca36f98'), ObjectId('62f6c939ba08cdedcca36fa0')] # 1
    
    # morous caribou medicates
    f_ids = [ObjectId('62fd0dea46a150340fcd21e0'), ObjectId('62fd0ded46a150340fcd21e7')] #2
    # f_ids = [ObjectId('62fd0db946a150340fcd2181'), ObjectId('62fd0dbb46a150340fcd2185')] #2
    # f_ids = [ObjectId('62fd0daf46a150340fcd2170'), ObjectId('62fd0dc546a150340fcd2198')] #1
    # get parameters for fitting
    RES_THRESH_X = parameters["residual_threshold_x"]
    RES_THRESH_Y = parameters["residual_threshold_y"]
    CONF_THRESH = parameters["conf_threshold"],
    REMAIN_THRESH = parameters["remain_threshold"]
    from data_feed import add_filter
    
    for f_id in f_ids:
        f = raw.find_one({"_id": f_id})
        # print(f_id, "fity ", f["fity"])
        f = add_filter(f, raw.collection, RES_THRESH_X, RES_THRESH_Y, 
                       CONF_THRESH, REMAIN_THRESH)
        # print(f["filter"])
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
    

    
    
    