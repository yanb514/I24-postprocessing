# -----------------------------
__file__ = 'pp_lite.py'
__doc__ = """
run only 1 pass, no parallel compute
1 data reader, 1 merge, 1 mcc, for each direction, and 1 reconciliation in total
"""
# -----------------------------

import multiprocessing as mp
import os
import signal
import time
import json
from collections import defaultdict
from i24_logger.log_writer import logger

# Custom modules
import data_feed as df
import min_cost_flow as mcf
import reconciliation as rec
import merge
from transform import main as transform

#%% SIGNAL HANDLING
class SIGINTException(Exception):
    pass

def main(raw_collection = None, reconciled_collection = None):
    
    
    def soft_stop_hdlr(sig, action):
        # send SIGINT to all subprocesses
        manager_logger.info("Manager received SIGINT")
        raise SIGINTException # so to exit the while true loop
        
        
    def finish_hdlr(sig, action):
        # kill data_reader only (from all nodes)
        manager_logger.info("Manager received SIGUSR1")
        for proc_name, proc_info in master_proc_map.items():
            if "feed" in proc_name:
                try:
                    proc_info["keep_alive"] = False
                    os.kill(pid_tracker[proc_name], signal.SIGINT)
                    manager_logger.info("Sent SIGINT to PID={} ({})".format(pid_tracker[proc_name], proc_name))
                except:
                    pass
        

    # register signals
    signal.signal(signal.SIGINT, soft_stop_hdlr)
    signal.signal(signal.SIGUSR1, finish_hdlr)


    # %% Parameters, data structures and processes
    # GET PARAMAETERS
    with open("config/parameters.json") as f:
        parameters = json.load(f)
    
    if raw_collection:
        parameters["raw_collection"] = raw_collection
    
    if reconciled_collection:
        parameters["reconciled_collection"] = reconciled_collection
    
    with open(os.path.join(os.environ["USER_CONFIG_DIRECTORY"], "db_param.json")) as f:
        db_param = json.load(f)
        
    # CHANGE NAME OF THE LOGGER
    manager_logger = logger
    manager_logger.set_name("postproc_manager")
    setattr(manager_logger, "_default_logger_extra",  {})
    HB = parameters["log_heartbeat"]
    
    # CREATE A MANAGER
    mp_manager = mp.Manager()
    manager_logger.info("Post-processing manager has PID={}".format(os.getpid()))

    # SHARED DATA STRUCTURES
    mp_param = mp_manager.dict()
    mp_param.update(parameters)
    mp_param["time_win"] = mp_param["master_time_win"]
    mp_param["stitcher_args"]["stitch_thresh"] = mp_param["stitcher_args"]["master_stitch_thresh"]
    # mp_param["stitcher_mode"] = "master" # switch from local to master
    
    # initialize some db collections
    df.initialize(mp_param, db_param) # read from raw collection
    manager_logger.info("Post-processing manager initialized db collections. Creating shared data structures")
    directions = ["eb", "wb"]
            
    
    # -- master processes (not videonode specific) START AFTER ALL THE LOCAL PROCESSES DIE
    master_queues_map = {} # key:proc_name, val:queue that this process writes to
    for dir in directions:
        master_queues_map[f"master_{dir}_feed"] = mp_manager.Queue() 
        master_queues_map[f"master_{dir}_merge"] = mp_manager.Queue() 
    master_queues_map["master_stitch"] = mp_manager.Queue() 
    master_queues_map["master_reconcile"] = mp_manager.Queue() 
    
    master_proc_map = defaultdict(dict)
    
    for dir in directions:
        
        # feed
        key1 = "master_"+dir+"_feed"
        master_proc_map[key1]["command"] = df.static_data_reader # TODO: modify data_feed
        master_proc_map[key1]["args"] = (mp_param, db_param, master_queues_map[key1], 
                                         {"direction":1 if dir=="eb" else -1},) # qeury from raw collection
        master_proc_map[key1]["predecessor"] = None
        master_proc_map[key1]["dependent_queue"] = None
        
        # merge
        key2 =  "master_"+dir+"_merge"
        master_proc_map[key2]["command"] = merge.merge_fragments 
        master_proc_map[key2]["args"] = (dir, master_queues_map[key1], master_queues_map[key2] , mp_param, key2, ) 
        master_proc_map[key2]["predecessor"] = [key1]
        master_proc_map[key2]["dependent_queue"] = [master_queues_map[key1]]
        
        # stitch
        key3 = "master_"+dir+"_stitch"
        master_proc_map[key3]["command"] = mcf.min_cost_flow_online_alt_path
        master_proc_map[key3]["args"] = (dir, master_queues_map[key2], master_queues_map["master_stitch"], mp_param, key3,)
        master_proc_map[key3]["predecessor"] = [key2]
        master_proc_map[key3]["dependent_queue"] = [master_queues_map[key2]]
    
    
    
    master_proc_map["reconciliation"] = {"command": rec.reconciliation_pool,
                                      "args": (mp_param, db_param, 
                                               master_queues_map["master_stitch"],
                                               master_queues_map["master_reconcile"] ,),
                                      "predecessor": [f"master_{dir}_stitch" for dir in directions],
                                      "dependent_queue": [master_queues_map["master_stitch"]]}
    
    master_proc_map["reconciliation_writer"] = {"command": rec.write_reconciled_to_db,
                                      "args": (mp_param, db_param, master_queues_map["master_reconcile"],),
                                      "predecessor": ["reconciliation"],
                                      "dependent_queue": [master_queues_map["master_reconcile"]]}

    # add PID to PID_tracker
    pid_tracker = {} # mp_manager.dict()
        
    #%% Master loop
    for proc_name, proc_info in master_proc_map.items():        
        subsys_process = mp.Process(target=proc_info["command"], args=proc_info["args"], name=proc_name, daemon=False)
        subsys_process.start()
        pid_tracker[proc_name] = subsys_process.pid
        master_proc_map[proc_name]["process"] = subsys_process # Process object cannot be pickled, thus local_proc_map cannot be a mp_manager.dict()

        
    start = time.time()
    begin = start
    while True:
        try:
            now = time.time()
            if now - begin > 14400 and all([q.empty() for _,q in master_queues_map.items()]): # 4hr
                manager_logger.info("Master processes exceed running for 4hr and all queues are empty.")
                break
            
            if now - begin > 20 and all([q.empty() for _,q in master_queues_map.items()]) and \
            not any([master_proc_map[proc]["process"].is_alive() for proc in master_proc_map]):
                manager_logger.info("Master processes complete in {} sec.".format(now-begin))
                break
                
    
            for proc_name, proc_info in master_proc_map.items():

                if not proc_info["process"].is_alive():
                    pred_alive = [False] if not proc_info["predecessor"] else \
                    [master_proc_map[pred]["process"].is_alive() for pred in proc_info["predecessor"]]
                    queue_empty = [True] if not proc_info["dependent_queue"] else \
                    [q.empty() for q in proc_info["dependent_queue"]]
                    
                    if not any(pred_alive) and all(queue_empty): # natural death
                        proc_info["keep_alive"] = False
#                         manager_logger.info("{} died naturally".format(proc_name))
                    else:
                        # resurrect this process
                        manager_logger.warning(f" Resurrect {proc_name}")
                        subsys_process = mp.Process(target=proc_info["command"], 
                                                    args=proc_info["args"], 
                                                    name=proc_name, daemon=False)
                        subsys_process.start()
                        pid_tracker[proc_name] = subsys_process.pid
                        master_proc_map[proc_name]["process"] = subsys_process 
                
            # Heartbeat queue sizes
            now = time.time()
            if now - start > HB:
                for proc_name, q in master_queues_map.items():
                    if not q.empty():
                        manager_logger.info("Queue size for {}: {}".format(proc_name, 
                                                                           master_queues_map[proc_name].qsize()))
                manager_logger.info("Master processes have been running for {} sec".format(now-begin))
                start = time.time()
                
        except SIGINTException: 
            manager_logger.info("Postprocessing interrupted by SIGINT.")
            for proc_name in master_proc_map:
                try:
                    os.kill(pid_tracker[proc_name], signal.SIGKILL)
                    manager_logger.info("Sent SIGKILL to PID={} ({})".format(pid_tracker[proc_name], proc_name))
                    time.sleep(0.5)
                    
                except:
                    pass
            for q_name, q in master_queues_map.items():
                if not q.empty():
                    manager_logger.info("Queue size after process {}: {}".format(q_name, q.qsize()))
            break # break the while true loop
    
    manager_logger.info("MASTER Postprocessing Mischief Managed.")
    
    #%% start transform on postproc data
    if parameters["transform_postproc"]:
        transform(database_name = parameters["reconciled_database"], 
                  collection_name = parameters["reconciled_collection"])
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    