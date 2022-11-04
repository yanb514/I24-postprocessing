# -----------------------------
__file__ = 'pp1_all_nodes.py'
__doc__ = """
first pipeline: run postproc in trajectory-indexed documents, do not transform
Parallelize on videonodes

TODOs
1. try to keep all processes alive except for SIGINT
2. signal handling for finish_processing. if soft_kill=True, then 
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



#%% SIGNAL HANDLING

class SIGINTException(Exception):
    pass


def main(raw_collection = None, reconciled_collection = None):
    
    def soft_stop_hdlr(sig, action):
        # send SIGINT to all subprocesses
        
        manager_logger.info("Manager received SIGINT")
        for proc_name, proc_info in proc_map.items():
            try:
                proc_info["keep_alive"] = False
                os.kill(proc_info["pid"], signal.SIGINT)
            except:
                pass
            manager_logger.info("Sent SIGINT to PID={} ({})".format(pid_tracker[proc_name], proc_name))
            
        raise SIGINTException # so to exit the while true loop
        
        
    def finish_hdlr(sig, action):
        # kill data_reader only (from all nodes)
        manager_logger.info("Manager received SIGUSR1")
        for proc_name, proc_info in proc_map.items():
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
    
    # CREATE A MANAGER
    mp_manager = mp.Manager()
    manager_logger.info("Post-processing manager has PID={}".format(os.getpid()))

    # SHARED DATA STRUCTURES
    mp_param = mp_manager.dict()
    mp_param.update(parameters)
    
    # initialize some db collections
    df.initialize_db(mp_param, db_param)
    manager_logger.info("Post-processing manager initialized db collections. Creating shared data structures")
    
    # initialize queues and processes
    # -- populated by data_reader and consumed by merger
    queue_map = {} #{"stitched_queue":stitched_queue, "reconciled_queue": reconciled_queue} # master object -> [1/.../9] -> [EB/WB] -> [raw_q/merged_q/stitched_q]
    q_sizes = [parameters["raw_queue_size"], parameters["merged_queue_size"], parameters["stitched_queue_size"]]
    
    proc_per_node = ["feed", "merge", "stitch"]
    raw_queues = defaultdict(list)
    merged_queues = defaultdict(list)
    stitched_queues = defaultdict(list)
    
    proc_map = {} #mp_manager.dict() #
    
    # -- local processes (run on each videonode)
    for n in range(parameters["num_video_nodes"]):
        node = "videonode"+str(int(n+1))
        for dir in ["eb", "wb"]:
            for i, proc in enumerate(proc_per_node):
                key = str(node)+"_"+dir+"_"+proc
                if i >=1:
                    prev_key = str(node)+"_"+dir+"_"+proc_per_node[i-1]
                    prev_queue = queue_map[prev_key]
                else:
                    prev_key,prev_queue = None,None
                    
                queue = mp_manager.Queue(maxsize=q_sizes[i]) 
                queue_map[key] = queue
                proc_map[key] = {"keep_alive": True}
                
                if "feed" in key:
                    proc_map[key]["command"] = df.static_data_reader # TODO: modify data_feed
                    proc_map[key]["args"] = (mp_param, db_param, queue, dir, node, 5000, key, )
                    proc_map[key]["predecessor"] = None # should be None
                    proc_map[key]["dependent_queue"] = None
                    raw_queues[dir].append(queue)
                    
                elif "merge" in proc:
                    proc_map[key]["command"] = merge.merge_fragments 
                    proc_map[key]["args"] = (dir, prev_queue, queue, mp_param, key, ) 
                    proc_map[key]["predecessor"] = [prev_key]
                    proc_map[key]["dependent_queue"] = [queue_map[prev_key]]
                    merged_queues[dir].append(queue)
                    
                elif "stitch" in proc:
                    proc_map[key]["command"] = mcf.min_cost_flow_online_alt_path
                    proc_map[key]["args"] = (dir, prev_queue, queue, mp_param, key,)
                    proc_map[key]["predecessor"] = [prev_key]
                    proc_map[key]["dependent_queue"] = [queue_map[prev_key]]
                    stitched_queues[dir].append(queue)
            
    
    # -- master processes (not videonode specific)
    master_queues = defaultdict(list) # {"eb": [raw, merged, stitched], "wb": [raw, merged, stitched]}
    master_queues_map = {} # key:proc_name, val:queue
    master_queues_map["master_eb_apprentice"] = mp_manager.Queue() 
    master_queues_map["master_wb_apprentice"] = mp_manager.Queue() 
    master_queues_map["master_eb_merge"] = mp_manager.Queue() 
    master_queues_map["master_wb_merge"] = mp_manager.Queue() 
    master_queues_map["master_stitch"] = mp_manager.Queue() 
    master_queues_map["master_reconcile"] = mp_manager.Queue() 
    
    master_procs = ["apprentice", "merge", "stitch"]
    master_proc_map = {}
    
    for dir in ["eb", "wb"]:
        
        # apprentice
        key1 = "master_"+dir+"_apprentice"
        master_proc_map[key1]["command"] = rec.apprentice,
        master_proc_map[key1]["args"] = (dir, stitched_queues[dir], master_queues_map[key1], mp_param, key1,)
        master_proc_map[key1]["predecessor"] = [proc_map[name] for name in proc_map if dir+"_stitch" in name]
        master_proc_map[key1]["dependent_queue"] = stitched_queues[dir]   
        
        # merge
        key2 = "master_"+dir+"_merge"
        master_proc_map[key2]["command"] = merge.merge_fragments 
        master_proc_map[key2]["args"] = (dir, master_queues_map[key1], master_queues_map[key2], mp_param, key2, ) 
        master_proc_map[key2]["predecessor"] = [key1]
        master_proc_map[key2]["dependent_queue"] = [master_queues_map[key1]]  
        
        # stitch
        key3 = "master_"+dir+"_stitch"
        master_proc_map[key3]["command"] = mcf.min_cost_flow_online_alt_path
        master_proc_map[key3]["args"] = (dir, master_queues_map[key2], master_queues_map["master_stitch"], mp_param, key3,)
        master_proc_map[key3]["predecessor"] = ["master_eb_merge", "master_wb_merge"]
        master_proc_map[key3]["dependent_queue"] = [master_queues_map[key2]]
    
    
    
    master_proc_map["reconciliation"] = {"command": rec.reconciliation_pool,
                                      "args": (mp_param, db_param, master_queues_map["master_stitch"], master_queues_map["master_reconcile"] ,),
                                      "predecessor": ["master_eb_stitch", "master_wb_stitch"],
                                      "dependent_queue": [master_queues_map["master_stitch"]],
                                      "keep_alive": True}
    master_proc_map["reconciliation_writer"] = {"command": rec.write_reconciled_to_db,
                                      "args": (mp_param, db_param, master_queues_map["master_reconcile"],),
                                      "predecessor": ["master_reconcile"],
                                      "dependent_queue": [master_queues_map["master_reconcile"]],
                                      "keep_alive": True}
    
   
    # add PID to PID_tracker
    pid_tracker = mp_manager.dict()
    for proc_name, proc_info in proc_map.items():
        subsys_process = mp.Process(target=proc_info["command"], args=proc_info["args"], name=proc_name, daemon=False)
        subsys_process.start()
        pid_tracker[proc_name] = subsys_process.pid
        proc_map[proc_name]["process"] = subsys_process # Process object cannot be pickled, thus proc_map cannot be a mp_manager.dict()


    # make dicts mp.dict()
    # queue_map = mp_manager.dict()
    # queue_map.update(queue_map_dict)
    # proc_map = mp_manager.dict()
    # proc_map.update(proc_map_dict)

     

#%% Run local processes until signals received  / all queues are empty
# if SIGINT received: raise SIGINTException error, and break
# if SIGUSR received: should still be in the while true loop try block. keep_alive flag will determine if a process should be kept alive      
    
    start = time.time()
    while True:
        try:
            now = time.time()
            
            if now - start > 20 and all([q.empty() for _,q in queue_map.items()]) :
                manager_logger.info("Local processes are complete. Proceed to master proceses")
                break
                
                
            for proc_name, proc_info in master_proc_map.items():

                if not proc_info["process"].is_alive():
                    pred_alive = [False] if not proc_map[proc_name]["predecessor"] else [proc_map[pred]["process"].is_alive() for pred in proc_map[proc_name]["predecessor"]]
                    queue_empty = [True] if not proc_map[proc_name]["dependent_queue"] else [q.empty() for q in proc_map[proc_name]["dependent_queue"]]
                    
                    if not any(pred_alive) and all(queue_empty): # natural death
                        proc_info["keep_alive"] = False
                    else:
                        # resurrect this process
                        manager_logger.info(f" Resurrect {proc_name}")
                        subsys_process = mp.Process(target=proc_info["command"], args=proc_info["args"], name=proc_name, daemon=False)
                        subsys_process.start()
                        pid_tracker[proc_name] = subsys_process.pid
                        proc_map[proc_name]["process"] = subsys_process 
                
            # Heartbeat queue sizes
            now = time.time()
            if now - start > 10:
                for dir, q_list in raw_queues.items():
                    manager_logger.info("RAW QUEUES {}: {}".format(dir, [q.qsize() for q in q_list]) )
                for dir, q_list in merged_queues.items():
                    manager_logger.info("MERGED QUEUES {}: {}".format(dir, [q.qsize() for q in q_list]) )
                for dir, q_list in stitched_queues.items():
                    manager_logger.info("STITCHED QUEUES {}: {}".format(dir, [q.qsize() for q in q_list]) )
                start = time.time()
           
                
        except SIGINTException:
            manager_logger.info("Postprocessing interrupted by SIGINT.")
            break # break the while true loop
            
        # except Exception as e:
        #     manager_logger.error("Other exceptions occured. Exit. Exception:{}".format(e))
        #     break
    
        
    #%% Master loop
    
    start = time.time()
    while True:
        try:
            now = time.time()
            
            # if now - start > 20 and all([q.empty() for _,q in master_queues_map.items()]) :
            #     manager_logger.info("Master processes complete")
            #     break
                
                
            for proc_name, proc_info in master_proc_map.items():

                if not proc_info["process"].is_alive():
                    pred_alive = [False] if not master_proc_map[proc_name]["predecessor"] else [master_proc_map[pred]["process"].is_alive() for pred in master_proc_map[proc_name]["predecessor"]]
                    queue_empty = [True] if not proc_map[master_proc_map]["dependent_queue"] else [q.empty() for q in master_proc_map[proc_name]["dependent_queue"]]
                    
                    if not any(pred_alive) and all(queue_empty): # natural death
                        proc_info["keep_alive"] = False
                    else:
                        # resurrect this process
                        manager_logger.info(f" Resurrect {proc_name}")
                        subsys_process = mp.Process(target=proc_info["command"], args=proc_info["args"], name=proc_name, daemon=False)
                        subsys_process.start()
                        pid_tracker[proc_name] = subsys_process.pid
                        master_proc_map[proc_name]["process"] = subsys_process 
                
            # Heartbeat queue sizes
            now = time.time()
            if now - start > 10:
                for proc_name, q in master_queues_map.items():
                    if not q.empty():
                        manager_logger.info("Queue size for {}: {}".format(proc_name, q.qsize()))
                
                start = time.time()
                
        except SIGINTException:
            manager_logger.info("Postprocessing interrupted by SIGINT.")
            break # break the while true loop
            
        # except Exception as e:
        #     manager_logger.error("Other exceptions occured. Exit. Exception:{}".format(e))
        #     break
    
    
    
    
    
    #%%
    manager_logger.info("Postprocessing Mischief Managed.")
    for q_name, q in queue_map.items():
        if q.qsize != 0:
            manager_logger.info("Queue size after process {}: {}".format(q_name, q.qsize()))
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    