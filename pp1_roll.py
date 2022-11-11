# -----------------------------
__file__ = 'pp1_roll.py'
__doc__ = """
2 pass
1. parallel run merge and stitch on each compute node, save result in local queues
2. stitch consecutive 2-node
3. reconcile whenever a traj is ready from step (2)

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
    HB = parameters["log_heartbeat"]
    
    # CREATE A MANAGER
    mp_manager = mp.Manager()
    manager_logger.info("Post-processing manager has PID={}".format(os.getpid()))
    
    # initialize some db collections
    mp_param = mp_manager.dict()
    mp_param.update(parameters)
    df.initialize(mp_param, db_param)
    
    # SHARED DATA STRUCTURES
 
    manager_logger.info("Post-processing manager initialized db collections. Creating shared data structures")
    
    # initialize queues and processes
    # -- populated by data_reader and consumed by merger
    queue_map = {} #{"stitched_queue":stitched_queue, "reconciled_queue": reconciled_queue} # master object -> [1/.../9] -> [EB/WB] -> [raw_q/merged_q/stitched_q]
    
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
                    
                queue = mp_manager.Queue()  #maxsize=q_sizes[i]
                queue_map[key] = queue
                proc_map[key] = {"keep_alive": True}
                
                if "feed" in key:
                    proc_map[key]["command"] = df.static_data_reader # TODO: modify data_feed
                    proc_map[key]["args"] = (mp_param, db_param, queue, dir, node, key, )
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
            
    
    # -- 2nd pass processes to stitch adjacent nodes
    master_queue_map = {} # key:proc_name, val:queue that this process writes to
    master_queue_map["stitched"] = mp_manager.Queue() 
    # master_queue_map["reconciled"] = mp_manager.Queue() 
    transition_queues = defaultdict(list) # to connect stitch_12 to stitch_23, for example
    transition_queues["eb"].append(stitched_queues["eb"][0]) # local queue for node1
    transition_queues["wb"].append(stitched_queues["wb"][0]) # local queue for node1
    
    master_proc_map = defaultdict(dict)
    
    for n in range(parameters["num_video_nodes"]-1): # should have num_video_nodes-1 transition processes
        
        for dir in ["eb", "wb"]:
            proc_name = "stitch_"+str(n+1)+str(n+2)+"_"+dir
            qout = mp_manager.Queue() # write result to here for the next stitcher in line
            transition_queues[dir].append(qout) # len = n+1
            qin1 = transition_queues[dir][n]
            qin2 = stitched_queues[dir][n+1]
            
            master_proc_map[proc_name]["command"] = mcf.stitch_rolling
            master_proc_map[proc_name]["args"] = (dir, qin1, qin2, qout, master_queue_map["stitched"], mp_param, n,)
            master_proc_map[proc_name]["predecessor"] = [proc_map["videonode"+str(n+1)+"_"+dir+"_stitch"], # TODO call objects instead of proc_name
                                                         proc_map["videonode"+str(n+2)+"_"+dir+"_stitch"]]
            master_proc_map[proc_name]["dependent_queue"] = [qin1, qin2]
            if n >= 1:# not the first
                master_proc_map[proc_name]["predecessor"].append(master_proc_map["stitch_"+str(n)+str(n+1)+"_"+dir])
            
            if n == parameters["num_video_nodes"]-2: # last
                master_proc_map[proc_name]["args"] = (dir, qin1, qin2, master_queue_map["stitched"], master_queue_map["stitched"], mp_param, n, )
                

    
    # master_proc_map["reconciliation"] = {"command": rec.reconciliation_pool,
    #                                   "args": (mp_param, db_param, master_queue_map["stitched"], master_queue_map["reconciled"],),
    #                                   "predecessor": [master_proc_map[proc] for proc in master_proc_map if "stitch" in proc], # all transition stitchers
    #                                   "dependent_queue": [master_queue_map["stitched"]],
    #                                   "keep_alive": True}
    # master_proc_map["reconciliation_writer"] = {"command": rec.write_reconciled_to_db,
    #                                   "args": (mp_param, db_param, master_queue_map["reconciled"],),
    #                                   "predecessor": [master_proc_map["reconciliation"]],
    #                                   "dependent_queue": [master_queue_map["stitched"]],
    #                                   "keep_alive": True}
    
   
    # add PID to PID_tracker
    pid_tracker = {} #mp_manager.dict()
    for proc_name, proc_info in proc_map.items():
        subsys_process = mp.Process(target=proc_info["command"], args=proc_info["args"], name=proc_name, daemon=False)
        subsys_process.start()
        pid_tracker[proc_name] = subsys_process.pid
        proc_map[proc_name]["process"] = subsys_process # Process object cannot be pickled, thus proc_map cannot be a mp_manager.dict()
        
     

#%% Run local processes until signals received  / all queues are empty
# if SIGINT received: raise SIGINTException error, and break
# if SIGUSR received: should still be in the while true loop try block. keep_alive flag will determine if a process should be kept alive      
    
    start = time.time()
    begin = time.time()
    
    while True:
        try:
            now = time.time()
            all_queue_empty = all([q.empty() for q_name, q in queue_map.items() if "stitch" not in q_name])
            all_stitcher_alive = [proc_info["process"].is_alive() for proc_name, proc_info in proc_map.items() if "stitch" in proc_name]
            
            if now - begin > 20 and all_queue_empty and not any(all_stitcher_alive): # all data are in stitched queues
                manager_logger.info("Local processes are completed in {} sec. Proceed to master proceses.".format(now-begin))
                break
                
                
            for proc_name, proc_info in proc_map.items():

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
                # print([q.qsize() for _,q in queue_map.items()])
                start = time.time()
           
                
        except SIGINTException:
            manager_logger.info("Postprocessing interrupted by SIGINT.")
            
            for proc_name in proc_map:
                try:
                    # proc_info["keep_alive"] = False
                    # print(proc_name, pid_tracker[proc_name])
                    os.kill(pid_tracker[proc_name], signal.SIGKILL) # TODO: send SIGINT does not kill stitcher!!! why???
                    manager_logger.info("Sent SIGKILL to PID={} ({})".format(pid_tracker[proc_name], proc_name))
                    time.sleep(0.1)
                except:
                    pass
                
            # THIS BLOCK q.qsize has EOF error
            for q_name, q in queue_map.items():
                if not q.empty():
                    print(type(q.Queue()))
                    manager_logger.info("Queue size after process {}: {}".format(q_name, q.qsize()))   
            break # break the while true loop
            
        # except Exception as e:
        #     manager_logger.error("Other exceptions occured. Exit. Exception:{}".format(e))
        #     break
    
        
    #%% Master loop
    
    # add PID to PID_tracker
    time.sleep(3)
    print(mp_param["transition_last_timestamp_eb"])
    print(mp_param["transition_last_timestamp_wb"])
    
    for n in range(parameters["num_video_nodes"]-1): # should have num_video_nodes-1 transition processes  
        for dir in ["eb", "wb"]:
            proc_name = "stitch_"+str(n+1)+str(n+2)+"_"+dir
            proc_info = master_proc_map[proc_name]
            subsys_process = mp.Process(target=proc_info["command"], args=proc_info["args"], name=proc_name, daemon=False)
            subsys_process.start()
            pid_tracker[proc_name] = subsys_process.pid
            master_proc_map[proc_name]["process"] = subsys_process # Process object cannot be pickled, thus proc_map cannot be a mp_manager.dict()
        
        time.sleep(mp_param["delay"]) # add some delay to start the next node
        
        
    start = time.time()
    begin = start
    
    while True:
        try:
            now = time.time()
            if now - begin > 20 and all([q.empty() for _,q in master_queue_map.items()]) and not any([master_proc_map[proc]["process"].is_alive() for proc in master_proc_map]):
                manager_logger.info("Master processes complete in {} sec.".format(now-begin))
                break
                
            for proc_name, proc_info in master_proc_map.items():

                if not proc_info["process"].is_alive():
                    pred_alive = [False] if not proc_info["predecessor"] else [pred["process"].is_alive() for pred in proc_info["predecessor"]]
                    queue_empty = [True] if not proc_info["dependent_queue"] else [q.empty() for q in proc_info["dependent_queue"]]
                    
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
            if now - start > HB:
                manager_logger.info("transition_last_timestamp_eb: {}".format(mp_param["transition_last_timestamp_eb"]))
                manager_logger.info("transition_last_timestamp_wb: {}".format(mp_param["transition_last_timestamp_wb"]))
                
                manager_logger.info("Transition queues EB: {}".format([q.qsize() for q in transition_queues["eb"]]))
                manager_logger.info("Transition queues WB: {}".format([q.qsize() for q in transition_queues["wb"]]))
                
                for proc_name, q in master_queue_map.items():
                    manager_logger.info("Queue size for {}: {}".format(proc_name, master_queue_map[proc_name].qsize()))
                
                manager_logger.info("Master processes have been running for {} sec".format(now-begin))
                start = time.time()
                
        except SIGINTException: 
            manager_logger.info("Postprocessing interrupted by SIGINT.")
            for proc_name in master_proc_map:
                try:
                    # proc_info["keep_alive"] = False
                    os.kill(pid_tracker[proc_name], signal.SIGKILL)
                    manager_logger.info("Sent SIGKILL to PID={} ({})".format(pid_tracker[proc_name], proc_name))
                    time.sleep(0.5)
                    
                except:
                    pass
            
            manager_logger.info("Transition queues EB: {}".format([q.qsize() for q in transition_queues["eb"]]))
            manager_logger.info("Transition queues WB: {}".format([q.qsize() for q in transition_queues["wb"]]))
            for proc_name, q in master_queue_map.items():
                manager_logger.info("Queue size for {}: {}".format(proc_name, master_queue_map[proc_name].qsize()))
                
            break # break the while true loop
            

    
    
    #%%
    manager_logger.info("Postprocessing Mischief Managed.")
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    