# -----------------------------
__file__ = 'pp1_all_nodes.py'
__doc__ = """
first pipeline: run postproc in trajectory-indexed documents, do not transform
1. create shared data structures (dictionaries, queues)
2. create subprocesses (reader, merger, stitcher, reconciliation pool)
3. register signals
"""
# -----------------------------

import multiprocessing as mp
import os
import signal
import time
import json
import signal
from i24_logger.log_writer import logger

# Custom modules
import data_feed as df
import min_cost_flow as mcf
import reconciliation as rec
import merge


class SIGINTException(Exception):
    pass


def main(raw_collection = None, reconciled_collection = None):
    #%% SIGNAL HANDLING
    def soft_stop_hdlr(sig, action):
        # send SIGINT to all subprocesses
        
        manager_logger.info("Manager received SIGINT")
        for pid_name, pid_val in pid_tracker.items():
            try:
                os.kill(pid_val, signal.SIGINT)
            except:
                pass
            # time.sleep(2)
            try:
                live_process_objects.pop(pid_name)
            except:
                pass
            manager_logger.info("Sent SIGINT to PID={} ({})".format(pid_val, pid_name))
            
        raise SIGINTException # so to exit the while true loop
        
        
    def finish_hdlr(sig, action):
        # kill data_reader only
        
        manager_logger.info("Manager received SIGUSR1")
        try:
            os.kill(pid_tracker["static_data_reader"], signal.SIGINT)
        except:
            pass
        try:
            live_process_objects.pop("static_data_reader")
        except:
            pass
        manager_logger.info("Sent SIGINT to PID={} (data_reader)".format(pid_tracker["static_data_reader"]))

    # register signals
    signal.signal(signal.SIGINT, soft_stop_hdlr)
    signal.signal(signal.SIGUSR1, finish_hdlr)


    # %%
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
    # -- master queues (not videonode specific)
    stitched_queue = mp_manager.Queue(maxsize=5*parameters["stitched_queue_size"]) 
    reconciled_queue = mp_manager.Queue() # maxsize=parameters["reconciled_trajectory_queue_size"]
    
    # -- populated by data_reader and consumed by merger
    queue_map = mp_manager.dict() # master object -> [1/.../9] -> [EB/WB] -> [raw_q/merged_q/stitched_q]
    q_sizes = [parameters["raw_queue_size"], parameters["merged_queue_size"], parameters["stitched_queue_size"]]
    proc_per_node = ["feed", "merge", "stitch"]
    proc_map = mp_manager.dict()
    
    for node in range(parameters["num_video_nodes"]):
        for dir in ["eb", "wb"]:
            for i, proc in enumerate(proc_per_node):
                key = "node"+str(node)+"_"+dir+"_"+proc
                if i >=1:
                    prev_key = "node"+str(node)+"_"+dir+"_"+proc_per_node[i-1]
                    prev_queue = queue_map[prev_key]
                else:
                    prev_key,prev_queue = None,None
                    
                queue = mp_manager.Queue(maxsize=q_sizes[i]) 
                queue_map[key] = queue
                proc_map[key] = {}
                if proc == "feed":
                    proc_map[key]["command"] = df.static_data_reader # TODO: modify data_feed
                    proc_map[key]["args"] = (node, dir, mp_param, db_param, queue, )
                    proc_map[key]["predecessor"] = None # should be None
                    proc_map[key]["dependent_queue"] = None
                    
                elif proc == "merge":
                    proc_map[key]["command"] = merge.merge_fragments 
                    proc_map[key]["args"] = (dir, prev_queue, queue, mp_param, ) 
                    proc_map[key]["predecessor"] = [prev_key]
                    proc_map[key]["dependent_queue"] = queue_map[prev_key]
                    
                elif proc == "stitch":
                    proc_map[key]["command"] = mcf.min_cost_flow_online_alt_path
                    proc_map[key]["args"] = (dir, prev_queue, queue, mp_param, )
                    proc_map[key]["dependent_queue"] = queue_map[prev_key]
            
    
    # global processes (not videonode specific)
    proc_map["eb_master_stitcher"] = {"command": mcf.min_cost_flow_online_alt_path,
                                      "args": ("eb", prev_queue, queue, mp_param, ),
                                      "predecessor": ,
                                      "dependent_queue": } # TODO: all local stitchers eb
    proc_map["wb_master_stitcher"] = {"command": mcf.min_cost_flow_online_alt_path,
                                      "args": ("wb", prev_queue, queue, mp_param, ),
                                      "predecessor": ,
                                      "dependent_queue": } # TODO: all local wb stitchers
    proc_map["reconciliation"] = {"command": rec.reconciliation_pool,
                                      "args": (mp_param, db_param, stitched_queue, reconciled_queue,),
                                      "predecessor": ["eb_master_stitcher", "wb_master_stitcher"],
                                      "dependent_queue": stitched_queue}
    proc_map["reconciliation_writer"] = {"command": rec.write_reconciled_to_db,
                                      "args": (mp_param, db_param, reconciled_queue,),
                                      "predecessor": ["reconciliation"],
                                      "dependent_queue": reconciled_queue}
    
    # add PID to proc_map
    for proc_name, proc_info in proc_map.items():
        subsys_process = mp.Process(target=proc_info["command"], args=proc_info["args"], name=proc_name, daemon=False)
        subsys_process.start()
        proc_map[proc_name]["pid"] = subsys_process.pid
            


    # Specify dependencies amongst subprocesses 
    # -- a process can only die if (all/any?) of its predecessors is not alive
    predecessor = {
        "static_data_reader": None, # no dependent
        "merger_e": ["static_data_reader"],
        "merger_w": ["static_data_reader"],
        "stitcher_e": ["merger_e"],
        "stitcher_w": ["merger_w"],
        "reconciliation": ["stitcher_e", "stitcher_w"], # and
        "reconciliation_writer": ["reconciliation"],
        }
    
    # corresponding queue has to be empty for the process to safety die
    dependent_queues = {
        "static_data_reader": None, # no dependent
        "merger_e": raw_fragment_queue_e,
        "merger_w": raw_fragment_queue_w,
        "stitcher_e": merged_queue_e, 
        "stitcher_w": merged_queue_w,
        "reconciliation": stitched_trajectory_queue, #if parameters["eval"] else stitched_trajectory_queue_copy, # change back if no evaluation
        "reconciliation_writer": reconciled_queue,
        }

    # Store subprocesses and their PIDs. A process has to start to have a PID
    live_process_objects = {}
    for process_name, (process_function, process_args) in processes_to_spawn.items():
        subsys_process = mp.Process(target=process_function, args=process_args, name=process_name, daemon=False)
        subsys_process.start()
        live_process_objects[process_name] = subsys_process
        pid_tracker[process_name] = subsys_process.pid




#%% Run indefinitely until signals received        
        
    while True:
        try:
            # for each process that is being managed at this level, check if it's still running
            time.sleep(2)
            # 
            if all([not p.is_alive() for _,p in live_process_objects.items()]) and all([dependent_queues[n].qsize() == 0 for n,_ in live_process_objects.items() if dependent_queues[n] is not None]):
                manager_logger.info("None of the processes is alive and dependent queues are all empty")
                break
            
            for pid_name, pid_val in pid_tracker.items():
                try:
                    child_process = live_process_objects[pid_name]
                    # print("child_process: ", child_process.name, child_process.is_alive())
                except:
                    continue
    
                # print(child_process.name, child_process.is_alive())
                if not child_process.is_alive():
                    try:
                        live_process_objects.pop(pid_name)
                        print("RIP {}, you will be missed".format(pid_name))
                    except:
                        pass
                    
                    # DEFAULT: restart processes ONLY if death is unnatural
                    process_name = child_process.name
                    preds = predecessor[process_name]
                    if preds is None:
                        no_pred_alive = True
                    else:
                        pred_alive = []
                        for p in preds:
                            try:
                                pred_alive.append(live_process_objects[p].is_alive())
                            except KeyError:
                                pred_alive.append(False)
    
                        # if the death is natural, let it be - no predecessor is alive and dependent queues are all empty
                        no_pred_alive = not any(pred_alive)
                        
                    queue_empty = dependent_queues[process_name] is None or dependent_queues[process_name].empty()
                    if no_pred_alive and queue_empty: 
                        # live_process_objects.pop(process_name)
                        manager_logger.debug("RIP: {} died of natural causes.".format(pid_name))
       
                    else: # if unatural death, restart
                        if not no_pred_alive:
                            manager_logger.warning("{} is resurrected by the Lord of Light because its predecessor is still alive".format(process_name))
                        elif not queue_empty: 
                            manager_logger.warning("{} is resurrected by the Lord of Light because queue is not empty".format(process_name))
                        
                        process_function, process_args = processes_to_spawn[process_name]
                        subsys_process = mp.Process(target=process_function, args=process_args, name=process_name, daemon=False)
                        subsys_process.start()
                        live_process_objects[pid_name] = subsys_process
                        pid_tracker[process_name] = subsys_process.pid
                        
                        
                else:
                    # Process is running; do nothing.
                    # if pid_name in live_process_objects:
                    # print("Long live {}! {}".format(pid_name, child_process))
                    time.sleep(2)
                    manager_logger.info("Current queue sizes, raw_e: {}, raw_w: {}, merge_e: {}, merge_w:{}, stitched: {}, reconciled: {}".format(raw_fragment_queue_e.qsize(), raw_fragment_queue_w.qsize(), 
                                                                                                                                                  merged_queue_e.qsize(), merged_queue_w.qsize(), 
                                                                                                                                                  stitched_trajectory_queue.qsize(), 
                                                                                                                                                  reconciled_queue.qsize()))
                    pass
                
        except SIGINTException:
            manager_logger.info("Postprocessing interrupted by SIGINT.")
            break # break the while true loop
            
        except Exception as e:
            manager_logger.warning("Other exceptions occured. Exit. Exception:{}".format(e))
            break
    
        
    manager_logger.info("Postprocessing Mischief Managed.")
    manager_logger.info("Final queue sizes, raw_e: {}, raw_w: {}, merge_e: {}, merge_w:{}, stitched: {}, reconciled: {}".format(raw_fragment_queue_e.qsize(), raw_fragment_queue_w.qsize(), 
                                                                                                                                  merged_queue_e.qsize(), merged_queue_w.qsize(), 
                                                                                                                                  stitched_trajectory_queue.qsize(), 
                                                                                                                                  reconciled_queue.qsize()))
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    