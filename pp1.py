# -----------------------------
__file__ = 'pp1.py'
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


#%% SIGNAL HANDLING

class SIGINTException(Exception):
    pass

def main(raw_collection = None, reconciled_collection = None, node=None):
    
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
    
    # Raw trajectory fragment queue
    # -- populated by data_reader and consumed by merger
    raw_fragment_queue_e = mp_manager.Queue(maxsize=parameters["raw_queue_size"]) # east direction
    raw_fragment_queue_w = mp_manager.Queue(maxsize=parameters["raw_queue_size"]) # west direction
    
    # Merged queues
    # -- populated by merger and consumed by stitcher
    merged_queue_e = mp_manager.Queue(maxsize=parameters["merged_queue_size"]) # east direction
    merged_queue_w = mp_manager.Queue(maxsize=parameters["merged_queue_size"]) # west direction
    
    # Stitched trajectory queue
    # -- populated by stitcher and consumed by reconciliation pool
    stitched_trajectory_queue = mp_manager.Queue(maxsize=parameters["stitched_queue_size"]) 
    
    # Reconciled queue
    # -- populated by stitcher and consumed by reconciliation_writer
    reconciled_queue = mp_manager.Queue() # maxsize=parameters["reconciled_trajectory_queue_size"]
    
    # PID tracker is a single dictionary of format {processName: PID}
    pid_tracker = mp_manager.dict()
    

#%% Define processes

    # ASSISTANT/CHILD PROCESSES
    # ----------------------------------
    # References to subsystem processes that will get spawned so that these can be recalled
    # upon any failure. Each list item contains the name of the process, its function handle, and
    # its function arguments for call.
    # ----------------------------------
    # -- raw_data_feed: populates the `raw_fragment_queue` from the database
    # -- stitcher: constantly runs trajectory stitching and puts completed trajectories in `stitched_trajectory_queue`
    # -- reconciliation: creates a pool of reconciliation workers and feeds them from `stitched_trajectory_queue`
    # -- log_handler: watches a queue for log messages and sends them to Elastic
    processes_to_spawn = {}
    
    processes_to_spawn["static_data_reader_e"] = (df.static_data_reader,
                    (mp_param, db_param, raw_fragment_queue_e, "eb", node, 5000, "data_reader_e",))
    
    processes_to_spawn["static_data_reader_w"] = (df.static_data_reader,
                    (mp_param, db_param, raw_fragment_queue_w, "wb", node, 5000, "data_reader_w", ))
    
    processes_to_spawn["merger_e"] = (merge.merge_fragments,
                      ("eb", raw_fragment_queue_e, merged_queue_e, mp_param, ))
    
    processes_to_spawn["merger_w"] = (merge.merge_fragments,
                      ("wb", raw_fragment_queue_w, merged_queue_w, mp_param, ))
    
    
    processes_to_spawn["stitcher_e"] = (mcf.min_cost_flow_online_alt_path,
                      ("eb", merged_queue_e, stitched_trajectory_queue, mp_param, ))
    
    processes_to_spawn["stitcher_w"] = (mcf.min_cost_flow_online_alt_path,
                      ("wb", merged_queue_w, stitched_trajectory_queue, mp_param, ))

   
    processes_to_spawn["reconciliation"] = (rec.reconciliation_pool,
                      (mp_param, db_param, stitched_trajectory_queue, reconciled_queue,))

    processes_to_spawn["reconciliation_writer"] = (rec.write_reconciled_to_db,
                      (mp_param, db_param, reconciled_queue,))
    

    # Specify dependencies amongst subprocesses 
    # -- a process can only die if (all/any?) of its predecessors is not alive
    predecessor = {
        "static_data_reader_e": None, # no dependent
        "static_data_reader_w": None, # no dependent
        "merger_e": ["static_data_reader_e"],
        "merger_w": ["static_data_reader_w"],
        "stitcher_e": ["merger_e"],
        "stitcher_w": ["merger_w"],
        "reconciliation": ["stitcher_e", "stitcher_w"], # and
        "reconciliation_writer": ["reconciliation"],
        }
    
    # corresponding queue has to be empty for the process to safety die
    dependent_queues = {
        "static_data_reader_e": None, # no dependent
        "static_data_reader_w": None, # no dependent
        "merger_e": raw_fragment_queue_e,
        "merger_w": raw_fragment_queue_w,
        "stitcher_e": merged_queue_e, 
        "stitcher_w": merged_queue_w,
        "reconciliation": stitched_trajectory_queue, #if parameters["eval"] else stitched_trajectory_queue_copy, # change back if no evaluation
        "reconciliation_writer": reconciled_queue,
        }

    # Store subprocesses and their PIDs, but don't start them just yet
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
    # for i in range(9):
    #     node = "videonode"+str(int(i+1))
    #     print(node)
    main(node="videonode2")
    
    
    
    
    
    