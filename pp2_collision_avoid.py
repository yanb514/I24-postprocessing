# -----------------------------
__file__ = 'pp2.py'
__doc__ = """
I-24 MOTION processing software.
Top level process for live post-processing
Spawns and manages child processes for trajectory fragment stitching and trajectory reconciliation.
"""
# -----------------------------

import multiprocessing as mp
import os
import signal
import time
import json
from i24_logger.log_writer import logger

# Custom modules
import data_feed as df
import min_cost_flow as mcf
import merge

# import _evaluation.evaluation as ev
import multi_opt as mo



def main(collection_name = None):
    # GET PARAMAETERS
    with open("config/parameters.json") as f:
        parameters = json.load(f)
    
    if collection_name is not None:
        parameters["raw_collection"] = collection_name
    
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
    # ----------------------------------
    # ----------------------------------
    mp_param = mp_manager.dict()
    mp_param.update(parameters)
    
    # initialize some db collections
    df.initialize_db(mp_param, db_param)
    manager_logger.info("Post-processing manager initialized db collections. Creating shared data structures")
    
    # DELETE ME LATER
    # if parameters["eval"]:
    # ev.eval_raw(mp_param, db_param)
    
    # Raw trajectory fragment queue
    # -- populated by database connector that listens for updates
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    raw_fragment_queue_e = mp_manager.Queue(maxsize=parameters["raw_trajectory_queue_size"]) # east direction
    raw_fragment_queue_w = mp_manager.Queue(maxsize=parameters["raw_trajectory_queue_size"]) # west direction
    
    merged_queue_e = mp_manager.Queue(maxsize=parameters["raw_trajectory_queue_size"]) # east direction
    merged_queue_w = mp_manager.Queue(maxsize=parameters["raw_trajectory_queue_size"]) # west direction
    
    # Stitched trajectory queue
    # -- populated by stitcher and consumed by reconciliation pool
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    stitched_trajectory_queue_e = mp_manager.Queue(maxsize=parameters["stitched_trajectory_queue_size"]) 
    stitched_trajectory_queue_w = mp_manager.Queue(maxsize=parameters["stitched_trajectory_queue_size"]) 
    
    reconciled_queue = mp_manager.Queue() # maxsize=parameters["reconciled_trajectory_queue_size"]
    
    # PID tracker is a single dictionary of format {processName: PID}
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
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


    
    processes_to_spawn["static_data_reader"] = (df.static_data_reader,
                    (mp_param, db_param, raw_fragment_queue_e, raw_fragment_queue_w, 1000,))
    
    processes_to_spawn["merger_e"] = (merge.merge_fragments,
                      ("eb", raw_fragment_queue_e, merged_queue_e, mp_param, ))
    
    processes_to_spawn["merger_w"] = (merge.merge_fragments,
                      ("wb", raw_fragment_queue_w, merged_queue_w, mp_param, ))
    
    processes_to_spawn["stitcher_e"] = (mcf.min_cost_flow_online_alt_path,
                      ("eb", merged_queue_e, stitched_trajectory_queue_e, mp_param, ))
    
    processes_to_spawn["stitcher_w"] = (mcf.min_cost_flow_online_alt_path,
                      ("wb", merged_queue_w, stitched_trajectory_queue_w, mp_param, ))
    
    
    # processes_to_spawn["stitcher_e"] = (mcf.dummy_stitcher,
    #                   ("eb", raw_fragment_queue_e, stitched_trajectory_queue, mp_param, ))
    
    # processes_to_spawn["stitcher_w"] = (mcf.dummy_stitcher,
    #                   ("wb", raw_fragment_queue_w, stitched_trajectory_queue, mp_param, ))
    
   
    # processes_to_spawn["reconciliation"] = (rec.reconciliation_pool,
    #                   (mp_param, db_param, stitched_trajectory_queue, reconciled_queue,))

    # processes_to_spawn["reconciliation_writer"] = (rec.write_reconciled_to_db,
    #                   (mp_param, db_param, reconciled_queue,))
    
    
    
    processes_to_spawn["preproc_reconcile_e"] = (mo.preprocess_reconcile,
                                                  ("eb", stitched_trajectory_queue_e, mp_param, db_param, ))
    
    processes_to_spawn["solve_collision_avoidance_rolling_e"] = (mo.solve_collision_avoidance_rolling,
                                                  ("eb", reconciled_queue, mp_param, db_param, ))  
    
    processes_to_spawn["preproc_reconcile_w"] = (mo.preprocess_reconcile,
                                                  ("wb", stitched_trajectory_queue_w, mp_param, db_param, ))
    
    processes_to_spawn["solve_collision_avoidance_rolling_w"] = (mo.solve_collision_avoidance_rolling,
                                                  ("wb", reconciled_queue, mp_param, db_param, ))  

    processes_to_spawn["postproc_reconcile"] = (mo.postprocess_reconcile,
                                                  (reconciled_queue, mp_param, db_param, ))  
    
    
    
    
    live_process_objects = {}
    if parameters["mode"] == "sequence":
        # a process can only die if (all/any?) of its predecessors is not alive
        predecessor = {
            "static_data_reader": None, # no dependent
            "merger_e": ["static_data_reader"],
            "merger_w": ["static_data_reader"],
            "stitcher_e": ["merger_e"],
            "stitcher_w": ["merger_w"],
            "preproc_reconcile_e": ["stitcher_e"],
            "preproc_reconcile_w": ["stitcher_w"],
            "solve_collision_avoidance_rolling_e": ["preproc_reconcile_e"],
            "solve_collision_avoidance_rolling_w": ["preproc_reconcile_w"],
            "postproc_reconcile": ["solve_collision_avoidance_rolling_e", "solve_collision_avoidance_rolling_w"],
            
            # "reconciliation": ["stitcher_e", "stitcher_w"], # and
            # "reconciliation_writer": ["reconciliation"],
            
            # "_eval_raw": None,
            # "_eval_merge_e": ["merger_e"],
            # "_eval_merge_w": ["merger_w"],
            # "_eval_stitch": ["stitcher_e", "stitcher_w"]
            }
        # corresponding queue has to be empty for the process to safety die
        dependent_queues = {
            "static_data_reader": None, # no dependent
            "merger_e": raw_fragment_queue_e,
            "merger_w": raw_fragment_queue_w,
            "stitcher_e": raw_fragment_queue_e, 
            "stitcher_w": raw_fragment_queue_w,
            "preproc_reconcile_e": stitched_trajectory_queue_e,
            "preproc_reconcile_w": stitched_trajectory_queue_w,
            "solve_collision_avoidance_rolling_e": None,
            "solve_collision_avoidance_rolling_w": None,
            "postproc_reconcile": reconciled_queue,
            

            # "reconciliation": stitched_trajectory_queue, #if parameters["eval"] else stitched_trajectory_queue_copy, # change back if no evaluation
            # "reconciliation_writer": reconciled_queue,
            # "_eval_raw": None,
            # "_eval_merge_e": merged_queue_e,
            # "_eval_merge_w": merged_queue_w,
            # "_eval_stitch": stitched_trajectory_queue
            }

    # Start all processes for the first time and put references to those processes in `live_process_objects`
    # manager_logger.info("Post-process manager beginning to spawn processes")

    for process_name, (process_function, process_args) in processes_to_spawn.items():
        manager_logger.info("Post-process manager spawning {}".format(process_name))
        subsys_process = mp.Process(target=process_function, args=process_args, name=process_name, daemon=False)
        subsys_process.start()
        # Put the process object in the dictionary, keyed by the process name.
        live_process_objects[process_name] = subsys_process
        # Each process is responsible for putting its own children's PIDs in the tracker upon creation (if it spawns).
        pid_tracker[process_name] = subsys_process.pid
        




#%% SIGNAL HANDLING

    # Simulate server control
    def hard_stop_hdlr(sig, action):
        manager_logger.info("Manager received hard stop signal")
        for pid_name, pid_val in pid_tracker.items():
            try:
                os.kill(pid_val, signal.SIGKILL)
            except:
                pass
            time.sleep(2)
            try:
                live_process_objects.pop(pid_name)
            except:
                pass
            manager_logger.info("Sent SIGKILL to PID={} ({})".format(pid_val, pid_name))
       
    def soft_stop_hdlr(sig, action):
        manager_logger.info("Manager received soft stop signal")
        for pid_name, pid_val in pid_tracker.items():
            try:
                os.kill(pid_val, signal.SIGINT)
            except:
                pass
            time.sleep(2)
            try:
                live_process_objects.pop(pid_name)
            except:
                pass
            manager_logger.info("Sent SIGINT to PID={} ({})".format(pid_val, pid_name))
            
    def finish_hdlr(sig, action):
        manager_logger.info("Manager received finish-processing signal")
        for pid_name, pid_val in pid_tracker.items():
            try:
                os.kill(pid_val, signal.SIGUSR1)
            except:
                pass
            time.sleep(2)
            try:
                live_process_objects.pop(pid_name)
            except:
                pass
            manager_logger.info("Sent SIGUSR1 to PID={} ({})".format(pid_val, pid_name))
            
    
    # register signals depending on the mode     
    if parameters["mode"] == "hard_stop":
        signal.signal(signal.SIGINT, hard_stop_hdlr)
        
    elif parameters["mode"] == "soft_stop":
        signal.signal(signal.SIGINT, soft_stop_hdlr)
        
    elif parameters["mode"] == "finish":
        manager_logger.warning("Currently do not support finish-processing. Manually kill live_data_read instead")
        # signal.signal(signal.SIGINT, finish_hdlr)
        
    elif parameters["mode"] == "sequence":
        manager_logger.info("In sequence mode")
    else:
        manager_logger.error("Unrecongnized signal")



#%% Run indefinitely until SIGINT received
    while True:
        
        # for each process that is being managed at this level, check if it's still running
        time.sleep(2)
        if all([not p.is_alive() for _,p in live_process_objects.items()]) and all([q.qsize() == 0 for _,q in dependent_queues.items() if q is not None]):
            manager_logger.info("None of the processes is alive and queues are all empty")
            break
        
        for pid_name, pid_val in pid_tracker.items():
            try:
                child_process = live_process_objects[pid_name]
                # print("child_process: ", child_process.name, child_process.is_alive())
            except:
                continue

            # print(child_process.name, child_process.is_alive())
            if not child_process.is_alive():
                # do not restart if in one of the stopping modes
                if parameters["mode"] in ["hard_stop", "soft_stop", "finish"]:
                    try:
                        live_process_objects.pop(pid_name)
                        print("RIP {}, you will be missed".format(pid_name))
                    except:
                        pass
                    
                elif parameters["mode"] == "sequence":
                    # keep the rest of the sequence live - a process cannot die if its predecessor is still alive
                    
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
                    # Restart the child process
                    process_name = child_process.name

                    if process_name in live_process_objects.keys():
                        manager_logger.warning("Restarting process: {}".format(process_name))
                        
                        # Get the function handle and function arguments to spawn this process again.
                        process_function, process_args = processes_to_spawn[process_name]
                        # Restart the process the same way we did originally.
                        subsys_process = mp.Process(target=process_function, args=process_args, name=process_name, daemon=False)
                        subsys_process.start()
                        # Re-write the process object in the dictionary and update its PID.
                        live_process_objects[pid_name] = subsys_process
                        pid_tracker[process_name] = subsys_process.pid
                    
            else:
                # Process is running; do nothing.
                # if pid_name in live_process_objects:
                # print("Long live {}! {}".format(pid_name, child_process))
                time.sleep(2)
                pass
        
        
    manager_logger.info("Postprocessing Mischief Managed.")
    manager_logger.info("Final queue sizes, raw east: {}, raw west: {}, stitched: {}, reconciled: {}".format(raw_fragment_queue_e.qsize(), raw_fragment_queue_w.qsize(), stitched_trajectory_queue_e.qsize(), reconciled_queue.qsize()))
    
    
    # start evaluation
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    