# -----------------------------
__file__ = 'postprocess_manager.py'
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

# Cusotmized APIs
from i24_configparse import parse_cfg
from i24_logger.log_writer import logger

config_path = os.path.join(os.getcwd(),"config")
# os.environ["USER_CONFIG_DIRECTORY"] = config_path 
os.environ["user_config_directory"] = config_path
os.environ["my_config_section"] = "TEST"
parameters = parse_cfg("my_config_section", cfg_name = "test_param.config")

# Customized modules
from live_data_feed import live_data_reader # change to live_data_read later
import min_cost_flow as mcf
from reconciliation import reconciliation_pool




    
if __name__ == '__main__':
    
    # CHANGE NAME OF THE LOGGER
    manager_logger = logger
    manager_logger.set_name("postproc_manager")
    # manager_logger.info("name is {}".format(manager_logger._logger.name))
    setattr(manager_logger, "_default_logger_extra",  {})
    
    # CREATE A MANAGER
    mp_manager = mp.Manager()
    manager_logger.info("Post-processing manager has PID={}".format(os.getpid()))

    # SHARED DATA STRUCTURES
    # ----------------------------------
    # ----------------------------------
    print("Post-processing manager creating shared data structures")
    
    
    # Raw trajectory fragment queue
    # -- populated by database connector that listens for updates
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    raw_fragment_queue_e = mp_manager.Queue(maxsize=parameters.raw_trajectory_queue_size) # east direction
    raw_fragment_queue_w = mp_manager.Queue(maxsize=parameters.raw_trajectory_queue_size) # west direction
    
    # Stitched trajectory queue
    # -- populated by stitcher and consumed by reconciliation pool
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    stitched_trajectory_queue = mp_manager.Queue(maxsize=parameters.stitched_trajectory_queue_size)
    
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
    processes_to_spawn = {
                            # "live_data_reader_e": (live_data_reader,
                            #             (parameters.default_host, parameters.default_port, 
                            #             parameters.readonly_user, parameters.default_password,
                            #             parameters.db_name, parameters.raw_collection, 
                            #             parameters.range_increment, "east",
                            #             raw_fragment_queue_e, 
                            #             parameters.buffer_time, parameters.min_queue_size,)),
                            "live_data_reader_w": (live_data_reader,
                                          (parameters, parameters.raw_collection, 
                                          parameters.range_increment, "west",
                                          raw_fragment_queue_w, 
                                          parameters.buffer_time, parameters.min_queue_size,)),
                            # "stitcher_e": (mcf.min_cost_flow_online_alt_path,
                            #                 ("east", raw_fragment_queue_e, stitched_trajectory_queue,
                            #                 parameters, )),
                            # "stitcher_w": (mcf.min_cost_flow_online_alt_path,
                            #                 ("west", raw_fragment_queue_w, stitched_trajectory_queue,
                            #                 parameters, )),
                            # "reconciliation": (reconciliation_pool,
                            #             (stitched_trajectory_queue,)),
                          }

    # Stores the actual mp.Process objects so they can be controlled directly.
    # PIDs are also tracked for now, but not used for management.
    subsystem_process_objects = {}

    # Start all processes for the first time and put references to those processes in `subsystem_process_objects`
    # manager_logger.info("Post-process manager beginning to spawn processes")
    for process_name, (process_function, process_args) in processes_to_spawn.items():
        manager_logger.info("Post-process manager spawning {}".format(process_name))
        # print("Post-process manager spawning {}".format(process_name))
        # Start up each process.
        # Can't make these subsystems daemon processes because they will have their own children; we'll use a
        # different method of cleaning up child processes on exit.
        subsys_process = mp.Process(target=process_function, args=process_args, name=process_name, daemon=False)
        subsys_process.start()
        # Put the process object in the dictionary, keyed by the process name.
        subsystem_process_objects[process_name] = subsys_process
        # Each process is responsible for putting its own children's PIDs in the tracker upon creation (if it spawns).
        pid_tracker[process_name] = subsys_process.pid

    # manager_logger.info("Started all processes.")
    print("Started all processes.")
    # print(pid_tracker)




#%% Handle signals
  
    try:
        while True:
        # while not killer.kill_now:
            # for each process that is being managed at this level, check if it's still running
            time.sleep(2)
            for child_key in subsystem_process_objects.keys():
                child_process = subsystem_process_objects[child_key]
                if child_process.is_alive():
                    # Process is running; do nothing.
                    print("Long live {}! {}".format(child_key, child_process))
                    pass
                else:
                    # Process has died. Let's restart it.
                    # Copy its name out of the existing process object for lookup and restart.
                    process_name = child_process.name
                    manager_logger.warning("Restarting process: {}".format(process_name))
                    print("RIP {} {}".format(process_name, child_process))
                    # Get the function handle and function arguments to spawn this process again.
                    process_function, process_args = processes_to_spawn[process_name]
                    # Restart the process the same way we did originally.
                    subsys_process = mp.Process(target=process_function, args=process_args, name=process_name, daemon=False)
                    subsys_process.start()
                    # Re-write the process object in the dictionary and update its PID.
                    subsystem_process_objects[child_key] = subsys_process
                    pid_tracker[process_name] = subsys_process.pid
          
            
    except KeyboardInterrupt:
        manager_logger.info("keyboard interrupt, qsize: {}".format(raw_fragment_queue_w.qsize()))
    #     # Catch KeyboardInterrupt, which is the same thing as a SIGINT
    #     # The command `kill -INT [PID]` with the manager PID, executed on the command line, will gracefully
    #     # shut down the whole postprocess with its child processes.
    #     # for pid_name, pid_val in pid_tracker.items():
    #     #     os.kill(pid_val, signal.SIGKILL)
    #     #     manager_logger.info("Sent SIGKILL to PID={} ({})".format(pid_val, pid_name))
        pass
    
    
    # manager_logger.info("keyboard interrupt, qsize: {}".format(raw_fragment_queue_w.qsize()))
    
