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
from live_data_feed import live_data_reader # change to live_data_read later
from config import parameters
from i24_logger.log_writer import logger
from stitcher import stitch_raw_trajectory_fragments
import reconciliation


if __name__ == '__main__':
    print("Post-processing manager starting up.")
    manager_PID = os.getpid()
    print("Post-processing manager has PID={}".format(manager_PID))
    mp_manager = mp.Manager()

    # SHARED DATA STRUCTURES
    # ----------------------------------
    # ----------------------------------
    print("Post-processing manager creating shared data structures")
    # Raw trajectory fragment queue
    # -- populated by database connector that listens for updates
    # TODO: specify the format of raw data as it will be stored in the queue (JSON, dict, etc?)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    raw_fragment_queue_e = mp_manager.Queue(maxsize=parameters.RAW_TRAJECTORY_QUEUE_SIZE) # east direction
    raw_fragment_queue_w = mp_manager.Queue(maxsize=parameters.RAW_TRAJECTORY_QUEUE_SIZE) # west direction
    # Stitched trajectory queue
    # -- populated by stitcher and consumed by reconciliation pool
    # TODO: specify the format of raw data as it will be stored in the queue
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    stitched_trajectory_queue = mp_manager.Queue(maxsize=parameters.STITCHED_TRAJECTORY_QUEUE_SIZE)

    # Log message queue
    # -- populated by all processes and consumed by log handler
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # log_message_queue = mp_manager.Queue(maxsize=parameters.LOG_MESSAGE_QUEUE_SIZE)
    # manager_logger = I24Logger(owner_process_name = "postprocessing_manager",
    #                             connect_console=True,
    #                             connect_file = True)
    

    # manager_logger = i24_logger.log_writer.logger
    
    # PID tracker is a single dictionary of format {processName: PID}
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    pid_tracker = mp_manager.dict()

#%%
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
                          # 'raw_data_feed_e': (raw_data_feed,
                          #                   (db_parameters.DB_NAME, 
                          #                    db_parameters.RAW_COLLECTION, 
                          #                    stitcher_parameters.RANGE_INCREMENT,
                          #                    raw_fragment_queue_e,
                          #                    "east",)),
                          # 'raw_data_feed_w': (raw_data_feed,
                          #                   (db_parameters.DB_NAME, 
                          #                    db_parameters.RAW_COLLECTION, 
                          #                    parameters.RANGE_INCREMENT,
                          #                    raw_fragment_queue_w,
                          #                    "west",)),
                          "live_data_reader_w": (live_data_reader,
                                                 )
                          live_data_reader(parameters.DEFAULT_HOST, parameters.DEFAULT_PORT, parameters.READONLY_USER,   
                                     parameters.DEFAULT_PASSWORD,
                                     parameters.DB_NAME, parameters.RAW_COLLECTION, 
                                     range_increment=parameters.RANGE_INCREMENT, 
                                     direction="west",
                                     raw_fragment_queue_w, 
                                     t_buffer = 0, min_queue_size = 10)
                          
                          # 'stitcher_e': (stitch_raw_trajectory_fragments,
                          #                ("east", raw_fragment_queue_e, stitched_trajectory_queue,)),
                            'stitcher_w': (stitch_raw_trajectory_fragments,
                                           ("west", raw_fragment_queue_w, stitched_trajectory_queue,)),
                            # 'reconciliation': (reconciliation.reconciliation_pool,
                            #                     (stitched_trajectory_queue, pid_tracker,)),
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

    try:
        while True:
            # for each process that is being managed at this level, check if it's still running
            time.sleep(2)
            for child_key in subsystem_process_objects.keys():
                child_process = subsystem_process_objects[child_key]
                if child_process.is_alive():
                    # Process is running; do nothing.
                    # print(child_key)
                    pass
                else:
                    # Process has died. Let's restart it.
                    # Copy its name out of the existing process object for lookup and restart.
                    process_name = child_process.name
                    manager_logger.warning("Restarting process: {}".format(process_name))
                    print("Restarting process: {}".format(process_name))
                    # Get the function handle and function arguments to spawn this process again.
                    process_function, process_args = processes_to_spawn[process_name]
                    # Restart the process the same way we did originally.
                    subsys_process = mp.Process(target=process_function, args=process_args, name=process_name, daemon=False)
                    subsys_process.start()
                    # Re-write the process object in the dictionary and update its PID.
                    subsystem_process_objects[child_key] = subsys_process
                    pid_tracker[process_name] = subsys_process.pid
      
            
    except KeyboardInterrupt:
        # Catch KeyboardInterrupt, which is the same thing as a SIGINT
        # The command `kill -INT [PID]` with the AIDSS_manager PID, executed on the command line, will gracefully
        # shut down the whole AI-DSS with its child processes.
        for pid_name, pid_val in pid_tracker.items():
            os.kill(pid_val, signal.SIGKILL)
            manager_logger.info("Sent SIGKILL to PID={} ({})".format(pid_val, pid_name))