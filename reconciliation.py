# -----------------------------
__file__ = 'reconciliation.py'
__doc__ = """

Update 6/27/2021
If each worker handles mongodb client connection, then they have to close the connection in order for the pool to join (in the event of graceful shutdown).
It is not recommended by mongodb to open and close connections all the time.
Instead, modify the code such that open the connection for dbreader and writer at the parent process. Each worker does not have direct access to mongodb client.
After done processing, each worker send results back to the parent process using a queue
"""

# -----------------------------
import multiprocessing
from multiprocessing import Pool
import time
import os
import signal
import sys
import queue

import i24_logger.log_writer as log_writer
from i24_database_api import DBClient

from utils.utils_reconciliation import receding_horizon_2d_l1, resample, receding_horizon_2d, combine_fragments, rectify_2d


def reconcile_single_trajectory(reconciliation_args, combined_trajectory, reconciled_queue) -> None:
    """
    Resample and reconcile a single trajectory, and write the result to a queue
    :param next_to_reconcile: a trajectory document
    :return:
    """
    
    rec_worker_logger = log_writer.logger
    rec_worker_logger.set_name("rec_worker")
    setattr(rec_worker_logger, "_default_logger_extra",  {})

    resampled_trajectory = resample(combined_trajectory)
    
    finished_trajectory = rectify_2d(resampled_trajectory, reg = "l1", **reconciliation_args)   
    rec_worker_logger.info("*** Reconciled a trajectory, duration: {:.2f}s, length: {}".format(finished_trajectory["last_timestamp"]-finished_trajectory["first_timestamp"], len(finished_trajectory["timestamp"])), extra = None)

    reconciled_queue.put(finished_trajectory)



def reconciliation_pool(parameters, stitched_trajectory_queue: multiprocessing.Queue, 
                        reconciled_queue: multiprocessing.Queue, ) -> None:
    """
    Start a multiprocessing pool, each worker 
    :param stitched_trajectory_queue: results from stitchers, shared by mp.manager
    :param pid_tracker: a dictionary
    :return:
    """
    # parameters
    reconciliation_args={}
    for key in ["lam2_x","lam2_y","lam1_x","lam1_y", "ph", "ih"]:
        reconciliation_args[key] = parameters[key]
    
    rec_parent_logger = log_writer.logger
    rec_parent_logger.set_name("rec_parent")
    setattr(rec_parent_logger, "_default_logger_extra",  {})

    # wait to get raw collection name
    while parameters["raw_collection"]=="":
        time.sleep(1)
    raw = DBClient(**parameters["db_param"], database_name = parameters["raw_database"], collection_name = parameters["raw_collection"])
   
    # Signal handling: 
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    worker_pool = Pool(processes=parameters["reconciliation_pool_size"])
    signal.signal(signal.SIGINT, original_sigint_handler)
    
    rec_parent_logger.info("** Reconciliation pool starts. Pool size: {}".format(parameters["reconciliation_pool_size"]), extra = None)
    
    # Signal handling: 
    # SIGINT raises KeyboardInterrupt,  close dbw, close pool and exit.
    # SIGUSR1 is ignored. The process terminates when queue is empty. Close and join the pool
    def handler(sigusr, frame):
        rec_parent_logger.warning("SIGUSR1 detected. Finish processing current queues.")
        signal.signal(sigusr, signal.SIG_IGN)    
        signal.signal(signal.SIGPIPE,signal.SIG_DFL) # reset SIGPIPE so that no BrokePipeError when SIGINT is received
        
    signal.signal(signal.SIGUSR1, handler) # ignore SIGUSR1

    
    while True:
        try:
            try:
                next_to_reconcile = stitched_trajectory_queue.get(timeout = parameters["stitched_trajectory_queue_get_timeout"]) #20sec
            except queue.Empty: 
                rec_parent_logger.warning("Getting from stitched trajectories queue is timed out after {}s. Close the reconciliation pool.".format(parameters["stitched_trajectory_queue_get_timeout"]))
                break
        
            # rec_parent_logger.debug("next_to_reconcile: {}".format(next_to_reconcile), extra = None)
            combined_trajectory = combine_fragments(raw.collection, next_to_reconcile)
            # rec_parent_logger.debug("*** 1. Combined stitched fragments.", extra = None)
            
            worker_pool.apply_async(reconcile_single_trajectory, (reconciliation_args, combined_trajectory, reconciled_queue, ))

        except (KeyboardInterrupt, BrokenPipeError): # handle SIGINT here
            rec_parent_logger.warning("SIGINT detected. Exit pool.")
            break
        
    # Finish up
    worker_pool.close()
    rec_parent_logger.info("Closed the pool, waiting to join...")
        
    worker_pool.join()
    rec_parent_logger.info("Joined the pool.")
    
    sys.exit(0)



def write_reconciled_to_db(parameters, reconciled_queue):
    
    
    reconciled_writer = log_writer.logger
    reconciled_writer.set_name("reconciled_writer")
    
    # Signal handling: 
    # SIGINT raises KeyboardInterrupt,  close dbw, terminate pool and exit. # TODO: pool.terminate() or close()?
    # SIGUSR1 is ignored. The process terminates when queue is empty. Close and join the pool
    def handler(sigusr, frame):
        reconciled_writer.warning("SIGUSR1 detected. Finish processing current queues.")
        signal.signal(sigusr, signal.SIG_IGN)    
        signal.signal(signal.SIGPIPE,signal.SIG_DFL) # reset SIGPIPE so that no BrokePipeError when SIGINT is received
        
    signal.signal(signal.SIGUSR1, handler) # ignore SIGUSR1
    reconciled_schema_path = os.getcwd() + "/config/" + parameters["reconciled_schema_path"]

    # wait to get raw collection name
    while parameters["reconciled_collection"]=="":
        time.sleep(1)
        
    dbw = DBClient(**parameters["db_param"], database_name = parameters["reconciled_database"], 
                   collection_name = parameters["reconciled_collection"], schema_file=reconciled_schema_path)
    REC_TIMEOUT = parameters["reconciliation_timeout"]
    # Write to db
    while True:
        try:
            try:
                reconciled_traj = reconciled_queue.get(timeout = REC_TIMEOUT)
            except queue.Empty:
                reconciled_writer.warning("Getting from reconciled_queue reaches timeout.")
                break
        
            dbw.write_one_trajectory(thread = True, **reconciled_traj)
            
        except (KeyboardInterrupt, BrokenPipeError): # handle SIGINT here 
            reconciled_writer.warning("SIGINT detected. Exit reconciled writer")
            break
    
    
    reconciled_writer.info("Final count in reconciled collection {}: {}".format(dbw.collection_name, dbw.count()))
    
    
    # Safely close the mongodb client connection
    del dbw
    reconciled_writer.warning("DBWriter closed. Exit reconciled_writer")
    sys.exit(0)



    
    
    
    
    
    

if __name__ == '__main__':
    import json
    # initialize parameters
    with open('config/parameters.json') as f:
        parameters = json.load(f)

    reconciliation_args={}
    for key in ["lam2_x","lam2_y","lam1_x","lam1_y", "ph", "ih"]:
        reconciliation_args[key] = parameters[key]
    
    # send some fragments to queue
    stitched_q = multiprocessing.Manager().Queue()
    reconciled_queue = multiprocessing.Manager().Queue()
    counter = 0 
    
    test_dbr = DBClient(**parameters["db_param"], database_name = "trajectories", latest_collection=True)
    
    for doc in test_dbr.collection.find({}):
        stitched_q.put([doc["_id"]])
        counter += 1
        print("doc length: ", len(doc["timestamp"]))
        if counter > 5:
            break
        
    print("current q size: ", stitched_q.qsize())
    
    # pool
    # reconciliation_pool(parameters, stitched_q)
    
    while not stitched_q.empty():
        fragment_list = stitched_q.get(block=False)
        combined_trajectory = combine_fragments(test_dbr.collection, fragment_list)
        reconcile_single_trajectory(reconciliation_args, combined_trajectory, reconciled_queue)
        
    print("final queue size: ",reconciled_queue.qsize())
        
    
    
    
    
    
    