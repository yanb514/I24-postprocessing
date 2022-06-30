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

import i24_logger.log_writer as log_writer
from i24_database_api.db_writer import DBWriter
from i24_database_api.db_reader import DBReader
from i24_configparse import parse_cfg
# import warnings
# warnings.filterwarnings("ignore")

import math
from utils.reconciliation_module import receding_horizon_2d_l1, resample, receding_horizon_2d, combine_fragments



def reconcile_single_trajectory(reconciliation_args, combined_trajectory, reconciled_queue) -> None:
    """
    Resample and reconcile a single trajectory, and write the result to a queue
    :param next_to_reconcile: a trajectory document
    :return:
    """
    
    rec_worker_logger = log_writer.logger
    rec_worker_logger.set_name("rec_worker")

    resampled_trajectory = resample(combined_trajectory)
    rec_worker_logger.debug("*** 2. Resampled.", extra = None)
    
    finished_trajectory = receding_horizon_2d(resampled_trajectory, **reconciliation_args)
    rec_worker_logger.debug("*** 3. Reconciled a trajectory. Trajectory duration: {:.2f}s, length: {}".format(finished_trajectory["last_timestamp"]-finished_trajectory["first_timestamp"], len(finished_trajectory["timestamp"])), extra = None)
   
    reconciled_queue.put(finished_trajectory)
    rec_worker_logger.debug("reconciled queue size: {}".format(reconciled_queue.qsize()))


def reconciliation_pool(parameters, stitched_trajectory_queue: multiprocessing.Queue,) -> None:
    """
    Start a multiprocessing pool, each worker 
    :param stitched_trajectory_queue: results from stitchers, shared by mp.manager
    :param pid_tracker: a dictionary
    :return:
    """
    # parameters
    reconciliation_args = {"lam2_x": parameters.lam2_x,
                           "lam2_y": parameters.lam2_y,
                           # "lam1_x": parameters.lam1_x, 
                           # "lam1_y": parameters.lam1_y,
                           "PH": parameters.ph,
                           "IH": parameters.ih}
    
    rec_parent_logger = log_writer.logger
    rec_parent_logger.set_name("rec_parent")
    setattr(rec_parent_logger, "_default_logger_extra",  {})

    # Reset collection
    reconciled_schema_path = os.path.join(os.environ["user_config_directory"],parameters.reconciled_schema_path)
    dbw = DBWriter(parameters, collection_name = parameters.reconciled_collection, schema_file=reconciled_schema_path)
    raw = DBReader(parameters, collection_name="garbage_dump_2")
    # dbw.reset_collection() # This line throws OperationFailure, not sure how to fix it

    # Signal handling
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    worker_pool = Pool(processes=parameters.reconciliation_pool_size)
    signal.signal(signal.SIGINT, original_sigint_handler)
    
    rec_parent_logger.info("** Reconciliation pool starts. Pool size: {}".format(parameters.reconciliation_pool_size), extra = None)
    
    if parameters.mode == "finish_processing":
        signal.signal(signal.SIGINT, signal.SIG_IGN)    
        signal.signal(signal.SIGPIPE,signal.SIG_DFL) # reset SIGPIPE so that no BrokePipeError when SIGINT is received
    
    # Create a shared queue to store workers results # TODO: max queue size
    reconciled_queue = multiprocessing.Manager().Queue()
    
    while True:
        try:
            next_to_reconcile = stitched_trajectory_queue.get(timeout = parameters.stitched_trajectory_queue_get_timeout)
        except: 
            rec_parent_logger.warning("Getting from stitched trajectories queue is timed out after {}s. Close the reconciliation pool.".format(parameters.stitched_trajectory_queue_get_timeout))
            break
        
        combined_trajectory = combine_fragments(raw.collection, next_to_reconcile)
        rec_parent_logger.debug("*** 1. Combined stitched fragments.", extra = None)
        
        worker_pool.apply_async(reconcile_single_trajectory, (reconciliation_args, combined_trajectory, reconciled_queue, ))

    # Finish up
    worker_pool.close()
    rec_parent_logger.info("Closed the pool, waiting to join...")
        
    worker_pool.join()
    rec_parent_logger.info("Joined the pool.")
    
    # Write to db
    while not reconciled_queue.empty():
        reconciled_traj = reconciled_queue.get(block = False)
        dbw.write_one_trajectory(thread = True, **reconciled_traj)
        rec_parent_logger.debug("Current count in {}: {}".format(dbw.collection_name, dbw.count()))
    
    time.sleep(2)
    rec_parent_logger.info("Final count in stitched collection: {}".format(dbw.count()))
    
    # TODO: Safely close the mongodb client connection?
    sys.exit(0)




if __name__ == '__main__':
    
    # initialize parameters
    config_path = os.path.join(os.getcwd(),"config")
    os.environ["user_config_directory"] = config_path
    os.environ["my_config_section"] = "TEST"
    parameters = parse_cfg("my_config_section", cfg_name = "test_param.config")

    reconciliation_args = {"lam2_x": parameters.lam2_x,
                           "lam2_y": parameters.lam2_y,
                           # "lam1_x": parameters.lam1_x, 
                           # "lam1_y": parameters.lam1_y,
                           "PH": parameters.ph,
                           "IH": parameters.ih}
    
    # send some fragments to queue
    stitched_q = multiprocessing.Manager().Queue()
    reconciled_queue = multiprocessing.Manager().Queue()
    counter = 0 
    
    test_dbr = DBReader(parameters, collection_name="garbage_dump_2")
    for doc in test_dbr.collection.find({}):
        stitched_q.put([doc["_id"]])
        counter += 1
        print("doc length: ", len(doc["timestamp"]))
        if counter > 5:
            break
        
    print("current q size: ", stitched_q.qsize())
    
    # pool
    reconciliation_pool(parameters, stitched_q)
    
    # while not stitched_q.empty():
    #     fragment_list = stitched_q.get(block=False)
    #     combined_trajectory = combine_fragments(test_dbr.collection, fragment_list)
    #     reconcile_single_trajectory(reconciliation_args, combined_trajectory, reconciled_queue)
        
    # print("final queue size: ",reconciled_queue.qsize())
        
    
    
    
    
    
    