# -----------------------------
__file__ = 'reconciliation.py'
__doc__ = """

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
import warnings
warnings.filterwarnings("ignore")

import math
from utils.reconciliation_module import receding_horizon_2d_l1, resample, receding_horizon_2d, combine_fragments

# config_path = os.path.join(os.getcwd(),"config")
# os.environ["user_config_directory"] = config_path
# os.environ["my_config_section"] = "DEBUG"
parameters = parse_cfg("my_config_section", cfg_name = "test_param.config")

# initiate a dbw and dbr object
reconciled_schema_path = os.path.join(os.environ["user_config_directory"],parameters.reconciled_schema_path)

dbw = DBWriter(parameters, collection_name = parameters.reconciled_collection, schema_file=reconciled_schema_path)
raw = DBReader(parameters, collection_name=parameters.raw_collection)

reconciliation_args = {"lam2_x": parameters.lam2_x,
                       "lam2_y": parameters.lam2_y,
                       # "lam1_x": parameters.lam1_x, 
                       # "lam1_y": parameters.lam1_y,
                       "PH": parameters.ph,
                       "IH": parameters.ih}


def reconcile_single_trajectory(next_to_reconcile) -> None:
    """
    Resample and reconcile a single trajectory, and write the result to DB
    :param next_to_reconcile: a trajectory document
    :return:
    """
    
    rec_worker_logger = log_writer.logger
    rec_worker_logger.set_name("rec_worker")
        
    # print("...got next...")
    rec_worker_logger.debug("*** 1. Got a stitched trajectory document.", extra = None)
    
    combined_trajectory = combine_fragments(raw.collection, next_to_reconcile)
    # print("...combined...")
    rec_worker_logger.debug("*** 2. Combined stitched fragments.", extra = None)

    resampled_trajectory = resample(combined_trajectory)
    # print("...resampled...")
    rec_worker_logger.debug("*** 3. Resampled.", extra = None)
    
    finished_trajectory = receding_horizon_2d(resampled_trajectory, **reconciliation_args)
    # print("...finished...")
    rec_worker_logger.debug("*** 4. Reconciled a trajectory. Trajectory duration: {:.2f}s.".format(finished_trajectory["last_timestamp"]-finished_trajectory["first_timestamp"]), extra = None)
   
    # print("writing to db...")
    dbw.write_one_trajectory(**finished_trajectory)
    # print("reconciled collection: ", dbw.db["reconciled_trajectories"].count_documents({}))
    rec_worker_logger.debug("*** 5. Reconciliation worker writes to database", extra = None)
    
    rec_worker_logger.info("reconciled_trajectories collection size: {}".format(dbw.collection.count_documents({})))
    # rec_worker_logger.info("reconciliation dbw: {}".format(id(dbw)))
    
    



def reconciliation_pool(stitched_trajectory_queue: multiprocessing.Queue,
                         ) -> None:
    """
    Start a multiprocessing pool, each worker 
    :param stitched_trajectory_queue: results from stitchers, shared by mp.manager
    :param pid_tracker: a dictionary
    :return:
    """
    
    
    rec_parent_logger = log_writer.logger
    rec_parent_logger.set_name("rec_parent")
    setattr(rec_parent_logger, "_default_logger_extra",  {})

    # Reset collection
    # rec_parent_logger.debug("before reset collection size: {}".format(dbw.count()))
    # dbw.reset_collection()
    # rec_parent_logger.debug("after reset collection size: {}".format(dbw.count()))

    # Signal handling
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    worker_pool = Pool(processes=parameters.reconciliation_pool_size)
    signal.signal(signal.SIGINT, original_sigint_handler)
    
    rec_parent_logger.info("** Reconciliation pool starts. Pool size: {}".format(parameters.reconciliation_pool_size), extra = None)

    # signal.signal(signal.SIGINT, signal.SIG_IGN)    
    # signal.signal(signal.SIGPIPE,signal.SIG_DFL) # reset SIGPIPE so that no BrokePipeError when SIGINT is received
    
    while True:
        try:
            traj = stitched_trajectory_queue.get(timeout = parameters.stitched_trajectory_queue_get_timeout)
        except: 
            rec_parent_logger.warning("Getting from stitched trajectories queue is timed out. Close the reconciliation pool.")
            break
        
        worker_pool.apply_async(reconcile_single_trajectory, (traj, ))

    worker_pool.close()
    rec_parent_logger.info("Closed the pool")
        
    worker_pool.join()
    rec_parent_logger.info("Joined the pool. Exit with code 0")
    
    sys.exit(0)




