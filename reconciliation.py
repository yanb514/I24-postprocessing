# -----------------------------
__file__ = 'reconciliation.py'
__doc__ = """

"""

# -----------------------------
import multiprocessing
from multiprocessing import Pool
import parameters, db_parameters
# from I24_logging.log_writer import I24Logger
from db_writer import DBWriter
from utils.reconciliation_module import receding_horizon_2d_l1, resample, receding_horizon_2d,combine_fragments
import os
import sys
from db_reader import DBReader
from i24_logger.log_writer import logger
reconciliation_logger = logger

import time

reconciliation_args = {"lam2_x": parameters.LAM2_X,
                       "lam2_y": parameters.LAM2_Y,
                       # "lam1_x": parameters.LAM1_X, 
                       # "lam1_y": parameters.LAM1_Y,
                       "PH": parameters.PH,
                       "IH": parameters.IH}

# initiate a dbw and dbr object
dbw = DBWriter(host=db_parameters.DEFAULT_HOST, port=db_parameters.DEFAULT_PORT, username=db_parameters.DEFAULT_USERNAME,   
               password=db_parameters.DEFAULT_PASSWORD,
               database_name=db_parameters.DB_NAME, server_id=1, process_name=1, process_id=1, session_config_id=1)
raw = DBReader(host=db_parameters.DEFAULT_HOST, port=db_parameters.DEFAULT_PORT, username=db_parameters.DEFAULT_USERNAME,   
               password=db_parameters.DEFAULT_PASSWORD,
               database_name=db_parameters.DB_NAME, collection_name=db_parameters.RAW_COLLECTION)



def reconcile_single_trajectory(stitched_trajectory_queue: multiprocessing.Queue) -> None:
    """
    Resample and reconcile a single trajectory, and write the result to DB
    :param stitched_trajectory_queue: 
    :param result_queue:
    :return:
    """
    
    # reconciliation_logger.put((logging.DEBUG, "Reconciling on trajectory {}.".format('0')))
    # get the pid of current worker
    # DO THE RECONCILIATION
    # sys.stdout.flush()

    next_to_reconcile = stitched_trajectory_queue.get(block=True) # path list
    # print("...got next...")
    reconciliation_logger.info("*** Got a stitched trajectory document.", extra = None)
    
    combined_trajectory = combine_fragments(raw.collection, next_to_reconcile)
    # print("...combined...")
    reconciliation_logger.info("*** Combined stitched fragments.", extra = None)

    resampled_trajectory = resample(combined_trajectory)
    # print("...resampled...")
    reconciliation_logger.info("*** Resampled.", extra = None)
    
    finished_trajectory = receding_horizon_2d(resampled_trajectory, **reconciliation_args)
    # print("...finished...")
    reconciliation_logger.info("*** Reconciled a trajectory. duration = {:.2f}s.".format(finished_trajectory["last_timestamp"]-finished_trajectory["first_timestamp"]), extra = None)

    # TODO: replace with schema validation in dbw before insert
    finished_trajectory["timestamp"] = list(finished_trajectory["timestamp"])
    finished_trajectory["x_position"] = list(finished_trajectory["x_position"])
    finished_trajectory["y_position"] = list(finished_trajectory["y_position"])
    
    # print("writing to db...")
    dbw.write_reconciled_trajectory(thread=True, **finished_trajectory)
    # print("reconciled collection: ", dbw.db["reconciled_trajectories"].count_documents({}))
    reconciliation_logger.info("*** write reconciled to database", extra = None)


def reconciliation_pool(stitched_trajectory_queue: multiprocessing.Queue,
                         pid_tracker) -> None:
    """
    Start a multiprocessing pool, each worker 
    :param stitched_trajectory_queue: results from stitchers, shared by mp.manager
    :param pid_tracker: a dictionary
    :return:
    """

    # print("** reconciliation starts...")
    # initiate a logger object
    # reconciliation_logger = I24Logger(owner_process_name = "reconciliation", connect_file=True, file_log_level='DEBUG', 
    #                     owner_parent_name = "postprocessing_manager", connect_console=True, console_log_level='INFO')

    worker_pool = Pool(processes=parameters.RECONCILIATION_POOL_SIZE)
    reconciliation_logger.info("** Reconciliation pool starts. Pool size: {}".format(parameters.RECONCILIATION_POOL_SIZE), extra = None)

    while True:
        worker_pool.apply_async(reconcile_single_trajectory, (stitched_trajectory_queue, ))
        time.sleep(0.2) # put some throttle so that while waiting for a job this loop does run tooo fast


    
# if __name__ == '__main__':
#     print("no code to run")
    
    