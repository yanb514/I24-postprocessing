# -----------------------------
__file__ = 'reconciliation.py'
__doc__ = """

"""

# -----------------------------
import multiprocessing
from multiprocessing import Pool
import parameters, db_parameters
from I24_logging.log_writer import I24Logger
from db_writer import DBWriter
from utils.reconciliation_module import receding_horizon_2d_l1, resample, receding_horizon_2d
import os
import sys
from db_reader import DBReader

import time

reconciliation_args = {"lam2_x": parameters.LAM2_X,
                       "lam2_y": parameters.LAM2_Y,
                       # "lam1_x": parameters.LAM1_X, 
                       # "lam1_y": parameters.LAM1_Y,
                       "PH": parameters.PH,
                       "IH": parameters.IH}

# initiate a dbw object
dbw = DBWriter(host=db_parameters.DEFAULT_HOST, port=db_parameters.DEFAULT_PORT, username=db_parameters.DEFAULT_USERNAME,   
               password=db_parameters.DEFAULT_PASSWORD,
               database_name=db_parameters.DB_NAME, server_id=1, process_name=1, process_id=1, session_config_id=1)

# initiate a logger object
# reconciliation_logger = I24Logger(owner_process_name = "reconciliation", connect_file=True, file_log_level='DEBUG', 
#                     connect_console=True, console_log_level='INFO')

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
    sys.stdout.flush()
    next_to_reconcile = stitched_trajectory_queue.get(block=True)
    resampled_trajectory = resample(next_to_reconcile)
    finished_trajectory = receding_horizon_2d(resampled_trajectory, **reconciliation_args)
    
    # TODO: replace with schema validation in dbw before insert
    finished_trajectory["timestamp"] = list(finished_trajectory["timestamp"])
    finished_trajectory["x_position"] = list(finished_trajectory["x_position"])
    finished_trajectory["y_position"] = list(finished_trajectory["y_position"])
    
    print("writing to db...")
    dbw.write_reconciled_trajectory(thread=True, **finished_trajectory)
    print("reconciled collection: ", dbw.db["reconciled_trajectories"].count_documents({}))

def reconciliation_pool(stitched_trajectory_queue: multiprocessing.Queue,
                         pid_tracker) -> None:
    """
    Start a multiprocessing pool, each worker 
    :param stitched_trajectory_queue: results from stitchers, shared by mp.manager
    :param pid_tracker: a dictionary
    :return:
    """

    print("** reconciliation starts...")
    print("reconciliation pool qsize", stitched_trajectory_queue.qsize())
    worker_pool = Pool(processes=parameters.RECONCILIATION_POOL_SIZE)
    # worker_pool = Pool(4)
    # TODO: decide about tasks per child
    while True:
        worker_pool.apply_async(reconcile_single_trajectory, (stitched_trajectory_queue, ))

    # TODO: try to track PIDs
    # TODO: make sure this will exit gracefully


    
# if __name__ == '__main__':
#     print("no code to run")
    
    