# -----------------------------
__file__ = 'reconciliation.py'
__doc__ = """

"""

# -----------------------------
import multiprocessing
from multiprocessing import Pool
import time
import os
import i24_logger.log_writer as log_writer

from i24_database_api.db_writer import DBWriter
from i24_database_api.db_reader import DBReader
from utils.reconciliation_module import receding_horizon_2d_l1, resample, receding_horizon_2d, combine_fragments
from i24_configparse import parse_cfg

import warnings
warnings.filterwarnings("ignore")

config_path = os.path.join(os.getcwd(),"config")
os.environ["user_config_directory"] = config_path
os.environ["my_config_section"] = "DEBUG"
parameters = parse_cfg("my_config_section", cfg_name = "test_param.config")

# initiate a dbw and dbr object
schema_path = os.path.join(os.environ["user_config_directory"],parameters.reconciled_schema_path)
dbw = DBWriter(host=parameters.default_host, port=parameters.default_port, 
                username=parameters.default_username, password=parameters.default_password,
                database_name=parameters.db_name, collection_name=parameters.reconciled_collection,
                server_id=1, process_name=1, process_id=1, session_config_id=1, 
                max_idle_time_ms = 20000, schema_file=schema_path)
raw = DBReader(host=parameters.default_host, port=parameters.default_port, 
            username=parameters.readonly_user, password=parameters.default_password,
            database_name=parameters.db_name, collection_name=parameters.raw_collection)

reconciliation_args = {"lam2_x": parameters.lam2_x,
                       "lam2_y": parameters.lam2_y,
                       # "lam1_x": parameters.lam1_x, 
                       # "lam1_y": parameters.lam1_y,
                       "PH": parameters.ph,
                       "IH": parameters.ih}



def reconcile_single_trajectory(stitched_trajectory_queue: multiprocessing.Queue) -> None:
    """
    Resample and reconcile a single trajectory, and write the result to DB
    :param stitched_trajectory_queue: 
    :param result_queue:
    :return:
    """
    
    rec_worker_logger = log_writer.logger
    rec_worker_logger.set_name("rec_worker")
    
    # Establish db connections (# connections = # pool workers)
    rec_worker_logger.info("*** Reconciliation worker started", extra = None)
    
    # while True:
        
    next_to_reconcile = stitched_trajectory_queue.get(block=True) # path list
    # print("...got next...")
    rec_worker_logger.info("*** 1. Got a stitched trajectory document.", extra = None)
    
    combined_trajectory = combine_fragments(raw.collection, next_to_reconcile)
    # print("...combined...")
    rec_worker_logger.info("*** 2. Combined stitched fragments.", extra = None)

    resampled_trajectory = resample(combined_trajectory)
    # print("...resampled...")
    rec_worker_logger.info("*** 3. Resampled.", extra = None)
    
    finished_trajectory = receding_horizon_2d(resampled_trajectory, **reconciliation_args)
    # print("...finished...")
    rec_worker_logger.info("*** 4. Reconciled a trajectory. duration = {:.2f}s.".format(finished_trajectory["last_timestamp"]-finished_trajectory["first_timestamp"]), extra = None)
   
    # print("writing to db...")
    dbw.write_one_trajectory(**finished_trajectory)
    # print("reconciled collection: ", dbw.db["reconciled_trajectories"].count_documents({}))
    rec_worker_logger.info("*** 5. Reconciliation worker writes to database", extra = None)
    
    # rec_worker_logger.info("reconciliation raw: {}".format(id(raw)))
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

    worker_pool = Pool(processes=parameters.reconciliation_pool_size)
    rec_parent_logger.info("** Reconciliation pool starts. Pool size: {}".format(parameters.reconciliation_pool_size), extra = None)

    while True: 
        worker_pool.apply_async(reconcile_single_trajectory, (stitched_trajectory_queue, ))
        time.sleep(0.5) # put some throttle so that while waiting for a job this loop does run tooo fast


if __name__ 