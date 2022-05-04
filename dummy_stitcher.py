# -----------------------------
__file__ = 'dummy_stitcher.py'
__doc__ = """
Operates the trajectory fragment stitcher, continuously consuming raw trajectory fragments as they are
written to the database.
"""

# -----------------------------
import multiprocessing
import db_parameters, parameters
from I24_logging.log_writer import I24Logger
from db_writer import DBWriter 
from collections import deque
from utils.stitcher_module import min_nll_cost
from utils.data_structures import Fragment, PathCache
import time
import sys

# helper functions
def _first(dict):
    try:
        key,val = next(iter(dict.items()))
        return key, val
    except StopIteration:
        raise StopIteration


def dummy_stitcher(direction, fragment_queue,
                                    stitched_trajectory_queue):
    """
    fragment_queue is sorted by last_timestamp
    :param direction: "east" or "west"
    :param fragment_queue: fragments sorted in last_timestamp
    :param stitched_trajectory_queue: to store stitched trajectories result
    :param log_queue:
    :return: None
    """

    dbw = DBWriter(host=db_parameters.DEFAULT_HOST, port=db_parameters.DEFAULT_PORT, username=db_parameters.DEFAULT_USERNAME,   
                   password=db_parameters.DEFAULT_PASSWORD,
                   database_name=db_parameters.DB_NAME, server_id=1, process_name=1, process_id=1, session_config_id=1)
    dbw.db[db_parameters.STITCHED_COLLECTION].drop()
    while True:
        sys.stdout.flush()
        time.sleep(2)
        fragment = fragment_queue.get(block = True) # make object
        # fragment.pop('_id')
        fragment["fragment_ids"] = [fragment["_id"]]
        stitched_trajectory_queue.put(fragment)
        dbw.write_one_trajectory(thread = False, collection_name = db_parameters.STITCHED_COLLECTION, **fragment)
        # print(dbw.db[db_parameters.STITCHED_COLLECTION].count_documents({}))
            

        
            
            
    
