# -----------------------------
__file__ = 'subsys_data.py'
__doc__ = """
Contains the data subsystem that receives various types of data from SwCS and populates and manages
the data cache for the AI-DSS.
"""

# -----------------------------
import multiprocessing
import parameters
import pymongo
import logging
from utils.reconciliation_module import receding_horizon_2d_l1, resample


def reconcile_single_trajectory(trajectory_data, result_queue: multiprocessing.Queue,
                                log_queue: multiprocessing.Queue) -> None:
    """

    :param trajectory_data: a dict
    :param result_queue:
    :param log_queue:
    :return:
    """
    log_queue.put((logging.DEBUG, "Reconciling on trajectory {}.".format('0')))

    # DO THE RECONCILIATION
    resampled_trajectory = resample(trajectory_data)
    finished_trajectory = receding_horizon_2d_l2(resampled_trajectory,**parameters.RECONCILIATION_PARAMS)
    result_queue.put(finished_trajectory)


def reconciled_results_handler(reconciled_queue: multiprocessing.Queue):
    client = pymongo.MongoClient(parameters.DATABASE_URL)
    db = client.trajectories
    collection = db.processed_trajectories

    while True:
        # Get the next result and wait as long as necessary.
        reconciled_result = reconciled_queue.get(block=True)
        # Insert the result in the database.
        # collection.insert(reconciled_result)


def handle_reconcile_error(process_exception):
    pass


def reconciliation_pool(stitched_trajectory_queue: multiprocessing.Queue,
                        log_queue: multiprocessing.Queue, pid_tracker) -> None:
    """

    :param stitched_trajectory_queue:
    :param log_queue:
    :param pid_tracker:
    :return:
    """
    # TODO: decide about tasks per child
    worker_pool = multiprocessing.pool.Pool(processes=parameters.RECONCILIATION_POOL_SIZE)
    reconciliation_results_queue = multiprocessing.Queue(maxsize=parameters.RECONCILED_TRAJECTORY_QUEUE_SIZE)
    results_handler = multiprocessing.Process(target=reconciled_results_handler, args=(reconciliation_results_queue,),
                                              name='reconciled_results_handler', daemon=True)
    
    # TODO: I'm not 100% sure this is correct...needs testing
    while True:
        next_to_reconcile = stitched_trajectory_queue.get(block=True)
        worker_pool.apply_async(func=reconcile_single_trajectory,
                                args=(next_to_reconcile, reconciliation_results_queue, log_queue))
        # TODO: try to track PIDs
        # TODO: make sure this will exit gracefully


if __name__ == '__main__':
    print("NO CODE TO RUN")
