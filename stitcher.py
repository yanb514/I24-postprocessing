# -----------------------------
__file__ = 'stitcher.py'
__doc__ = """
Operates the trajectory fragment stitcher, continuously consuming raw trajectory fragments as they are
written to the database.
"""

# -----------------------------
import multiprocessing
import parameters
import time
import logging
# TODO: check and test database implementation
import pymongo
import pymongo.errors

from mongodb_reader import DataReader
from stitcher_module import spatial_temporal_match_online


def get_raw_fragments(fragment_queue: multiprocessing.Queue, log_queue: multiprocessing.Queue) -> None:
    """
    :param fragment_queue:
    :param log_queue:
    :return:
    """
    # TODO: there are some issues with this workflow that need to be resolved...
    # When is it appropriate to re-connect versus resume?

    print("Raw fragment ingester started.")
    log_queue.put((logging.INFO, "Raw fragment ingester started."))
    while True:
        try:
            # TODO: not sure how correct any of this DB connection is
            # client = pymongo.MongoClient(parameters.DATABASE_URL)
            # db = client.trajectories
            dr = DataReader(**parameters.DB_PARAMS, vis=False)
            # dr.db -> get a database
            # dr.collection -> get a collection
            
        except pymongo.errors.ConnectionFailure:
            log_queue.put((logging.ERROR, "..."))
            # Go past the change stream loop and try the connection again.
            continue

        # Initialize at None since we have no usable resume token to start.
        resume_token = None
        while True:
            try:
                # TODO: fix corner case where resume_token is stuck at a value but won't work
                watch_for = [{'$match': {'operationType': 'insert'}}]
                with dr.collection.watch(watch_for, resume_after=resume_token) as stream:
                    for insert_change in stream:
                        # TODO: parse the change in whatever way needed to properly organize `fragment_queue`
                        fragment_queue.put(insert_change)
                        resume_token = stream.resume_token
            except pymongo.errors.PyMongoError:
                # The ChangeStream encountered an unrecoverable error or the resume attempt failed.
                if resume_token is None:
                    # There is no usable resume token because there was a failure during ChangeStream initialization.
                    log_queue.put((logging.ERROR, "..."))
                    # Break out of this secondary WHILE loop so that the connection can be restarted
                    break
                # We have a resume token, so let's go back and try to use it.
                else:
                    continue


def stitch_raw_trajectory_fragments(fragment_queue: multiprocessing.Queue,
                                    stitched_trajectory_queue: multiprocessing.Queue,
                                    log_queue: multiprocessing.Queue) -> None:
    """
    :param fragment_queue:
    :param stitched_trajectory_queue:
    :param log_queue:
    :return:
    """

    while True:
        # Get next fragment and wait until one is available if necessary.
        next_fragment = fragment_queue.get(block=True)

        # DO THE PROCESSING ON THE FRAGMENT
        stitched_trajectory = None

        stitched_trajectory_queue.put(stitched_trajectory)


if __name__ == '__main__':
    print("NO CODE TO RUN")
