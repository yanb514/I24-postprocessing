# -----------------------------
__file__ = 'stitcher.py'
__doc__ = """
Operates the trajectory fragment stitcher, continuously consuming raw trajectory fragments as they are
written to the database.
"""

# -----------------------------
import multiprocessing
import stitcher_parameters
import logging
# TODO: check and test database implementation
import pymongo
import pymongo.errors
from db_reader import DBReader
from db_writer import DBWriter
from collections import deque, OrderedDict
from utils.stitcher_module import _getCost
from utils.data_structures import Fragment, PathCache

# for debugging
from time import sleep
import queue
    """
    read from database with change stream
    :param fragment_queue:
    :param log_queue:
    :return:
    """
    # TODO: there are some issues with this workflow that need to be resolved...
    # When is it appropriate to re-connect versus resume?

    print("Raw fragment ingester started.")
    log_queue.put((logging.INFO, "Raw fragment ingester started."))
    
    while True:
        # Make a connection to database for reading
        try:
            dh = DataHandler(**parameters.DB_PARAMS)
            
        except pymongo.errors.ConnectionFailure:
            log_queue.put((logging.ERROR, "MongoDB connection failed"))
            # Go past the change stream loop and try the connection again.
            continue

        # Initialize at None since we have no usable resume token to start.
        resume_token = None
        while True:
            try:
                # TODO: fix corner case where resume_token is stuck at a value but won't work
                watch_for = [{'$match': {'operationType': 'insert'}}]
                with dh.collection.watch(watch_for, resume_after=resume_token) as stream:
                    for insert_change in stream:
                        # TODO: parse the change in whatever way needed to properly organize `fragment_queue`
                        fragment_queue.put(insert_change)
                        resume_token = stream.resume_token
            except pymongo.errors.PyMongoError:
                # The ChangeStream encountered an unrecoverable error or the resume attempt failed.
                if resume_token is None:
                    # There is no usable resume token because there was a failure during ChangeStream initialization.
                    log_queue.put((logging.ERROR, "MongoDB error in get_raw_fragments"))
                    # Break out of this secondary WHILE loop so that the connection can be restarted
                    break
                # We have a resume token, so let's go back and try to use it.
                else:
                    continue


def stitch_raw_trajectory_fragments(fragment_queue,
                                    stitched_trajectory_queue,
                                    log_queue):
    """
    fragment_queue is sorted by last_timestamp
    :param fragment_queue: fragments sorted in last_timestamp
    :param stitched_trajectory_queue: to store stitched trajectories result
    :param log_queue:
    :return: None
    """
    # Get parameters
    TIME_WIN = stitcher_parameters.TIME_WIN
    VARX = stitcher_parameters.VARX
    VARY = stitcher_parameters.VARY
    THRESH = stitcher_parameters.THRESH
    X_MAX = stitcher_parameters.X_MAX
    X_MIN = stitcher_parameters.X_MIN
    IDLE_TIME = stitcher_parameters.IDLE_TIME
    
    # Initialize some data structures
    curr_fragments = deque()  # fragments that are in current window (left, right), sorted by last_timestamp
    past_fragments = OrderedDict()  # set of ids indicate end of fragment ready to be matched
    path_cache = PathCache() # an LRU cache of Fragment object (see utils.data_structures)

    # Make database connection for writing
    dw = DBWriter(parameters.STITCHED_COLLECTION)
    
    print("** Stitching starts. fragment_queue size: ", fragment_queue.qsize())
    while True: 

        current_fragment = fragment_queue.get(block=True) # fragment = dh._get_first('last_timestamp') # get the earliest ended fragment
        print("*** getting fragment")
        # DO THE PROCESSING ON THE FRAGMENT  
        curr_id = current_fragment['_id'] # last_fragment = fragment['id']
        fragment = Fragment(current_fragment)
        path_cache[curr_id] = curr_id
        right = fragment.t[-1] # right pointer: current end time
        left = right - TIME_WIN
        print("left, right: ", left, right)
        
        # compute fragment statistics (1d motion model)
        fragment.compute_stats()

        # remove out of sight fragments 
        while curr_fragments and curr_fragments[0].t[-1] < left: 
            past_fragment = curr_fragments.popleft()
            past_fragments[past_fragment.id] = past_fragment
        

        # compute score from every fragment in curr to fragment, update Cost
        for curr_fragment in curr_fragments:
            cost = _getCost(curr_fragment, fragment, TIME_WIN, VARX, VARY)
            if cost > THRESH:
                curr_fragment.add_conflict(fragment)
            elif cost > 0:
                curr_fragment.add_suc(cost, fragment)
                fragment.add_pre(cost, curr_fragment)
                        
        prev_size = 0
        curr_size = len(past_fragments)
        
        # start iterative matching
        while curr_size > 0 and curr_size != prev_size:
            prev_size = len(past_fragments)
            remove_keys = set()
            for _, ready in past_fragments.items(): # all fragments in past_fragments are ready to be matched to tail
                best_head = ready.get_first_suc()
                if not best_head or not best_head.pre: # if ready has no match or best head already matched to other fragments# go to the next ready
                    # past_fragments.pop(ready.id)
                    remove_keys.add(ready.id)
                
                else:
                    try: best_tail = best_head.get_first_pre()
                    except: best_tail = None
                    if best_head and best_tail and best_tail.id == ready.id and best_tail.id not in ready.conflicts_with:
                        # print("** match tail of {} to head of {}".format(best_tail.id, best_head.id))
                        # path[best_head.id] = path[best_tail.id]
                        path_cache.union(best_head.id, best_tail.id)
                        remove_keys.add(ready.id)
                        Fragment.match_tail_head(best_tail, best_head)

            [past_fragments.pop(key) for key in remove_keys]
            curr_size = len(past_fragments)

        curr_fragments.append(fragment)    
        
        # write paths from path_cache if 
        while next(iter(path_cache.cache)).last_modified_time < curr_fragment["last_timestamp"] - IDLE_TIME:
            try:
                path = path_cache.pop_first_path()
                
                stitched_trajectory_queue.put(path) # doesn't know ObjectId
                dw.write_stitch(path)
            except: # no path ready to be written
                pass


        
            
            
    
