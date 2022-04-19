# -----------------------------
__file__ = 'stitcher.py'
__doc__ = """
Operates the trajectory fragment stitcher, continuously consuming raw trajectory fragments as they are
written to the database.
"""

# -----------------------------
import multiprocessing
import stitcher_parameters
import db_parameters
import logging
# TODO: check and test database implementation
import pymongo
import pymongo.errors
from db_reader import DBReader
from db_writer import DBWriter
from collections import deque, OrderedDict
from utils.stitcher_module import min_nll_cost
from utils.data_structures import Fragment, PathCache

# helper functions
def _first(dict):
    try:
        key,val = next(iter(dict.items()))
        return key, val
    except StopIteration:
        raise StopIteration


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
    IDLE_TIME = stitcher_parameters.IDLE_TIME
    
    # Initialize some data structures
    curr_fragments = deque()  # fragments that are in current window (left, right), sorted by last_timestamp
    past_fragments = OrderedDict()  # set of ids indicate end of fragment ready to be matched
    P = PathCache() # an LRU cache of Fragment object (see utils.data_structures)

    # Make database connection for writing
    dbw = DBWriter(host=db_parameters.DEFAULT_HOST, port=db_parameters.DEFAULT_PORT, username=db_parameters.DEFAULT_USERNAME,   
                   password=db_parameters.DEFAULT_PASSWORD,
                   database_name=db_parameters.DB_NAME, server_id=1, process_name=1, process_id=1, session_config_id=1)
    
    print("** Stitching starts. fragment_queue size: ", fragment_queue.qsize())
    # while True: 
    while fragment_queue.qsize() > 0:
        current_fragment = fragment_queue.get() # fragment = dh._get_first('last_timestamp') # get the earliest ended fragment
        print("*** getting fragment")  
        fragment = Fragment(current_fragment)
        P.add_node(current_fragment)
        right = fragment.t[-1] # right pointer: current end time
        left = right - TIME_WIN
        print("left, right: ", left, right)
        
        # compute fragment statistics (1d motion model)
        fragment.compute_stats()
        print("Curr_fragments size: ", len(curr_fragments))
        
        # remove out of sight fragments 
        while curr_fragments and curr_fragments[0].t[-1] < left: 
            past_fragment = curr_fragments.popleft()
            past_fragments[past_fragment.id] = past_fragment
        print("Past_fragments size: ", len(past_fragments))
        
        # compute score from every fragment in curr to fragment, update Cost
        for curr_fragment in curr_fragments:
            cost = min_nll_cost(curr_fragment, fragment, TIME_WIN, VARX, VARY)
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
                        print("** match tail of {} to head of {}".format(best_tail.ID, best_head.ID))
                        # path[best_head.id] = path[best_tail.id]
                        P.union(best_head.id, best_tail.id)
                        remove_keys.add(ready.id)
                        Fragment.match_tail_head(best_tail, best_head)

            [past_fragments.pop(key) for key in remove_keys]
            curr_size = len(past_fragments)

        curr_fragments.append(fragment)    
        # cache_size_pre = len(P.cache)
        
        # write paths from P if time out is reached
        while True:
            try:
                _, root = _first(P.cache)
                if root.last_modified_timestamp < left - IDLE_TIME:
                    path = P.pop_first_path()
                    print("write to db: root {}, last_modified {:.2f}, path length: {}".format(root.ID, root.last_modified_timestamp,len(path)))
                    stitched_trajectory_queue.put(path) # doesn't know ObjectId
                    dbw.write_stitch(path)
                else: # break if first in cache is not timed out yet
                    break
            except StopIteration: # break if nothing in cache
                break
            
        cache_size_post = len(P.cache)    
        print("Cache size: ", cache_size_post)


        
            
            
    
