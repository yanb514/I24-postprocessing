# -----------------------------
__file__ = 'stitcher.py'
__doc__ = """
Operates the trajectory fragment stitcher, continuously consuming raw trajectory fragments as they are
written to the database.
"""

# -----------------------------
import os
from i24_logger.log_writer import logger 
from i24_database_api.db_writer import DBWriter
from collections import deque
from utils.stitcher_module import min_nll_cost
from utils.data_structures import Fragment, PathCache
import time


# TODO: exiting condition needs to be tested

def stitch_raw_trajectory_fragments(direction, fragment_queue,
                                    stitched_trajectory_queue, parameters):
    """
    fragment_queue is sorted by last_timestamp
    :param direction: "east" or "west"
    :param fragment_queue: fragments sorted in last_timestamp
    :param stitched_trajectory_queue: to store stitched trajectories result
    :param parameters: config_parse object
    :return: None
    """
    stitcher_logger = logger
    stitcher_logger.set_name("stitcher_"+direction)
    # Get parameters
    TIME_WIN = parameters.time_win
    VARX = parameters.varx
    VARY = parameters.vary
    THRESH = parameters.thresh
    IDLE_TIME = parameters.idle_time
    TIMEOUT = parameters.raw_trajectory_queue_get_timeout
    
    # Initialize some data structures
    curr_fragments = deque()  # fragments that are in current window (left, right), sorted by last_timestamp
    past_fragments = dict()  # set of ids indicate end of fragment ready to be matched, insertion ordered
    P = PathCache(attr_name=parameters.fragment_attr_name) # an LRU cache of Fragment object (see utils.data_structures)

    # Make a database connection for writing
    dbw = DBWriter(host=parameters.default_host, port=parameters.default_port, 
               username=parameters.default_username, password=parameters.default_password,
               database_name=parameters.db_name, collection_name = parameters.stitched_collection, 
               server_id=1, process_name=1, process_id=1, session_config_id=1, max_idle_time_ms = None,
               schema_file=parameters.stitched_schema_path)
        
    stitcher_logger.info("** Stitching starts. fragment_queue size: {}".format(fragment_queue.qsize()),extra = None)
    

    while True:
        try: # grab a fragment from the queue if queue is not empty
            fragment = Fragment(fragment_queue.get(block = True, timeout = TIMEOUT)) # make object
            fragment.curr = True
            P.add_node(fragment)
            
            # specify time window for curr_fragments
            # right = fragment.last_timestamp # right pointer: current end time
            left = fragment.first_timestamp - TIME_WIN
            
            # compute fragment statistics (1d motion model)
            fragment.compute_stats()
            
        except: # if queue is empty, process the remaining
            if not curr_fragments: # if all remaining fragments are processed
                stitcher_logger.info("remaining fragments: {}".format(len(P.path)))
                left += IDLE_TIME + 1e-6 # to make all fragments "ready" for tail matching
                if len(P.path) == 0:# exit the stitcher if all fragments are written to database
                    stitcher_logger.info("All remaining fragments are processed. Exiting stitcher")
                    break
            else: 
                # specify time window for curr_fragments
                left = P.get_fragment(curr_fragments[0]).last_timestamp + 1e-6
            fragment = None
           
        # print("left: ", left)
        # if fragment:
        #     print("curr fragment ID: ", fragment.ID)
        #     print("curr fragment time: {:.2f}-{:.2f}".format(fragment.first_timestamp, fragment.last_timestamp))
            
        # print("curr_fragment size (before): ", len(curr_fragments))
        # remove out of sight fragments 
        while curr_fragments and P.get_fragment(curr_fragments[0]).last_timestamp < left: 
            past_id = curr_fragments.popleft()
            past_fragment = P.get_fragment(past_id)
            past_fragment.curr = False
            past_fragment.past = True
            past_fragments[past_id] = past_fragment # could use keys only: past_fragments[past_id] = None if memory is an issue 
        
        # print("curr_fragment size (after): ", len(curr_fragments))
        # print("Past_fragments size: ", len(past_fragments))
        
        # compute score from every fragment in curr to fragment, update Cost
        if fragment is not None:
            for curr_id in curr_fragments:
                curr_fragment = P.get_fragment(curr_id)
                cost = min_nll_cost(curr_fragment, fragment, TIME_WIN, VARX, VARY)
                if cost > THRESH:
                    pass
                elif cost > -999:
                    curr_fragment.add_suc(cost, fragment)
                    fragment.add_pre(cost, curr_fragment)
    
                
        prev_size = 0
        curr_size = len(past_fragments)
        
        # start iterative matching
        while curr_size > 0 and curr_size != prev_size:
            prev_size = len(past_fragments)
            gone_ids = set() # fragments that have no suc to match or already matched to a suc
            for id, ready in past_fragments.items(): # all fragments in past_fragments are ready to be matched to tail
                best_succ = ready.peek_first_suc() # peek the first successor
                if not best_succ or not best_succ.pre: # if ready has no match or best head already matched to other fragment-> go to the next ready
                    gone_ids.add(id)
                
                else:
                    try: best_pre = best_succ.peek_first_pre() #  peek the first "unmatched" predecessor
                    except: 
                        best_pre = None
                        continue

                    # if best_pre.id == ready.id and best_pre.id not in ready.conflicts_with:
                    if getattr(best_pre, parameters.fragment_attr_name) == getattr(ready, parameters.fragment_attr_name):
                        # stitcher_logger.info("** match tail of {} to head of {}".format(int(best_pre.ID), int(best_succ.ID)), extra = None)
                        # print("** match tail of {} to head of {}".format(int(best_pre.ID), int(best_succ.ID)))
                        # if best_succ.ID//100000 != best_pre.ID//100000:
                        #     print("wrong matching!")
                        try:
                            P.union(getattr(best_pre, parameters.fragment_attr_name), getattr(best_succ, parameters.fragment_attr_name)) # update path cache
                            Fragment.match_tail_head(best_pre, best_succ) # update both fragments
                            gone_ids.add(getattr(ready, parameters.fragment_attr_name)) # 
                        except KeyError: # a fragment is already stitched and written to queue pre-maturally
                            # stitcher_logger.warning("fragment {} is already written to queue".format(best_pre.id))
                            # print("fragment {} is already written to queue".format(best_pre.id))
                            pass

            # bookkeep cleanup
            for gone_id in gone_ids:
                past_fragment = past_fragments.pop(gone_id)
                past_fragment.past = False
                past_fragment.gone = True
                
            curr_size = len(past_fragments)

        if fragment is not None:
            curr_fragments.append(getattr(fragment, parameters.fragment_attr_name))    
        
        # write paths from P if time out is reached
        while True:
            try:  
                root = P.first_node()
                if not root:
                    break
                # print(root.ID, root.tail_time, left)
                
                if root.gone and root.tail_time < left - IDLE_TIME:
                    # print("root's tail time: {:.2f}, current time window: {:.2f}-{:.2f}".format(root.tail_time, left, right))
                    path = P.pop_first_path()  
                    # stitcher_logger.info("** write to db: root {}, last_modified {:.2f}, path length: {}".format(root.ID, root.tail_time,len(path)))
                    stitched_trajectory_queue.put(path, timeout = parameters.stitched_trajectory_queue_put_timeout)
                    # dbw.write_one_trajectory(thread = True, fragment_ids = path)
                else: # break if first in cache is not timed out yet
                    break
            except StopIteration: # break if nothing in cache
                break
       

        
            
            
    
