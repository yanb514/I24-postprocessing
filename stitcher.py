# -----------------------------
__file__ = 'stitcher.py'
__doc__ = """
Operates the trajectory fragment stitcher, continuously consuming raw trajectory fragments as they are
written to the database.
"""

# -----------------------------
import db_parameters, parameters
from i24_logger.log_writer import logger 
from db_writer import DBWriter 
from collections import deque
from utils.stitcher_module import min_nll_cost
from utils.data_structures import Fragment, PathCache
import time
# import sys

# helper functions
def _first(dict):
    try:
        key,val = next(iter(dict.items()))
        return key, val
    except StopIteration:
        raise StopIteration


def stitch_raw_trajectory_fragments(direction, fragment_queue,
                                    stitched_trajectory_queue):
    """
    fragment_queue is sorted by last_timestamp
    :param direction: "east" or "west"
    :param fragment_queue: fragments sorted in last_timestamp
    :param stitched_trajectory_queue: to store stitched trajectories result
    :param log_queue:
    :return: None
    """
    stitcher_logger = logger
    # Get parameters
    TIME_WIN = parameters.TIME_WIN
    VARX = parameters.VARX
    VARY = parameters.VARY
    THRESH = parameters.THRESH
    IDLE_TIME = parameters.IDLE_TIME
    
    # Initialize some data structures
    curr_fragments = deque()  # fragments that are in current window (left, right), sorted by last_timestamp
    past_fragments = dict()  # set of ids indicate end of fragment ready to be matched, insertion ordered
    P = PathCache() # an LRU cache of Fragment object (see utils.data_structures)

    # Make a database connection for writing
    dbw = DBWriter(host=db_parameters.DEFAULT_HOST, port=db_parameters.DEFAULT_PORT, username=db_parameters.DEFAULT_USERNAME,   
                   password=db_parameters.DEFAULT_PASSWORD,
                   database_name=db_parameters.DB_NAME, server_id=1, process_name=1, process_id=1, session_config_id=1)

    stitcher_logger.info("** Stitching starts. fragment_queue size: {}".format(fragment_queue.qsize()),extra = None)

    # For benchmark run time
    fgmt_count = 0
    t_start = time.time()
    tail_time = []
    process_time = []
    cache_size = []
    while True:
        try:
            fragment = Fragment(fragment_queue.get(block = False)) # make object
            fgmt_count += 1
            tail_time.append(fragment.last_timestamp)
            process_time.append(time.time()-t_start)
            cache_size.append(len(P.path))
            if fgmt_count % 100 == 0:
                stitcher_logger.info("Processed {} fragments, in cache: {}".format(fgmt_count, len(P.path)))
        except:
            return fgmt_count, [t-tail_time[0] for t in tail_time], process_time,cache_size
        
        fragment.curr = True
        P.add_node(fragment)
        
        # specify time window for curr_fragments
        right = fragment.last_timestamp # right pointer: current end time
        left = min(fragment.first_timestamp, right - TIME_WIN)
        
        # compute fragment statistics (1d motion model)
        fragment.compute_stats()
        
        # remove out of sight fragments 
        # if len(curr_fragments) > 0:
        while curr_fragments and P.get_fragment(curr_fragments[0]).last_timestamp < left: 
            past_id = curr_fragments.popleft()
            past_fragment = P.get_fragment(past_id)
            past_fragment.curr = False
            past_fragment.past = True
            past_fragments[past_id] = past_fragment # could use keys only: past_fragments[past_id] = None if memory is an issue 
        # print("Past_fragments size: ", len(past_fragments))
        
        # compute score from every fragment in curr to fragment, update Cost
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
                    if best_pre.id == ready.id:
                        # stitcher_logger.info("** match tail of {} to head of {}".format(int(best_pre.ID), int(best_succ.ID)), extra = None)
                        # print("** match tail of {} to head of {}".format(int(best_pre.ID), int(best_succ.ID)))
                        # if best_succ.ID//100000 != best_pre.ID//100000:
                        #     print("wrong matching!")
                        try:
                            P.union(best_pre.id, best_succ.id) # update path cache
                            Fragment.match_tail_head(best_pre, best_succ) # update both fragments
                            gone_ids.add(ready.id) # 
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

        curr_fragments.append(fragment.id)    
        
        # write paths from P if time out is reached
        # This block can be written as an async operation
        while True:
            try:
                root = P.first_node()
                # print(root.ID, root.gone, root.tail_time, left)
                if root.gone and root.tail_time < left - IDLE_TIME:
                    # print("root's tail time: {:.2f}, current time window: {:.2f}-{:.2f}".format(root.tail_time, left, right))
                    path = P.pop_first_path()  
                    # print("write to db: root {}, last_modified {:.2f}, path length: {}".format(root.ID, root.tail_time,len(path)))
                    stitcher_logger.info("** write to db: root {}, last_modified {:.2f}, path length: {}".format(root.ID, root.tail_time,len(path)))
                    stitched_trajectory_queue.put(path) # doesn't know ObjectId
                    dbw.write_stitched_trajectory(thread = True, fragment_ids = path)
                else: # break if first in cache is not timed out yet
                    break
            except StopIteration: # break if nothing in cache
                break
       

        
            
            
    
