# -----------------------------
__file__ = 'stitcher.py'
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

import sys

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
    # Initiate a logger
    # stitcher_logger = I24Logger(owner_process_name = "stitcher_"+direction, connect_file=True, file_log_level='DEBUG', 
    #                     connect_console=True, console_log_level='INFO')
    
    # Get parameters
    # print("in stitcher")
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
    # print("before dbw")
    dbw = DBWriter(host=db_parameters.DEFAULT_HOST, port=db_parameters.DEFAULT_PORT, username=db_parameters.DEFAULT_USERNAME,   
                   password=db_parameters.DEFAULT_PASSWORD,
                   database_name=db_parameters.DB_NAME, server_id=1, process_name=1, process_id=1, session_config_id=1)
    # print("after dbw")
    # stitcher_logger.info("** Stitching starts. fragment_queue size: {}".format(fragment_queue.qsize()),extra = None)

    
    while True:
        sys.stdout.flush()
        # print(("fragment_queue size: {}".format(fragment_queue.qsize())))
        # print("*** getting fragment")  
        # try:
        # stitcher_logger.info("fragment_queue size: {}".format(fragment_queue.qsize()))
        fragment = Fragment(fragment_queue.get(block = True)) # make object
        # time.sleep(1)
        # except: # handle queue empty case
            # stitcher_logger.warning("fragment_queue is empty", extra={})
            # break
            # pass
        # print("\n")
        # print("current fragment: ", fragment.last_timestamp)
        # print(("fragment_queue size: {}".format(fragment_queue.qsize())))
        fragment.curr = True
        P.add_node(fragment)
        
        # specify time window for curr_fragments
        right = fragment.last_timestamp # right pointer: current end time
        left = min(fragment.first_timestamp, right - TIME_WIN)
        # print("left, right: ", left, right)
        
        # compute fragment statistics (1d motion model)
        fragment.compute_stats()
        # print("Curr_fragments size: ", len(curr_fragments))
        
        # remove out of sight fragments 
        if curr_fragments:
            while P.get_fragment(curr_fragments[0]).last_timestamp < left: 
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
            #     curr_fragment.add_conflict(fragment)
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
                    # past_fragments.pop(ready.id)
                    gone_ids.add(id)
                
                else:
                    try: best_pre = best_succ.peek_first_pre() #  peek the first "unmatched" predecessor
                    except: 
                        best_pre = None
                        continue

                    # if best_pre.id == ready.id and best_pre.id not in ready.conflicts_with:
                    if best_pre.id == ready.id:
                        # stitcher_logger.info("** match tail of {} to head of {}".format(best_tail.ID, best_head.ID), extra = None)
                        print("** match tail of {} to head of {}".format(int(best_pre.ID), int(best_succ.ID)))
                        if best_succ.ID//100000 != best_pre.ID//100000:
                            print("wrong matching!")
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
                    print("write to db: root {}, last_modified {:.2f}, path length: {}".format(root.ID, root.tail_time,len(path)))
                    # stitcher_logger.info("write to db: root {}, last_modified {:.2f}, path length: {}".format(root.ID, root.tail_time,len(path)))
                    stitched_trajectory_queue.put(path) # doesn't know ObjectId
                    dbw.write_stitched_trajectory(path)
                else: # break if first in cache is not timed out yet
                    break
            except StopIteration: # break if nothing in cache
                break
            
        # num_roots = P.count() # number of roots
        # num_cache = len(P.path)
        # print("Number of roots: ", num_roots,num_cache)
    
    
    # print("Total stitched trajectories: ", num_roots)
    # P.print_cache()
    # all_paths = P.get_all_paths("ID")
    # print(all_paths)


        
            
            
    
