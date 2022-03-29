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
from utils.stitcher_module import *



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


def stitch_raw_trajectory_fragments(TIME_WIN, THRESH, VARX, VARY, 
                                    curr_fragments, past_fragments, path,
                                    fragment_queue: multiprocessing.Queue,
                                    stitched_trajectory_queue: multiprocessing.Queue,
                                    log_queue: multiprocessing.Queue) -> None:
    """
    :param TIME_WIN:
    :param THRESH:
    :param VARX:
    :param VARY:
    :param curr_fragments:
    :param past_fragments:
    :param path:
    :param fragment_queue:
    :param stitched_trajectory_queue:
    :param log_queue:
    :return:
    """
    
    while True: # keep grabbing fragments from queue TODO: add wait time
        # Get next fragment and wait until one is available if necessary.
        fragment = fragment_queue.get(block=True) # fragment = dr._get_first('last_timestamp') # get the earliest ended fragment
        
        # DO THE PROCESSING ON THE FRAGMENT  
        curr_id = fragment._id # last_fragment = fragment['id']
        fragment = Fragment(curr_id, fragment['timestamp'], fragment['x_position'], fragment['y_position'])
        
        path[curr_id] = curr_id
        right = fragment.t[-1] # right pointer: current end time
        left = right-1
        while left < right: # get all fragments that started but not ended at "right"
            start_fragment = dr._get_first('first_timestamp')
            start_time = start_fragment['first_timestamp']
            if start_fragment['last_timestamp'] >= right:
                heapq.heappush(start_times_heap,  start_time) # TODO: check for processed documents on database side, avoid repeatedly reading
        
        
        
        running_ids = dr._filter(last_timestamp >= right and first_timestamp < right)
        # start_times_heap[0] is the left pointer of the moving window
        try: 
            left = max(0, start_times_heap[0] - TIME_WIN)
        except: left = 0

        # compute fragment statistics (1d motion model)
        fragment._computeStats()

        # print("window size :", right-left)
        # remove out of sight fragments 
        while curr_fragments and curr_fragments[0].t[-1] < left: 
            past_fragment = curr_fragments.popleft()
            past_fragments[past_fragment.id] = past_fragment

        # print("Curr_fragments ", [i.id for i in curr_fragments])
        # print("past_fragments ", past_fragments.keys())
        # compute score from every fragment in curr to fragment, update Cost
        for curr_fragment in curr_fragments:
            cost = _getCost(curr_fragment, fragment, TIME_WIN, VARX, VARY)
            if cost > THRESH:
                curr_fragment._addConflict(fragment)
            elif cost > 0:
                curr_fragment._addSuc(cost, fragment)
                fragment._addPre(cost, curr_fragment)
                        
        prev_size = 0
        curr_size = len(past_fragments)
        while curr_size > 0 and curr_size != prev_size:
            prev_size = len(past_fragments)
            remove_keys = set()
            for _, ready in past_fragments.items(): # all fragments in past_fragments are ready to be matched to tail
                best_head = ready._getFirstSuc()
                if not best_head or not best_head.pre: # if ready has no match or best head already matched to other fragments# go to the next ready
                    # past_fragments.pop(ready.id)
                    remove_keys.add(ready.id)
                
                else:
                    try: best_tail = best_head._getFirstPre()
                    except: best_tail = None
                    if best_head and best_tail and best_tail.id == ready.id and best_tail.id not in ready.conflicts_with:
                        # print("** match tail of {} to head of {}".format(best_tail.id, best_head.id))
                        path[best_head.id] = path[best_tail.id]
                        remove_keys.add(ready.id)
                        Fragment._matchTailHead(best_tail, best_head)

            [past_fragments.pop(key) for key in remove_keys]
            curr_size = len(past_fragments)

        # check if current fragment reaches the boundary, if yes, write its path to queue 
        if (fragment.dir == 1 and fragment.x[-1] > x_bound_max) or (fragment.dir == -1 and fragment.x[-1] < x_bound_min):
            key = fragment.id
            stitched_ids = [key]
            while key != path[key]:
                stitched_ids.append(path[key])
                key = path[key]
            stitched_trajectory_queue.put(stitched_ids)
        else:
            curr_fragments.append(fragment)        
        # running_fragments.pop(fragment.id) # remove fragments that ended
            
        
        
        
            


if __name__ == '__main__':
    print("NO CODE TO RUN")
