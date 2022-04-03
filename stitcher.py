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

from mongodb_handler import DataReader
from utils.stitcher_module import _getCost
import heapq
from utils.data_structures import Fragment



def get_raw_fragments_naive(BOOKMARK, lock, fragment_queue_e: multiprocessing.Queue,fragment_queue_w: multiprocessing.Queue, log_queue: multiprocessing.Queue) -> None:
    """
    :param BOOKMARK: timestamp such that the next X framgnets are read and put to queue #TODO: keep a bookmark "cursor", and only get the next X fragments
    :param fragment_queue:
    :param log_queue:
    :return:
    """
    # no change stream
    # try getting the first fragment
    print("** get raw fragment starts...")
    try:
        dr = DataReader(**parameters.DB_PARAMS, vis=False)
        print('database connected')
    
    except pymongo.errors.ConnectionFailure:
        log_queue.put((logging.ERROR, "MongoDB connection failed"))
            
#    while True:
    
    if not fragment_queue_e.empty and not fragment_queue_w.empty: # if both queues are not empty
        return
    else: # if either queue is empty, refill
        print("Raw fragment ingester started.")
        log_queue.put((logging.INFO, "Raw fragment ingester started."))
        
        # Grab the next batch of fragments whose last_timestamps fall between BOOKMARK and BOOKMARK + 50
        print("*********",BOOKMARK.value,BOOKMARK.value+parameters.TIME_RANGE)
        docs = dr._get_range("raw", "last_timestamp", BOOKMARK.value,BOOKMARK.value+parameters.TIME_RANGE)
        print(len(list(docs)))
        
        for doc in list(docs):
            print("*** insert to queue")
            if doc['direction'] == 1:
                fragment_queue_e.put(doc)
            else:
                fragment_queue_w.put(doc)
                
        with lock:
            
            BOOKMARK.value += parameters.TIME_RANGE
        

        print("Final queue size (east/west)", fragment_queue_e.qsize(), "/", fragment_queue_e.qsize())
        
            
                
def get_raw_fragments(fragment_queue: multiprocessing.Queue, log_queue: multiprocessing.Queue) -> None:
    """
    read from database with watch
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
            dr = DataReader(**parameters.DB_PARAMS, vis=False)
            
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
                with dr.collection.watch(watch_for, resume_after=resume_token) as stream:
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


def stitch_raw_trajectory_fragments(PARAMS, 
                                    INIT,
                                    fragment_queue,
                                    stitched_trajectory_queue,
                                    log_queue):
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
    # unpack variables
    TIME_WIN = PARAMS['TIME_WIN']
    VARX = PARAMS['VARX']
    VARY = PARAMS['VARY']
    THRESH = PARAMS['THRESH']
    X_MAX = PARAMS['X_MAX']
    X_MIN = PARAMS['X_MIN']
    
    curr_fragments = INIT['curr_fragments']
    past_fragments = INIT['past_fragments']
    path = INIT['path']
    start_times_heap = INIT['start_times_heap']
    
    print("** Stitching starts. fragment_queue size: ", fragment_queue.qsize())
    while True: # keep grabbing fragments from queue TODO: add wait time
        # Get next fragment and wait until one is available if necessary.
        print("*** in while loop")
        fragment = fragment_queue.get(block=True) # fragment = dr._get_first('last_timestamp') # get the earliest ended fragment
        # DO THE PROCESSING ON THE FRAGMENT  
        curr_id = fragment['_id'] # last_fragment = fragment['id']
        fragment = Fragment(curr_id, fragment['timestamp'], fragment['x_position'], fragment['y_position'])
        path[curr_id] = curr_id
        right = fragment.t[-1] # right pointer: current end time
        left = right-1
        while left < right: # get all fragments that started but not ended at "right"
            start_fragment = fragment_queue.get(block=True) # dr._get_first('first_timestamp')
            start_time = start_fragment['first_timestamp']
            if start_fragment['last_timestamp'] >= right:
                heapq.heappush(start_times_heap,  start_time) # TODO: check for processed documents on database side, avoid repeatedly reading    
        print("start_Times_heap size:",len(start_times_heap))
        # start_times_heap[0] is the left pointer of the moving window
        try: 
            left = max(0, start_times_heap[0] - TIME_WIN)
        except: left = 0
        print("left, right: ", left, right)
        # compute fragment statistics (1d motion model)
        fragment._computeStats()

        # print("window size :", right-left)
        # remove out of sight fragments 
        while curr_fragments and curr_fragments[0].t[-1] < left: 
            past_fragment = curr_fragments.popleft()
            past_fragments[past_fragment.id] = past_fragment
        
        print("past_fragments size:",len(past_fragments))
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
        if (fragment.dir == 1 and fragment.x[-1] > X_MAX) or (fragment.dir == -1 and fragment.x[-1] < X_MIN):
            key = fragment.id
            stitched_ids = [key]
            while key != path[key]:
                stitched_ids.append(path[key])
                key = path[key]
            stitched_trajectory_queue.put(stitched_ids)
            print("Stitched: ", len(stitched_trajectory_queue))
        else:
            curr_fragments.append(fragment)        
        # running_fragments.pop(fragment.id) # remove fragments that ended
            
        
        
        
            


if __name__ == '__main__':
    print("NO CODE TO RUN")
