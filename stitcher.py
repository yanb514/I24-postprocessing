# -----------------------------
__file__ = 'stitcher.py'
__doc__ = """
Operates the trajectory fragment stitcher, continuously consuming raw trajectory fragments as they are
written to the database.
"""

# -----------------------------
import multiprocessing
import parameters
import logging
# TODO: check and test database implementation
import pymongo
import pymongo.errors
from data_handler import DataReader, DataWriter

from utils.stitcher_module import _getCost
from utils.data_structures import Fragment

# for debugging
from time import sleep
import queue


def get_raw_fragments_naive(BOOKMARK, lock, fragment_queue: multiprocessing.Queue, log_queue: multiprocessing.Queue) -> None:
    """
    no change stream
    read fragments in batch (static database)
    put (start_timestamp, _id) into start_time_heap
    put (end_timestamp, _id) into end_time_heap
    :param BOOKMARK: timestamp such that the next X framgnets are read and put to queue #TODO: keep a bookmark "cursor", and only get the next X fragments
    :param fragment_queue:
    :param log_queue:
    :return:
    """
    # no change stream
    # try getting the first fragment
    print("** get raw fragment starts...")
    try:
        dr = DataReader(parameters.RAW_COLLECTION)
        print('database connected')
    
    except pymongo.errors.ConnectionFailure:
        log_queue.put((logging.ERROR, "MongoDB connection failed"))
     
    while True:
#        print("check if queue empty", fragment_queue.empty())
        if fragment_queue.empty() and BOOKMARK.value < parameters.END: # start refilling the queue
    
            print("Raw fragment ingester started.")
            log_queue.put((logging.INFO, "Raw fragment ingester started."))
            
            # Grab the next batch of fragments whose last_timestamps fall between BOOKMARK and BOOKMARK + 50
            print("********* reading from time ",BOOKMARK.value,BOOKMARK.value+parameters.TIME_RANGE)
            docs = dr.get_range("last_timestamp", BOOKMARK.value,BOOKMARK.value+parameters.TIME_RANGE)
            docs = list(docs)

            print(len(docs), "documents writing to queue")
            
            for doc in docs:
                fragment_queue.put(doc)
                    
            with lock:  
                BOOKMARK.value += parameters.TIME_RANGE
            
            print("Current queue size ", fragment_queue.qsize())
            
                
def get_raw_fragments(fragment_queue: multiprocessing.Queue, log_queue: multiprocessing.Queue) -> None:
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


def stitch_raw_trajectory_fragments(PARAMS, 
                                    INIT,
                                    fragment_queue,
                                    stitched_trajectory_queue,
                                    log_queue):
    """
    fragment_queue is sorted by last_timestamp
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
    
    # TODO: take these data structure as input of the function
    curr_fragments = INIT['curr_fragments'] # fragments that are in current window (left, right)
    past_fragments = INIT['past_fragments'] # fragments whose tails are ready to be matched
    path = INIT['path'] # assignment list
#    start_times_heap = INIT['start_times_heap']
    
    # Make database connection for writing
    dw = DataWriter(parameters.STITCHED_COLLECTION)
    
    print("** Stitching starts. fragment_queue size: ", fragment_queue.qsize())
    while True: # keep grabbing fragments from queue TODO: add wait time

        current_fragment = fragment_queue.get(block=True) # fragment = dh._get_first('last_timestamp') # get the earliest ended fragment
        print("*** getting fragment")
        # DO THE PROCESSING ON THE FRAGMENT  
        curr_id = current_fragment['_id'] # last_fragment = fragment['id']
        fragment = Fragment(current_fragment)
        path[curr_id] = curr_id
        right = fragment.t[-1] # right pointer: current end time
        left = right - TIME_WIN
        print("left, right: ", left, right)
        
        # compute fragment statistics (1d motion model)
        fragment._computeStats()

        # remove out of sight fragments 
        while curr_fragments and curr_fragments[0].t[-1] < left: 
            past_fragment = curr_fragments.popleft()
            past_fragments[past_fragment.id] = past_fragment
        

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
        
        # start iterative matching
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
        # TODO: keep a cache for "cold" fragments? Yes LRU style
        if (fragment.dir == 1 and fragment.x[-1] > X_MAX) or (fragment.dir == -1 and fragment.x[-1] < X_MIN):
            key = fragment.id
            stitched_ids = [key]
            while key != path[key]:
                stitched_ids.append(path[key])
                key = path[key]
            for id in stitched_ids:
                path.remove(id)
            # put to queue and write to database print(record.get('_id').generation_time)
            doc = {
                    "fragment_ids": stitched_ids,
                    }
            stitched_trajectory_queue.put(doc)
            dw.insert(doc)
            print("Stitched: ", stitched_trajectory_queue.qsize())
        else:
            curr_fragments.append(fragment)        
        # running_fragments.pop(fragment.id) # remove fragments that ended
            
        
        
        
            


if __name__ == '__main__':
    print("run main")
    q = queue.Queue()
    print("got queue")
    dr = DataReader(parameters.RAW_COLLECTION)
    print("connected to database")
    docs = dr.get_range("last_timestamp", 10,11)
    print("get range")
    docs = list(docs)
    print(len(docs))
#    for doc in docs:
#        q.put(doc)
#    print(q.qsize())
    
