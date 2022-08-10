#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:55:04 2022

@author: yanbing_wang
6/30
Live data read should be only one process, and distribute to 2 queues (east/west) based on document direction
two seperate live_data_feed processes will mess up the change stream
"""
from i24_database_api import DBClient
import i24_logger.log_writer as log_writer
from i24_logger.log_writer import catch_critical
import time
import signal
import sys
import os
import heapq
import random
import numpy as np
from sklearn import linear_model
import threading

verbs = ["medicates", "taunts", "sweettalks", "initiates", "harasses", "smacks", "boggles", "negotiates", "castigates", "disputes", "cajoles", "improvises",
         "surrenders", "escalates", "mumbles", "juxtaposes", "excites", "lionizes", "ruptures", "yawns","administers","flatters","foreshadows","buckles"]
max_trials = 10

    
class SIGINTException(Exception):
    pass

class SignalHandler():
    '''
    Signal handling: in live/static data read, SIGINT and SIGUSR1 are handled in the same way
    Soft terminate current process. Close ports and exit.
    '''
    run = True
    # count = 0 # count the number of times a signal is received
    # signal.signal(signal.SIGINT, signal.SIG_IGN)  
    
    def __init__(self):
        signal.signal(signal.SIGINT, self.shut_down)
        signal.signal(signal.SIGUSR1, self.shut_down)
        signal.signal(signal.SIGPIPE,signal.SIG_DFL) # reset SIGPIPE so that no BrokePipeError when SIGINT is received
    
    def shut_down(self, *args):
        self.run = False
        raise SIGINTException
        # self.count += 1
        # logger.info("{} detected {} times".format(signal.Signals(args[0]).name, self.count))
        
 
def initialize_db(parameters):
    '''
    initialize the postprocessing pipeline (called by manager)
    1. get the latest raw collection if parameters["raw_collection"] == ""
    2. create a new stitched collection
    3. create a new reconciled collection
    '''
    
    # get the latest collection if raw_collection is empty
    dbc = DBClient(**parameters["db_param"], database_name = parameters["raw_database"], 
                          collection_name = parameters["raw_collection"], latest_collection=True)
    parameters["raw_collection"] = dbc.collection_name # raw_collection should NOT be empty
      
    rec_db = dbc.client[parameters["reconciled_database"]]
    existing_cols = rec_db.list_collection_names()

    connected = False
    while not connected:
        verb = random.choice(verbs)
        reconciled_name = parameters["raw_collection"]+"__"+verb
        
        if reconciled_name not in existing_cols:
            parameters["stitched_collection"] = reconciled_name
            parameters["reconciled_collection"] = reconciled_name
            print("** initialized reconcield name, ", reconciled_name)
            connected = True

    # save metadata
    dbc.client[parameters["meta_database"]]["metadata"].insert_one(document = {"collection_name": reconciled_name, "parameters": parameters._getvalue()},
                                  bypass_document_validation=True)
    
    del dbc
    return
    
def thread_update_one(raw, _id, filter, fitx, fity):
    raw.update_one({"_id": _id}, {"$set": {"filter": list(filter),
                                                "fitx": list(fitx),
                                                "fity": list(fity)}}, upsert = True)


    
@catch_critical(errors = (Exception))
def add_filter(traj, raw, residual_threshold_x, residual_threshold_y, 
               conf_threshold, remain_threshold):
    '''
    add a filter to trajectories based on
    - RANSAC fit on x and
    - bad detection confidence
    get total mask (both lowconf and outlier)
    apply ransac again on y-axis
    save fitx, fity and tot_mask
    and save filter field to raw collection
    '''
    filter = True
    t = np.array(traj["timestamp"])
    x = np.array(traj["x_position"])
    y = np.array(traj["y_position"])
    conf = np.array(traj["detection_confidence"])
        
    length = len(t)
    
    # get confidence mask
    lowconf_mask = np.array(conf < conf_threshold)
    highconf_mask = np.logical_not(lowconf_mask)
    
    # fit x only on highconf
    ransacx = linear_model.RANSACRegressor(residual_threshold=residual_threshold_x)
    X = t.reshape(1, -1).T
    ransacx.fit(X[highconf_mask], x[highconf_mask])
    fitx = [ransacx.estimator_.coef_[0], ransacx.estimator_.intercept_]
    inlier_mask = ransacx.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask) # mask if True
    
    # total mask (filtered by both outlier and by low confidence)
    mask1 = np.arange(length)[lowconf_mask] # all the bad indices
    mask2 = np.arange(length)[highconf_mask][outlier_mask]
    bad_idx = np.concatenate((mask1, mask2))
    remain = length-len(bad_idx)
    # print("bad rate: {}".format(bad_ratio))
    if remain < remain_threshold:
        filter = []
  
    # fit y only on mask
    ransacy = linear_model.RANSACRegressor(residual_threshold=residual_threshold_y)
    ransacy.fit(X[highconf_mask][inlier_mask], y[highconf_mask][inlier_mask])
    fity = [ransacy.estimator_.coef_[0], ransacy.estimator_.intercept_]
    
    # save to raw collection
    if filter:
        filter = length*[1]
        for i in bad_idx:         
            filter[i]=0


    # save filter to database- non-blocking
    _id = traj["_id"]
    thread = threading.Thread(target=thread_update_one, args=(raw, _id, filter, fitx, fity,))
    thread.start()


    # update traj document
    traj["filter"] = filter
    traj["fitx"] = fitx
    traj["fity"] = fity
    return traj
        
def change_stream_simulator(default_param, insert_rate):
    '''
    When no live-streaming data, simulate one by reading from a collection (default_param.raw_collection), and insert to a new collection (write_to_collection)
    insert_rate: insert no. of documents per second
    query in ascending first_timestamp, such that the write_to_collection is slightly out of order in terms of last_timestamp
    '''
    sig_hdlr = SignalHandler()  
    
    # initialize the logger
    logger = log_writer.logger
    logger.set_name("change_stream_simulator")
    setattr(logger, "_default_logger_extra",  {})
    
    # reset the new collection
    write_to_collection = default_param.raw_collection + "_simulated"
    raw_schema_path = "config/" + default_param.raw_schema_path
    dbw = DBClient(**default_param["db_param"], database_name =default_param.database_name, collection_name = write_to_collection, schema_file = raw_schema_path)
    dbw.reset_collection()
    logger.info("DBWriter initiated")
    
    # initiate data reader
    dbr = DBClient(**default_param["db_param"], database_name =default_param.database_name, collection_name=default_param.raw_collection)
    logger.info("DBReader initiated")
    
    start = dbr.get_min("first_timestamp") - 1e-6
    # end = start + 3
    end = dbr.get_max("first_timestamp") + 1e-6
    cur = dbr.get_range("first_timestamp", start, end)
    
    # write to simulated collection
    count = 0
    # time.sleep(3) # wait for change stream to get initialized
    try:
        for doc in cur:
            time.sleep(1/insert_rate)
            # print("insert: {}".format(doc["first_timestamp"]))
            doc.pop("_id")
            dbw.write_one_trajectory(thread = False, **doc)
            count += 1
            if count % 100 == 0:
                logger.info("{} docs written to dbw".format(count))
                time.sleep(2) 
    except SIGINTException:
        logger.info("SIGINT/SIGUSR1 received in change_stream_simulator.")
        
    except Exception as e:
        logger.warning("Other exceptions occured. Exit. Exception:{}".format(e))    
    # exit
    logger.info(f"Finished writing {count} to simulated collection. Exit")
    
    del dbr, dbw
    sys.exit(0)
    
   
    
def live_data_reader(default_param, east_queue, west_queue, t_buffer = 1, read_from_simulation = True):
    """
    Monitor the insert change of a database collection, insert to a heap so that the documents can be sorted in last_timestamp
    Pop the heap to queue when a buffer time is reached
    :param default_param: default parameter object / dictionary for database collection and live_data_read related parameters, including
        :raw_collection: collection name to read from. If collection is not streaming (static), then use static_data_reader instead
        :host: Database connection host name.
        :port: Database connection port number.
        :username: Database authentication username.
        :password: Database authentication password.
        :database_name: Name of database to connect to (do not confuse with collection name).
    :param east_queue/west_queue: Process-safe queues to which records that are "ready" are written.  multiprocessing.Queue
    :param t_buffer: safe buffer time in sec such that a doc is ready to be written to queue if doc["last_timestamp"] < current_change_time - t_buffer
    :param read_from_simulation: bool. Set to true to read from a simulated streaming collection ([raw_collection]_simulated), otherwise read from [raw_collection]
    :return:
    """
    # running_mode = os.environ["my_config_section"]
    logger = log_writer.logger
    logger.set_name("live_data_reader")
    setattr(logger, "_default_logger_extra",  {})
    
    # Connect to a database reader
    if read_from_simulation:
        # time.sleep(4) # wait for change_stream_simulator to start
        raw_collection = default_param.raw_collection + "_simulated"
    else:
        raw_collection = default_param.raw_collection
    dbr = DBReader(default_param, collection_name=raw_collection)
            
    sig_hdlr = SignalHandler()  
    pipeline = [{'$match': {'operationType': 'insert'}}]
            
        
    heap = [] # keep a heap sorted by last_timestamp
    discard = 0
    resume_token = None
    change_stream_timeout = 10 # close the stream if timeout is reached
    idle_time = 0
    last_change_time = time.time() # initialize last checkpoint
    
    # have an internal time out for changes

    with dbr.collection.watch(pipeline, resume_after = None) as stream:
        # close the stream if SIGINT received
        
        try:
            while stream.alive and sig_hdlr.run: # change stream is still alive even when there's no changes
                change = stream.try_next() # first change 
                # Note that the ChangeStream's resume token may be updated
                # even when no changes are returned.
                if change is not None:
                    last_change_time = time.time()
                    idle_time = 0 # reset idle_time
                    # resume_token = stream.resume_token
                    
                    # print("Change document: %r" % (change['fullDocument']['first_timestamp'],))
                    # push to heap
                    safe_query_time = change["fullDocument"]['first_timestamp']-t_buffer
                    heapq.heappush(heap,(change["fullDocument"]['last_timestamp'],change["fullDocument"]['_id'],change['fullDocument']))
                    
                    # check if heap[0] is ready, pop until it's not ready
                    while heap and heap[0][0] < safe_query_time:
                        _, _,doc = heapq.heappop(heap)
                        # print("pop: {},".format(doc["last_timestamp"]))
                        if len(doc["timestamp"]) > 3: 
                            if doc["direction"] == 1:
                                # logger.debug("write a doc to east queue, dir={}".format(doc["direction"]))
                                east_queue.put(doc)
                            else:
                                west_queue.put(doc)
                        else:
                            discard += 1
                    continue
                # We end up here when there are no recent changes.
                # Sleep for a while before trying again to avoid flooding
                # the server with getMore requests when no changes are
                # available.
                else:
                    time.sleep(2)
                    idle_time += time.time() - last_change_time
                    # print("idle time: ", idle_time)
                    if idle_time > change_stream_timeout:
                        # print("change stream timeout reached. Close the stream")
                        stream.close()
                        break
        
        except SIGINTException:
            logger.info("SIGINT/SIGINT received. Close stream")
            try: 
                stream.close() 
            except: pass
            
        except Exception as e:
            logger.warning("Other exceptions occured. Exit. Exception:{}".format(e))
            try: 
                stream.close() 
            except: pass
            
        # out of while loop
        logger.info("stream is no longer alive or SIGINT/SIGINT received")
        try:
            del dbr 
        except:
            pass
        logger.info("Process the rest of heap")


    #pop all the rest of heap
    while heap:
        _, _, doc = heapq.heappop(heap)
        # print("pop: {}".format(doc["last_timestamp"]))
        if len(doc["timestamp"]) > 3: 
            if doc["direction"] == 1:
                # logger.debug("write a doc to east queue, dir={}".format(doc["direction"]))
                east_queue.put(doc)
            else:
                west_queue.put(doc)
        else:
            discard += 1
            
    
    logger.info("Discarded {} short tracks".format(discard))
    logger.info("DBReader closed. Exiting live_data_reader while loop.")
    sys.exit() # for linux
    
    
    
    
def static_data_reader(default_param, east_queue, west_queue, min_queue_size = 1000):
    """
    Read data from a static collection, sort by last_timestamp and write to queues
    :param host: Database connection host name.
    :param port: Database connection port number.
    :param username: Database authentication username.
    :param password: Database authentication password.
    :param database_name: Name of database to connect to (do not confuse with collection name).
    :param collection_name: Name of database collection from which to query.
    :param ready_queue: Process-safe queue to which records that are "ready" are written.  multiprocessing.Queue
    :return:
    """
    # running_mode = os.environ["my_config_section"]
    logger = log_writer.logger
    logger.set_name("static_data_reader")
    setattr(logger, "_default_logger_extra",  {})
    
    # Connect to a database reader
    while default_param["raw_collection"] == "":
        time.sleep(1)
     
    # get parameters for fitting
    RES_THRESH_X = default_param["residual_threshold_x"]
    RES_THRESH_Y = default_param["residual_threshold_y"]
    CONF_THRESH = default_param["conf_threshold"],
    REMAIN_THRESH = default_param["remain_threshold"]
    
    dbr = DBClient(**default_param["db_param"], database_name = default_param["raw_database"], collection_name=default_param["raw_collection"])  
    # default_param["raw_collection"] = dbr.collection_name
    # print("default param raw ", default_param["raw_collection"])
    
    
    # start from min and end at max if collection is static
    rri = dbr.read_query_range(range_parameter='last_timestamp', range_increment=default_param["range_increment"], query_sort= [("last_timestamp", "ASC")])

    # Signal handling: in live data read, SIGINT and SIGUSR1 are handled in the same way    
    sig_hdlr = SignalHandler()  
    
    
    # for debug only
    # rri._reader.range_iter_stop = rri._reader.range_iter_start + 60

    discard = 0 # counter for short (<3) tracks
    
    while sig_hdlr.run: 
        logger.debug("* current lower: {}, upper: {}, start: {}, stop: {}".format(rri._current_lower_value, rri._current_upper_value, rri._reader.range_iter_start, rri._reader.range_iter_stop))
        
        try:
            # keep filling the queues so that they are not low in stock
            while east_queue.qsize() <= min_queue_size or west_queue.qsize() <= min_queue_size:
           
                next_batch = next(rri)  
    
                for doc in next_batch:
                    if len(doc["timestamp"]) > 3: 
                        doc = add_filter(doc, dbr.collection, RES_THRESH_X, RES_THRESH_Y, 
                                       CONF_THRESH, REMAIN_THRESH)
                        
                        if doc["direction"] == 1:
                            # logger.debug("write a doc to east queue, dir={}".format(doc["direction"]))
                            east_queue.put(doc)
                        else:
                            west_queue.put(doc)
                    else:
                        discard += 1
                        # logger.info("Discard a fragment with length less than 3")
    
                logger.info("** qsize for raw_data_queue: east {}, west {}".format(east_queue.qsize(), west_queue.qsize()))
                
                            
            # if queue has sufficient number of items, then wait before the next iteration (throttle)
            logger.info("** queue size is sufficient. wait")     
            time.sleep(2)
          
         
            
        except StopIteration:  # rri reaches the end
            logger.warning("static_data_reader reaches the end of query range iteration. Exit")
            break
        
        except SIGINTException:  # rri reaches the end
            logger.warning("SIGINT/SIGUSR1 detected. Checkpoint not implemented.")
            break
        
        except Exception as e:
            logger.warning("Other exceptions occured. Exit. Exception:{}".format(e))
            break

        
    
    # logger.info("outside of while loop:qsize for raw_data_queue: east {}, west {}".format(east_queue.qsize(), west_queue.qsize()))
    logger.info("Discarded {} short tracks".format(discard))
    del dbr 
    logger.info("DBReader closed. Exiting live_data_reader while loop.")
    sys.exit() # for linux

    
    
    