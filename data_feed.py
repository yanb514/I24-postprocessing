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

import time
import signal
import sys
import os
import heapq
import random
import utils.misc as misc

verbs = ["medicates", "taunts", "sweettalks", "initiates", "harasses", "negotiates", "castigates", "disputes", "cajoles", "improvises",
         "surrenders", "escalates", "mumbles", "juxtaposes", "excites", "lionizes", "ruptures", "yawns","administers","flatters","foreshadows","buckles",
         "moans", "gasps", "yells", "praises", "impersonates", "giggles", "roars", "articulates", "capitalizes", "calibrates", "protests", "conforms"]
max_trials = 10

    
class SIGINTException(SystemExit):
    pass

def soft_stop_hdlr(sig, action):
    '''
    Signal handling: in live/static data read, SIGINT and SIGUSR1 are handled in the same way
    Soft terminate current process. Close ports and exit.
    '''
    raise SIGINTException # so to exit the while true loop
        
 
def initialize(parameters, db_param):
    '''
    initialize the postprocessing pipeline (called by manager)
    1. get the latest raw collection if parameters["raw_collection"] == ""
    2. create a new stitched collection
    3. create a new reconciled collection
    4. list all unique compute nodes and update "compute_node_list"
    '''
    
    # get the latest collection if raw_collection is empty
    dbc = DBClient(**db_param, database_name = parameters["raw_database"], 
                          collection_name = parameters["raw_collection"], latest_collection=True)
    parameters["raw_collection"] = dbc.collection_name # raw_collection should NOT be empty
      
    # get all the unique values for compute_node_id field
    dbc.collection.create_index("compute_node_id")
    rec_db = dbc.client[parameters["reconciled_database"]]
    existing_cols = rec_db.list_collection_names()
    
    if not parameters["reconciled_collection"]:
        connected = False
        while not connected:
            verb = random.choice(verbs)
            reconciled_name = parameters["raw_collection"]+"__"+verb
            
            if reconciled_name not in existing_cols:
                parameters["stitched_collection"] = reconciled_name
                parameters["reconciled_collection"] = reconciled_name
                print("** initialized reconcield name, ", reconciled_name)
                connected = True

    else:
        reconciled_name = parameters["reconciled_collection"]
        print("reconciled_collection name is already provided: ", parameters["reconciled_collection"])
    
    if not parameters["compute_node_list"]:
        compute_node_list = dbc.collection.distinct("compute_node_id")
        # sort compute_node_list based on starting_x
        reference_x = [dbc.collection.find_one({"compute_node_id": node})["starting_x"] for node in compute_node_list]
        _, sorted_compute_node_list = zip(*sorted(zip(reference_x, compute_node_list)))
        parameters["compute_node_list"] = sorted_compute_node_list
        
    # get collection t_min and t_max (for static only)
    t_min = dbc.get_min("last_timestamp")
    t_max = dbc.get_max("last_timestamp")
    parameters["t_min"] = t_min
    parameters["t_max"] = t_max
    
    # save metadata
    dbc.client[parameters["meta_database"]]["metadata"].insert_one(document = {"collection_name": reconciled_name, "parameters": parameters._getvalue()},
                                  bypass_document_validation=True)
    
    # drop the temp collection
    if parameters["reconciled_collection"] in dbc.client[parameters["temp_database"]].list_collection_names():
        inp = input("About to drop {} from temp database, proceed? [y/n]".format(parameters["reconciled_collection"]))
        if inp in ["y", "Y"]:
            dbc.client[parameters["temp_database"]][parameters["reconciled_collection"]].drop()

    del dbc
    return


def initialize_master(parameters, db_param):
    """
    same as initialize(), but called only during master process
    1. read raw trajectories from temp collection, specified in parameters["temp_collection"]
    """
    if not parameters["reconciled_collection"]:
        raise Exception("reconciled_collection has to be specified before calling initialize_master")
        
    # initialize compute_node_id
    if "temp_collection" in parameters and parameters["temp_collection"]:
        read_collection = parameters["temp_collection"]
    else:
        read_collection = parameters["reconciled_collection"]

    dbc = DBClient(**db_param, database_name = parameters["temp_database"], 
                          collection_name = read_collection, latest_collection=True)
    dbc.collection.create_index("compute_node_id")
    if not parameters["compute_node_list"]:
        compute_node_list = dbc.collection.distinct("compute_node_id")
        reference_x = [dbc.collection.find_one({"compute_node_id": node})["starting_x"] for node in compute_node_list]
        _, sorted_compute_node_list = zip(*sorted(zip(reference_x, compute_node_list)))
        parameters["compute_node_list"] = sorted_compute_node_list
    print(f"************** initialize_master {parameters['compute_node_list']}")
    
    # get time range
    t_min = dbc.get_min("last_timestamp")
    t_max = dbc.get_max("last_timestamp")
    parameters["t_min"] = t_min
    parameters["t_max"] = t_max
    
    return

def thread_update_one(raw, _id, filter, fitx, fity):
    filter = [1 if i else 0 for i in filter]
    raw.update_one({"_id": _id}, {"$set": {"filter": filter,
                                            "fitx": list(fitx),
                                            "fity": list(fity)}}, upsert = True)
    
    
    
def static_data_reader(default_param, db_param, raw_queue, query_filter, name=None):
    """
    Read data from a static collection, sort by last_timestamp and write to queues
    :param default_param
        :param host: Database connection host name.
        :param port: Database connection port number.
        :param username: Database authentication username.
        :param password: Database authentication password.
        :param database_name: Name of database to connect to (do not confuse with collection name).
        :param collection_name: Name of database collection from which to query.
    :param raw_queue: Process-safe queue to which records that are "ready" are written.  multiprocessing.Queue
    :param dir: "eb" or "wb"
    :param: node: (str) compute_node_id for videonode
    :return:
    """
    # Signal handling: in live data read, SIGINT and SIGUSR1 are handled in the same way    
    signal.signal(signal.SIGINT, soft_stop_hdlr)
    
    # running_mode = os.environ["my_config_section"]
    logger = log_writer.logger
    if name is None:
        name = "static_data_reader"
    logger.set_name(name)
        
    setattr(logger, "_default_logger_extra",  {})
    
    # Connect to a database reader
    # -- if mode is master, read from a temp database which stores all local stitched results
    if "master" in name:
        while default_param["reconciled_collection"] == "":
            time.sleep(1)
        read_database = default_param["temp_database"]
        # initialize read/write collections
        if "temp_collection" in default_param and default_param["temp_collection"]:
            read_collection = default_param["temp_collection"]
        else:
            read_collection = default_param["reconciled_collection"]
    else:
        while default_param["raw_collection"] == "":
            time.sleep(1)
        read_database = default_param["raw_database"]
        read_collection = default_param["raw_collection"]
     
    # get parameters for fitting
    logger.info("{} starts reading from DATABASE {} COLLECTION {}".format(name, read_database, read_collection))
    dbr = DBClient(**db_param, database_name = read_database, collection_name=read_collection)  
        
    # start from min and end at max if collection is static
    rri = dbr.read_query_range(range_parameter='last_timestamp', range_increment=default_param["range_increment"], query_sort= [("last_timestamp", "ASC")],
                               query_filter = query_filter)
    
    # for debug only
    # rri._reader.range_iter_stop = rri._reader.range_iter_start + 60
    min_queue_size = default_param["min_queue_size"]
    discard = 0 # counter for short (<3) tracks
    cntr = 0
    next_batch = []

    
    while True: 
        
        try:
            # keep filling the queues so that they are not low in stock
            while raw_queue.qsize() <= min_queue_size :#or west_queue.qsize() <= min_queue_size:
#                 logger.debug("* current lower: {}, upper: {}, start: {}, stop: {}".format(rri._current_lower_value, rri._current_upper_value, rri._reader.range_iter_start, rri._reader.range_iter_stop))
                next_batch = next(rri)  
                for doc in next_batch:
                    cntr += 1

                    if len(doc["timestamp"]) > 3: 
                        doc = misc.interpolate(doc)
                        raw_queue.put(doc)          
                    else:
                        print("****** discard ",doc["_id"])
                        discard += 1
                        # logger.info("Discard a fragment with length less than 3")
    
                # logger.info("** qsize for raw_data_queue: east {}, west {}".format(east_queue.qsize(), west_queue.qsize()))
#                 logger.debug("** qsize for raw_data_queue:{}".format(raw_queue.qsize()))
                

            # if queue has sufficient number of items, then wait before the next iteration (throttle)
            logger.info("** queue size is sufficient. wait")     
            time.sleep(2)
          
         
            
        except StopIteration:  # rri reaches the end
            logger.warning("static_data_reader reaches the end of query range iteration. Exit")
            break
        
        except SIGINTException:  # SIGINT detected
            logger.warning("SIGINT/SIGUSR1 detected. Checkpoint not implemented.")
            break
        
        except Exception as e:
            logger.warning("Other exceptions occured. Exit. Exception:{}".format(str(e)))
            break

        
    
    # logger.info("outside of while loop:qsize for raw_data_queue: east {}, west {}".format(east_queue.qsize(), west_queue.qsize()))
    logger.debug("Discarded {} short tracks".format(discard))
    del dbr 
    logger.info("DBReader closed. Exit {}.".format(name))
    sys.exit() # for linux

    

   
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
    
 
    
 
    
if __name__ == '__main__':

    import queue
    import json
    with open("config/parameters.json") as f:
        parameters = json.load(f)

    parameters["raw_collection"] = "pristine_stork--RAW_GT1"
    east_queue = queue.Queue()
    west_queue = queue.Queue()
    static_data_reader(parameters, east_queue, west_queue, min_queue_size = 1000)
    
    
    