#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:55:04 2022

@author: yanbing_wang
6/30
Live data read should be only one process, and distribute to 2 queues (east/west) based on document direction
two seperate live_data_feed processes will mess up the change stream
"""
from i24_database_api.db_reader import DBReader
from i24_database_api.db_writer import DBWriter
import i24_logger.log_writer as log_writer
import time
import signal
import sys
import os



def change_stream_simulator(default_param, insert_rate):
    '''
    When no live-streaming data, simulate one by reading from a collection (default_param.raw_collection), and insert to a new collection (write_to_collection)
    insert_rate: insert no. of documents per second
    query in ascending first_timestamp, such that the write_to_collection is slightly out of order in terms of last_timestamp
    '''
    # initialize the logger
    logger = log_writer.logger
    logger.set_name("change_stream_simulator")
    setattr(logger, "_default_logger_extra",  {})
    
    # reset the new collection
    write_to_collection = default_param.raw_collection + "_simulated"
    raw_schema_path = os.path.join(os.environ["user_config_directory"],default_param.raw_schema_path)
    dbw = DBWriter(default_param, collection_name = write_to_collection, schema_file = raw_schema_path)
    dbw.collection.drop()
    dbw = DBWriter(default_param, collection_name = write_to_collection, schema_file = raw_schema_path)
    logger.info("DBWriter initiated")
    
    # initiate data reader
    dbr = DBReader(default_param, collection_name=default_param.raw_collection)
    logger.info("DBReader initiated")
    
    start = dbr.get_min("first_timestamp") - 1e-6
    end = start + 5
    cur = dbr.get_range("first_timestamp", start, end)
    
    # write to simulated collection
    count = 0
    # time.sleep(3) # wait for change stream to get initialized
    for doc in cur:
        time.sleep(1/insert_rate)
        print("insert: {}".format(doc["first_timestamp"]))
        doc.pop("_id")
        dbw.write_one_trajectory(thread = False, **doc)
        count += 1
        if count % 100 == 0:
            logger.info("{} docs written to dbw".format(count))
            time.sleep(3)
    
    # exit
    logger.info(f"Finished writing {count} to simulated collection. Exit")
    
    del dbr, dbw
    sys.exit(0)
    
   
    
def live_data_reader(default_param, east_queue, west_queue, t_buffer = 100, min_queue_size = 1000, read_from_simulation = True):
    """
    Runs a database stream update listener on top of a managed cache that buffers data for a safe amount of time so
        that it can be assured to be time-ordered. Refill data queue if the queue size is below a threshold AND the next query range is before change_stream t_max - t_buffer
    ** THIS PROCEDURE AND FUNCTION IS STILL UNDER DEVELOPMENT **
    ** NEEDS TO DETERMINE **
    t_buffer: buffer time (in sec) such that no new fragment will be inserted before t_max - t_buffer   
    
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
    logger.set_name("live_data_reader")
    setattr(logger, "_default_logger_extra",  {})
    
    # time.sleep(3) # wait for cs simulator to get started
    
    # Connect to a database reader
    if read_from_simulation:
        # time.sleep(4) # wait for change_stream_simulator to start
        raw_collection = default_param.raw_collection + "_simulated"
    else:
        raw_collection = default_param.raw_collection
    dbr = DBReader(default_param, collection_name=raw_collection)


    # Signal handling: in live data read, SIGINT and SIGUSR1 are handled in the same way
    class SignalHandler():

        run = True
        count = 0 # count the number of times a signal is received
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        def __init__(self):
            signal.signal(signal.SIGINT, self.shut_down)
            signal.signal(signal.SIGUSR1, self.shut_down)
            signal.signal(signal.SIGPIPE,signal.SIG_DFL) # reset SIGPIPE so that no BrokePipeError when SIGINT is received
        
        def shut_down(self, *args):
            self.run = False
            self.count += 1
            logger.info("{} detected {} times".format(signal.Signals(args[0]).name, self.count))
            
    sig_hdlr = SignalHandler()  
    pipeline = [{'$match': {'operationType': 'insert'}}]
    
    # Initialize rri to raise StopIteration exception
    rri = dbr.read_query_range(range_parameter='last_timestamp', 
                               range_greater_than =-1-default_param.range_increment , range_less_than=-1, 
                               range_increment=default_param.range_increment,
                               query_sort = ("last_timestamp", "ASC"))
    safe_query_time = -1
    dbr.range_iter_stop = safe_query_time
    

    # have an internal time out for changes
    with dbr.collection.watch(pipeline) as stream:
        # close the stream if SIGINT received
        while stream.alive:
            change = stream.try_next() # first change 
            # Note that the ChangeStream's resume token may be updated
            # even when no changes are returned.
            # print("Current resume token: %r" % (stream.resume_token,))
            if change is not None:
                print("Change document: %r" % (change['fullDocument']['first_timestamp'],))
                east_queue.put(change['fullDocument'])
                continue
            # We end up here when there are no recent changes.
            # Sleep for a while before trying again to avoid flooding
            # the server with getMore requests when no changes are
            # available.
            time.sleep(5)
        # end up here where the stream is no longer alive
        print("stream is no longer alive")
        stream.close()
    print("stream closed")
    
    
    
    # wait indefinitely for changes
    # resume_token = None
    # try:
    #     with dbr.collection.watch(pipeline=pipeline,resume_after=resume_token) as stream:
    #         for insert_change in stream:
    #             print("Change document: %r" % (insert_change['fullDocument']['first_timestamp'],))
    #             # block until this document is ready to put to queue
    #             if insert_change['fullDocument']['last_timestamp'] < safe_query_time
    #             east_queue.put(insert_change['fullDocument']) 
    #             resume_token = stream.resume_token
    
    # except Exception as e:
    #     print(e)
        
    
    # logger.info("outside of while loop:qsize for raw_data_queue: east {}, west {}".format(east_queue.qsize(), west_queue.qsize()))
    del dbr 
    logger.info("DBReader closed. Exiting live_data_reader while loop.")
    sys.exit() # for linux
    
    
    
    
def static_data_reader(default_param, east_queue, west_queue, t_buffer = 100, min_queue_size = 1000, read_from_simulation = True):
    """
    Runs a database stream update listener on top of a managed cache that buffers data for a safe amount of time so
        that it can be assured to be time-ordered. Refill data queue if the queue size is below a threshold AND the next query range is before change_stream t_max - t_buffer
    ** THIS PROCEDURE AND FUNCTION IS STILL UNDER DEVELOPMENT **
    ** NEEDS TO DETERMINE **
    t_buffer: buffer time (in sec) such that no new fragment will be inserted before t_max - t_buffer   
    
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
    logger.set_name("live_data_reader")
    setattr(logger, "_default_logger_extra",  {})
    
    # Connect to a database reader
    if read_from_simulation:
        # time.sleep(4) # wait for change_stream_simulator to start
        raw_collection = default_param.raw_collection + "_simulated"
    else:
        raw_collection = default_param.raw_collection
    dbr = DBReader(default_param, collection_name=raw_collection)
    # temporary: start from min and end at max
    rri = dbr.read_query_range(range_parameter='last_timestamp', range_increment=default_param.range_increment)

    # Signal handling: in live data read, SIGINT and SIGUSR1 are handled in the same way
    class SignalHandler():

        run = True
        count = 0 # count the number of times a signal is received
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        def __init__(self):
            signal.signal(signal.SIGINT, self.shut_down)
            signal.signal(signal.SIGUSR1, self.shut_down)
            signal.signal(signal.SIGPIPE,signal.SIG_DFL) # reset SIGPIPE so that no BrokePipeError when SIGINT is received
        
        def shut_down(self, *args):
            self.run = False
            self.count += 1
            logger.info("{} detected {} times".format(signal.Signals(args[0]).name, self.count))
            
    sig_hdlr = SignalHandler()  
    
    
    # for debug only
    # if running_mode == "TEST":
    # rri._reader.range_iter_stop = rri._reader.range_iter_start + 60
    
    
    pipeline = [{'$match': {'operationType': 'insert'}}] # watch for insertion only
    first_change_time = dbr.get_max("last_timestamp") # to keep track of the first change during each change stream event
    safe_query_time = first_change_time - t_buffer # guarantee time-order up until safe_query_time


    discard = 0
    no_change_time = 0
    start_timeout = time.time()
    
    while sig_hdlr.run:
        
        logger.info("* current lower: {}, upper: {}, safe_query_time: {}, start: {}, stop: {}".format(rri._current_lower_value, rri._current_upper_value, safe_query_time, rri._reader.range_iter_start, rri._reader.range_iter_stop))
    
        try:
            # logger.info("current queue size: {}, first_change_time: {:.2f}, query range: {:.2f}-{:.2f}".format(ready_queue.qsize(),first_change_time, rri._current_lower_value, rri._current_upper_value))
            if east_queue.qsize() <= min_queue_size or west_queue.qsize() <= min_queue_size: # only move to the next query range if queue is low in stock
                stream = dbr.collection.watch(pipeline) 
                first_insert_change = stream.try_next() # get the first insert since last listen
                # logger.debug("first_insert_change: {}".format(first_insert_change))
  
                if first_insert_change: # if there is updates by the time collection.watch() is called
                    first_change_time = max(first_insert_change["fullDocument"]["last_timestamp"], first_change_time)
                    safe_query_time = first_change_time - t_buffer
                    dbr.range_iter_stop = safe_query_time
                    no_change_time = 0
                    start_timeout = time.time()
                    
                else: # if no change event in this iteration
                    no_change_time += time.time() - start_timeout
                    start_timeout = time.time()
                    if no_change_time > 8: # this time out should be slightly less than stitcher get timeout
                        logger.debug("** Have no updates for 8 seconds. Increment safe_query_time.")
                        safe_query_time += default_param.range_increment
                    
                
                if rri._current_upper_value > safe_query_time and rri._current_upper_value < rri._reader.range_iter_stop: # if not safe to query and current range is not the end, then wait 
                    logger.debug("** not safe to query. Wait.")
                    time.sleep(2)
                    
                else: # if safe to query
                    if rri._current_lower_value >= rri._reader.range_iter_stop:
                        logger.warning("Current query range is above iter stop. Break reader")
                        break
                    # logger.info("read next query range: {:.2f}-{:.2f}".format(rri._current_lower_value, rri._current_upper_value))
                    
                    lower, upper = rri._current_lower_value, rri._current_upper_value
                    logger.info("** current lower: {}, upper: {}, safe_query_time: {}, start: {}, stop: {}".format(lower, upper, safe_query_time, rri._reader.range_iter_start, rri._reader.range_iter_stop))
                    next_batch = next(rri)
                    

                    for doc in next_batch:
                        if len(doc["timestamp"]) > 3: 
                            if doc["direction"] == 1:
                                # logger.debug("write a doc to east queue, dir={}".format(doc["direction"]))
                                east_queue.put(doc)
                            else:
                                west_queue.put(doc)
                        else:
                            discard += 1
                            # logger.info("Discard a fragment with length less than 3")

                    logger.info("** qsize for raw_data_queue: east {}, west {}".format(east_queue.qsize(), west_queue.qsize()))
                    
                        
            else: # if queue has sufficient number of items, then wait before the next iteration (throttle)
                logger.info("** queue size is sufficient")     
                time.sleep(2)
          
         
            
        except StopIteration:  # rri reaches the end
            logger.warning("live_data_reader reaches the end of query range iteration. Exit")
            break
        
        
        except Exception as e:
            if sig_hdlr.run:
                logger.warning("live_data_reader reaches the end of query range iteration. Exit. Exception:{}".format(e))
            else:
                logger.warning("SIGINT/SIGUSR1 detected. Checkpoint not implemented. Exception:{}".format(e))
            break

        
     
    
    # logger.info("outside of while loop:qsize for raw_data_queue: east {}, west {}".format(east_queue.qsize(), west_queue.qsize()))
    logger.info("Discarded {} short tracks".format(discard))
    del dbr 
    logger.info("DBReader closed. Exiting live_data_reader while loop.")
    sys.exit() # for linux

    
    
    