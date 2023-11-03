# -----------------------------
__file__ = 'reconciliation.py'
__doc__ = """

Update 6/27/2021
If each worker handles mongodb client connection, then they have to close the connection in order for the pool to join (in the event of graceful shutdown).
It is not recommended by mongodb to open and close connections all the time.
Instead, modify the code such that open the connection for dbreader and writer at the parent process. Each worker does not have direct access to mongodb client.
After done processing, each worker send results back to the parent process using a queue
"""

# -----------------------------
import multiprocessing
from multiprocessing import Pool
import time
import os
import signal
import sys
import queue
import heapq

import i24_logger.log_writer as log_writer
from i24_database_api import DBClient

# from utils.utils_reconciliation import receding_horizon_2d_l1, resample, receding_horizon_2d, combine_fragments, rectify_2d
from utils.utils_opt import combine_fragments, resample, opt1, opt2, opt1_l1, opt2_l1, opt2_l1_constr


class SIGINTException(SystemExit):
    pass

def soft_stop_hdlr(sig, action):
    '''
    Signal handling for SIGINT
    Soft terminate current process. Close and join the pool.
    '''
    raise SIGINTException # so to exit the while true loop
    
# def soft_stop_hdlr(sig, frame):
    # signal.signal(signal.SIGPIPE,signal.SIG_DFL) # reset SIGPIPE so that no BrokePipeError when SIGINT is received
    

def apprentice(dir, queue_list, sorted_queue, name):
    """
    :param queue_list: a list of queues to store stitched fragment lists
    :param sorted_queue: queue with fragments sorted on last_timestamp
    1. combine fragments from queue_list
    2. put them to a heap
    3. pop from heap to sorted_queue
    It is right now a batch process-> store everything in memory
    """
    logger = log_writer.logger 
    logger.set_name(name)
    logger.info("Process starts.")
    
    h = []
    
    for q in queue_list:
        while not q.empty():
            traj_docs = q.get(block=False)
            combined_trajectory = combine_fragments(traj_docs)
            resampled_trajectory = resample(combined_trajectory, dt=0.1, fillnan=True)
            # print(sum(resampled_trajectory["x_position"]))
            heapq.heappush(h, (resampled_trajectory["last_timestamp"],resampled_trajectory["_id"], resampled_trajectory)) # orderd by last_timestamp but it may not be unique
            
    # pop from queue -> sorted in last_timestamp
    logger.info("Heap size: {}".format(len(h)))
    while len(h)>0:
        _, _, traj = heapq.heappop(h)
        sorted_queue.put(traj)
    
    logger.info("sorted queue size: {}".format(sorted_queue.qsize()))
    return


def write_queues_2_db(db_param, parameters, all_queues, name=None):
    """
    write all documents from all queues immediately to a temporary database
    each document is a list of stitched trajectories
    """
    writer_logger = log_writer.logger 
    if not name:
        name = "temp_writer"
    writer_logger.set_name(name)
    setattr(writer_logger, "_default_logger_extra",  {})
    
    dbc = DBClient(**db_param, database_name = parameters["temp_database"], collection_name = parameters["reconciled_collection"])
    dbc.collection.create_index("compute_node_id")
    dbc.collection.create_index("direction")
    dbc.collection.create_index("last_timestamp")
    dbc.collection.create_index("first_timestamp")
    
    TIMEOUT = parameters["write_temp_timeout"] 
    HB = parameters["log_heartbeat"]
    
    # Exit the while loop if all queues are empty for TIMEOUT seconds
    start = time.time()
    begin = start
    
    while time.time() - start < TIMEOUT:
        
        for q in all_queues:
            if not q.empty():
                traj_docs = q.get(block=False)
                if isinstance(traj_docs, list):
                    combined_trajectory = combine_fragments(traj_docs)
                else:
                    combined_trajectory = combine_fragments([traj_docs])
                doc = resample(combined_trajectory, dt=0.04, fillnan=False)
#                 doc = combined_trajectory
                # convert arrays to list
                for key in ["timestamp", "x_position", "y_position"]:
                    doc[key] = list(doc[key])
                for key in ["length", "width", "height"]:
                    doc[key] = float(doc[key])
                    
                dbc.thread_insert(doc)
                start = time.time()
            
            now = time.time()
            if now - begin > HB :
                writer_logger.info("Est. count in [temp] {}: {}".format(dbc.collection_name, dbc.collection.estimated_document_count()))
                begin = now
    
    writer_logger.info("Temp Writer timed out. Est. count in temp collection {}: {}".format(dbc.collection_name, dbc.collection.estimated_document_count()))
    return
    
                
    
    
    

    
def reconcile_single_trajectory(reconciliation_args, combined_trajectory, reconciled_queue) -> None:
    """
    Resample and reconcile a single trajectory, and write the result to a queue
    :param next_to_reconcile: a trajectory document
    :return:
    """
    
    rec_worker_logger = log_writer.logger 
    rec_worker_logger.set_name("rec_worker")
    setattr(rec_worker_logger, "_default_logger_extra",  {})

    resampled_trajectory = resample(combined_trajectory, dt=0.04)
    if "post_flag" in resampled_trajectory:
        # skip reconciliation
        rec_worker_logger.info("+++ Flag as low conf, skip reconciliation", extra = None)

    else:
        try:
            # finished_trajectory = rectify_2d(resampled_trajectory, reg = "l1", **reconciliation_args)  
            finished_trajectory = opt2_l1_constr(resampled_trajectory, **reconciliation_args)  
            # finished_trajectory = opt2(resampled_trajectory, **reconciliation_args)  
            reconciled_queue.put(finished_trajectory)
            # rec_worker_logger.debug("*** Reconciled a trajectory, duration: {:.2f}s, length: {}".format(finished_trajectory["last_timestamp"]-finished_trajectory["first_timestamp"], len(finished_trajectory["timestamp"])), extra = None)
        
        except Exception as e:
            rec_worker_logger.info("+++ Flag as {}, skip reconciliation".format(str(e)), extra = None)



def reconciliation_pool(parameters, db_param, stitched_trajectory_queue: multiprocessing.Queue, 
                        reconciled_queue: multiprocessing.Queue, ) -> None:
    """
    Start a multiprocessing pool, each worker 
    :param stitched_trajectory_queue: results from stitchers, shared by mp.manager
    :param pid_tracker: a dictionary
    :return:
    """
    # Signal handling: https://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python/35134329#35134329
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    n_proc = min(multiprocessing.cpu_count(), parameters["worker_size"])
    worker_pool = Pool(processes= n_proc)
    # signal.signal(signal.SIGINT, original_sigint_handler)
    signal.signal(signal.SIGINT, soft_stop_hdlr)
    
    # parameters
    reconciliation_args=parameters["reconciliation_args"]
    # for key in ["lam2_x","lam2_y", "lam3_x", "lam3_y","lam1_x","lam1_y"]:
    #     reconciliation_args[key] = parameters[key]
    
    rec_parent_logger = log_writer.logger
    rec_parent_logger.set_name("reconciliation")
    setattr(rec_parent_logger, "_default_logger_extra",  {})

    # wait to get raw collection name
    while parameters["raw_collection"]=="":
        time.sleep(1)
    
    rec_parent_logger.info("** Reconciliation pool starts. Pool size: {}".format(n_proc), extra = None)
    TIMEOUT = parameters["reconciliation_pool_timeout"]
    
    cntr = 0
    while True:
        try:
            try:
                traj_docs = stitched_trajectory_queue.get(timeout = TIMEOUT) #20sec
                cntr += 1
            except queue.Empty: 
                rec_parent_logger.warning("Reconciliation pool is timed out after {}s. Close the reconciliation pool.".format(TIMEOUT))
                worker_pool.close()
                break
            if isinstance(traj_docs, list):
                combined_trajectory = combine_fragments(traj_docs)
            else:
                combined_trajectory = combine_fragments([traj_docs])
            # combined_trajectory = combine_fragments(traj_docs)  
            worker_pool.apply_async(reconcile_single_trajectory, (reconciliation_args, combined_trajectory, reconciled_queue, ))
            
            # if time.time()-begin > HB:
            #     rec_parent_logger.info("reconciled_queue size: {}".format(reconciled_queue.qsize()))

        except SIGINTException: # handle SIGINT here
            rec_parent_logger.warning("SIGINT detected. Terminate pool.")
            worker_pool.terminate() # immediately terminate all tasks
            break
        
        except Exception as e: # other exception
            rec_parent_logger.warning("{}, Close the pool".format(e))
            worker_pool.close() # wait until all processes finish their task
            break
            
            
        
    # Finish up  
    worker_pool.join()
    rec_parent_logger.info("Joined the pool.")
    
    sys.exit(0)



def write_reconciled_to_db(parameters, db_param, reconciled_queue):
    
    signal.signal(signal.SIGINT, soft_stop_hdlr)
    reconciled_writer = log_writer.logger
    reconciled_writer.set_name("reconciliation_writer")
    
    # Signal handling: 
    # SIGINT raises KeyboardInterrupt,  close dbw, terminate pool and exit. # TODO: pool.terminate() or close()?
    # SIGUSR1 is ignored. The process terminates when queue is empty. Close and join the pool
    # def handler(sigusr, frame):
    #     reconciled_writer.warning("SIGUSR1 detected. Finish processing current queues.")
    #     signal.signal(sigusr, signal.SIG_IGN)    
    #     signal.signal(signal.SIGPIPE,signal.SIG_DFL) # reset SIGPIPE so that no BrokePipeError when SIGINT is received
        
    # signal.signal(signal.SIGUSR1, handler) # ignore SIGUSR1
    reconciled_schema_path = os.path.join(os.environ["USER_CONFIG_DIRECTORY"], parameters["reconciled_schema_path"])

    # wait to get raw collection name
    while parameters["reconciled_collection"]=="":
        time.sleep(1)
        
    dbw = DBClient(**db_param, database_name = parameters["reconciled_database"], 
                   collection_name = parameters["reconciled_collection"], schema_file=reconciled_schema_path)
    TIMEOUT = parameters["reconciliation_writer_timeout"]
    # cntr = 0
    HB = parameters["log_heartbeat"]
    begin = time.time()
    t_max, t_min = parameters["t_max"], parameters["t_min"]
    t_curr_eb, t_curr_wb = t_min, t_min
    
    # Write to db
    while True:
        try:
            try:
                reconciled_traj = reconciled_queue.get(timeout = TIMEOUT)
            except queue.Empty:
                reconciled_writer.warning("Getting from reconciled_queue reaches timeout {} sec.".format(TIMEOUT))
                break
            # add this if statement when doing pp_lite_reverse
            # if isinstance(reconciled_traj, list):
            #     reconciled_traj = combine_fragments(reconciled_traj)
            #     reconciled_traj = resample(reconciled_traj, dt=0.04, fillnan=True)
            #     reconciled_traj["timestamp"] = reconciled_traj["timestamp"].tolist()
            #     reconciled_traj["x_position"] = reconciled_traj["x_position"].tolist()
            #     reconciled_traj["y_position"] = reconciled_traj["y_position"].tolist()
            
            dbw.write_one_trajectory(thread = True, **reconciled_traj)
            if reconciled_traj["direction"] == 1:
                t_curr_eb = reconciled_traj["last_timestamp"]
            else:
                t_curr_wb = reconciled_traj["last_timestamp"]
            # cntr += 1
            if time.time()-begin > HB:
                begin = time.time()
                # log progress
                progress_eb = (t_curr_eb-t_min)/(t_max-t_min)*100
                progress_wb = (t_curr_wb-t_min)/(t_max-t_min)*100
                reconciled_writer.info(
                    "Est. count in [reconciled] {}: {} | EB: {:.2f}% | WB: {:.2f}%".format(dbw.collection_name, 
                                                             dbw.collection.estimated_document_count(), 
                                                             progress_eb, progress_wb))
                
        except SIGINTException: # handle SIGINT here 
            reconciled_writer.warning("SIGINT detected. Exit reconciled writer")
            break
    
    
    reconciled_writer.info(
                    "Est. count in [reconciled] {}: {}".format(dbw.collection_name, 
                                                             dbw.collection.estimated_document_count()))
    
    # Safely close the mongodb client connection
    del dbw
    reconciled_writer.warning("DBWriter closed. Exit reconciled_writer")
    sys.exit(0)



    
    
    
    
    
    

if __name__ == '__main__':
    import json
    import numpy as np
    from bson.objectid import ObjectId
    import data_feed as df
    # initialize parameters
    with open("config/parameters.json") as f:
        parameters = json.load(f)
        
    with open(os.path.join(os.environ["USER_CONFIG_DIRECTORY"], "db_param.json")) as f:
        db_param = json.load(f)
        
    RES_THRESH_X = parameters["residual_threshold_x"]
    RES_THRESH_Y = parameters["residual_threshold_y"]
    CONF_THRESH = parameters["conf_threshold"],
    REMAIN_THRESH = parameters["remain_threshold"]
    
    # reconciliation_args={}
    # for key in ["lam3_x","lam3_y", "lam2_x", "lam2_y", "lam1_x", "lam1_y"]:
    #     reconciliation_args[key] = parameters[key]
    reconciliation_args = parameters["reconciliation_args"]
    
    # send some fragments to queue
    stitched_q = multiprocessing.Manager().Queue()
    reconciled_queue = multiprocessing.Manager().Queue()
    raw_queue = multiprocessing.Manager().Queue()
    counter = 0 
    
    test_dbr = DBClient(**db_param, database_name = "trajectories", collection_name = "636332547c61e6427c5ad508_short")
    
    for doc in test_dbr.collection.find({}).sort("starting_x",-1).limit(3):
        # doc = add_filter(doc, test_dbr.collection, RES_THRESH_X, RES_THRESH_Y, 
        #                CONF_THRESH, REMAIN_THRESH)
        stitched_q.put([doc])
        counter += 1
        print("doc length: ", len(doc["timestamp"]))
        raw_queue.put(doc)
        # stitched_q.put(doc)
        # print(doc["_id"])
        # print(doc["fragment_ids"])
        
    print("current q size: ", stitched_q.qsize())
    
    # pool
    # reconciliation_pool(parameters, stitched_q)
    
    while not stitched_q.empty():
        fragment_list = stitched_q.get(block=False)
        combined_trajectory = combine_fragments(fragment_list)
        reconcile_single_trajectory(reconciliation_args, combined_trajectory, reconciled_queue)
        # doc["timestamp"] = np.array(doc["timestamp"])
        # doc["x_position"] = np.array(doc["x_position"])
        # doc["y_position"] = np.array(doc["y_position"])
        # rec_doc = rectify_2d(doc, reg = "l1", **reconciliation_args)  
     
    # r = reconciled_queue.get()
    # print("final queue size: ",reconciled_queue.qsize())
    # print(reconciliation_args)
    # print(r["x_score"], r["y_score"])
    
    #%% plot
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1,2)
    # ax[0].scatter(doc["timestamp"], doc["x_position"])
    # # filter = np.array(doc["filter"], dtype=bool)
    # ax[0].scatter(np.array(doc["timestamp"]),np.array(doc["x_position"]), c="lightgrey")
    # ax[0].scatter(r["timestamp"], r["x_position"], s=1)
    # ax[1].scatter(doc["timestamp"], doc["y_position"])
    # # filter = np.array(doc["filter"], dtype=bool)
    # ax[1].scatter(np.array(doc["timestamp"]),np.array(doc["y_position"]), c="lightgrey")
    # ax[1].scatter(r["timestamp"], r["y_position"], s=1)
    
    from multi_opt_viz import plot_track
    while not reconciled_queue.empty():
        r = reconciled_queue.get()
        doc = raw_queue.get()
    
        plot_track([doc, r])
    
    
    
    
    
    