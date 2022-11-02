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
    

    
def reconcile_single_trajectory(reconciliation_args, combined_trajectory, reconciled_queue) -> None:
    """
    Resample and reconcile a single trajectory, and write the result to a queue
    :param next_to_reconcile: a trajectory document
    :return:
    """
    
    rec_worker_logger = log_writer.logger 
    rec_worker_logger.set_name("rec_worker")
    setattr(rec_worker_logger, "_default_logger_extra",  {})

    resampled_trajectory = resample(combined_trajectory)
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
    worker_pool = Pool(processes=multiprocessing.cpu_count())
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
    
    rec_parent_logger.info("** Reconciliation pool starts. Pool size: {}".format(multiprocessing.cpu_count()), extra = None)
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
        
            combined_trajectory = combine_fragments(traj_docs)    
            worker_pool.apply_async(reconcile_single_trajectory, (reconciliation_args, combined_trajectory, reconciled_queue, ))
            
            if cntr % 100 == 0:
                rec_parent_logger.info("reconciled_queue size: {}".format(reconciled_queue.qsize()))

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
    cntr = 0
    # Write to db
    while True:
        try:
            try:
                reconciled_traj = reconciled_queue.get(timeout = TIMEOUT)
            except queue.Empty:
                reconciled_writer.warning("Getting from reconciled_queue reaches timeout {} sec.".format(TIMEOUT))
                break
        
            dbw.write_one_trajectory(thread = True, **reconciled_traj)
            cntr += 1
            if cntr % 100 == 0:
                reconciled_writer.info("Est. count in collection {}: {}".format(dbw.collection_name, dbw.collection.estimated_document_count()))
                
        except SIGINTException: # handle SIGINT here 
            reconciled_writer.warning("SIGINT detected. Exit reconciled writer")
            break
    
    
    reconciled_writer.info("Est. count in reconciled collection {}: {}".format(dbw.collection_name, dbw.collection.estimated_document_count()))
    
    
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
    counter = 0 
    
    test_dbr = DBClient(**db_param, database_name = "trajectories", collection_name = "sanctimonious_beluga--RAW_GT1")
    
    for doc in test_dbr.collection.find({"_id": ObjectId("62fd2a29b463d2b0792821c1")}):
        # doc = add_filter(doc, test_dbr.collection, RES_THRESH_X, RES_THRESH_Y, 
        #                CONF_THRESH, REMAIN_THRESH)
        stitched_q.put([doc])
        counter += 1
        print("doc length: ", len(doc["timestamp"]))
        if counter > 15:
            break
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
     
    r = reconciled_queue.get()
    print("final queue size: ",reconciled_queue.qsize())
    print(reconciliation_args)
    print(r["x_score"], r["y_score"])
    
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
    
    from multi_opt import plot_track
    plot_track([doc, r])
    
    
    
    
    
    