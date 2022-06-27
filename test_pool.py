#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 12:25:35 2022

@author: yanbing_wang
"""


# -----------------------------
import multiprocessing
from multiprocessing import Pool
import time
import os
import signal
import sys

import i24_logger.log_writer as log_writer
from i24_configparse import parse_cfg
import warnings
warnings.filterwarnings("ignore")

import math


config_path = os.path.join(os.getcwd(),"config")
# os.environ["user_config_directory"] = config_path
# os.environ["my_config_section"] = "DEBUG"
os.environ["user_config_directory"] = config_path
os.environ["my_config_section"] = "TEST"
parameters = parse_cfg("my_config_section", cfg_name = "test_param.config")

reconciliation_args = {"lam2_x": parameters.lam2_x,
                       "lam2_y": parameters.lam2_y,
                       # "lam1_x": parameters.lam1_x, 
                       # "lam1_y": parameters.lam1_y,
                       "PH": parameters.ph,
                       "IH": parameters.ih}


    
    
def dummy_worker(x, res_q):

    rec_worker_logger = log_writer.logger
    rec_worker_logger.set_name("rec_worker")
    # Does worker automatically shutdown when queue is empty?
   
    val = math.factorial(99999)
    res_q.put(x)
    rec_worker_logger.info("did some work")
    rec_worker_logger.info("x={}".format(x))
    
    
    


def reconciliation_pool(stitched_trajectory_queue: multiprocessing.Queue,
                         reconciled_queue: multiprocessing.Queue,) -> None:
    """
    Start a multiprocessing pool, each worker 
    :param stitched_trajectory_queue: results from stitchers, shared by mp.manager
    :param pid_tracker: a dictionary
    :return:
    """

    rec_parent_logger = log_writer.logger
    rec_parent_logger.set_name("rec_parent")
    setattr(rec_parent_logger, "_default_logger_extra",  {})

    # Signal handling
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    worker_pool = Pool(processes=4)
    signal.signal(signal.SIGINT, original_sigint_handler)
    
    rec_parent_logger.info("** Reconciliation pool starts. Pool size: {}".format(parameters.reconciliation_pool_size), extra = None)

    # signal.signal(signal.SIGINT, signal.SIG_IGN)    
    # signal.signal(signal.SIGPIPE,signal.SIG_DFL) # reset SIGPIPE so that no BrokePipeError when SIGINT is received
    
    while True:
        try:
            x = stitched_trajectory_queue.get(timeout = 1)
        except:
            break
        worker_pool.apply_async(dummy_worker, (x, reconciled_queue,))
        rec_parent_logger.info("remaining qsize: {}, res size: {}".format(stitched_trajectory_queue.qsize(), reconciled_queue.qsize()))
    

    worker_pool.close() # terminate() does not wait for each worker to finish current job, whereas close() does
    rec_parent_logger.info("Gracefully closed")
        
    worker_pool.join()
    rec_parent_logger.info("joined pool. Exiting")





if __name__ == '__main__':
    
    q = multiprocessing.Manager().Queue()
    q_res = multiprocessing.Manager().Queue()
    for i in range(10):
        q.put(i)
        print("add job ", i)
        time.sleep(0.1)
        
    reconciliation_pool(q, q_res)
    print("result: ")
    while q_res:
        print(q_res.get())
    
    

    