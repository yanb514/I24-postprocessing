#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 12:34:23 2022

@author: yanbing_wang
not tested
"""

import multiprocessing as mp
import os
import json
from i24_logger.log_writer import logger
from i24_sysctl import ServerControl


# Custom modules
import data_feed as df
import min_cost_flow as mcf
import reconciliation as rec


class ManagedProcess(ServerControl):
    
    def __init__(self):
        # GET PARAMAETERS
        with open("config/parameters.json") as f:
            parameters = json.load(f)
        
        
        # CHANGE NAME OF THE LOGGER
        manager_logger = logger
        manager_logger.set_name("postproc_manager")
        setattr(manager_logger, "_default_logger_extra",  {})
        
        # CREATE A MANAGER
        mp_manager = mp.Manager()
        manager_logger.info("Post-processing manager has PID={}".format(os.getpid()))

        # SHARED DATA STRUCTURES
        # ----------------------------------
        # ----------------------------------
        self.mp_param = mp_manager.dict()
        self.mp_param.update(parameters)
        
        # initialize some db collections
        df.initialize_db(self.mp_param)
        manager_logger.info("Post-processing manager initialized db collections. Creating shared data structures")
        
        # Raw trajectory fragment queue
        # -- populated by database connector that listens for updates
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        self.raw_fragment_queue_e = mp_manager.Queue(maxsize=parameters["raw_trajectory_queue_size"]) # east direction
        self.raw_fragment_queue_w = mp_manager.Queue(maxsize=parameters["raw_trajectory_queue_size"]) # west direction
        
        # Stitched trajectory queue
        # -- populated by stitcher and consumed by reconciliation pool
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        self.stitched_trajectory_queue = mp_manager.Queue(maxsize=parameters["stitched_trajectory_queue_size"]) 
        self.reconciled_queue = mp_manager.Queue(maxsize=parameters["reconciled_trajectory_queue_size"])
        
    
    def get_additional_args(self):
        # CODEWRITER TODO - Implement any shared variables (queues etc. here)
        # each entry is a tuple (args,kwargs) (list,dict)
        # to be added to process args/kwargs and key is process_name
        

        
        
        # key: process_name, val: [args],{kwargs}
        additional_args = {
                            "static_data_reader": ([], {
                                "default_param": self.mp_param, 
                                "east_queue": self.raw_fragment_queue_e, 
                                "west_queue": self.raw_fragment_queue_w
                                }),              
                            "stitcher_e": ([], {
                                "direction": "east", 
                                "fragment_queue": self.raw_fragment_queue_e, 
                                "stitched_trajectory_queue": self.stitched_trajectory_queue, 
                                "parameters": self.mp_param
                                }),
                            "stitcher_w": ([], {
                                "direction": "west", 
                                "fragment_queue": self.raw_fragment_queue_w, 
                                "stitched_trajectory_queue": self.stitched_trajectory_queue, 
                                "parameters": self.mp_param,
                                }),        
                            "reconciliation": ([], {
                                "parameters": self.mp_param, 
                                "stitched_trajectory_queue": self.stitched_trajectory_queue, 
                                "reconciled_queue": self.reconciled_queue,
                                }),
                            "reconciliation_writer": ([], {
                                "parameters": self.mp_param,
                                "reconciled_queue": self.reconciled_queue,
                                }),
                        }
        
        
        return additional_args






if __name__ == "__main__":
  
    # CODEWRITER TODO - add your process targets to register_functions
    config_path = os.path.join(os.getcwd(),"config")
    os.environ["USER_CONFIG_DIRECTORY"] = config_path 
    
    # jobs.jpl
    
    # register_functions = [df.static_data_reader, mcf.min_cost_flow_online_alt_path, rec.reconciliation_pool, rec.write_reconciled_to_db]
    # name_to_process = dict([(fn.__name__, fn) for fn in register_functions])
    
    name_to_process = {
        "static_data_reader": df.static_data_reader,
        "stitcher_e": mcf.min_cost_flow_online_alt_path,
        "stitcher_w": mcf.min_cost_flow_online_alt_path,
        "reconciliation": rec.reconciliation_pool,
        "reconciliation_writer": rec.write_reconciled_to_db
        }
    
    s = ManagedProcess(name_to_process) # name to process?
    
    # start server / process manager
    s.main()
    
    


    