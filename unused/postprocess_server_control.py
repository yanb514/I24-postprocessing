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
from server_control import ServerControl


# Custom modules
import data_feed as df
import min_cost_flow as mcf
import reconciliation as rec


class ManagedProcess(ServerControl):
    
    
    def get_additional_args(self):
        # CODEWRITER TODO - Implement any shared variables (queues etc. here)
        # each entry is a tuple (args,kwargs) (list,dict)
        # to be added to process args/kwargs and key is process_name
        
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
        mp_param = mp_manager.dict()
        mp_param.update(parameters)
        
        # initialize some db collections
        df.initialize_db(mp_param)
        manager_logger.info("Post-processing manager initialized db collections. Creating shared data structures")
        
        # Raw trajectory fragment queue
        # -- populated by database connector that listens for updates
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        raw_fragment_queue_e = mp_manager.Queue(maxsize=parameters["raw_trajectory_queue_size"]) # east direction
        raw_fragment_queue_w = mp_manager.Queue(maxsize=parameters["raw_trajectory_queue_size"]) # west direction
        
        # Stitched trajectory queue
        # -- populated by stitcher and consumed by reconciliation pool
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        stitched_trajectory_queue = mp_manager.Queue(maxsize=parameters["stitched_trajectory_queue_size"]) 
        reconciled_queue = mp_manager.Queue(maxsize=parameters["reconciled_trajectory_queue_size"])
        
        
        # key: process_name, val: [args],{kwargs}
        additional_args = {
                            "static_data_reader": ([], {
                                "default_param": mp_param, 
                                "east_queue": raw_fragment_queue_e, 
                                "west_queue": raw_fragment_queue_w
                                }),              
                            "stitcher_e": ([], {
                                "direction": "east", 
                                "fragment_queue": raw_fragment_queue_e, 
                                "stitched_trajectory_queue": stitched_trajectory_queue, 
                                "parameters": mp_param
                                }),
                            "stitcher_w": ([], {
                                "direction": "west", 
                                "fragment_queue": raw_fragment_queue_w, 
                                "stitched_trajectory_queue": stitched_trajectory_queue, 
                                "parameters": mp_param,
                                }),        
                            "reconciliation": ([], {
                                "parameters": mp_param, 
                                "stitched_trajectory_queue": stitched_trajectory_queue, 
                                "reconciled_queue": reconciled_queue,
                                }),
                            "reconciliation_writer": ([], {
                                "parameters": mp_param,
                                "reconciled_queue": reconciled_queue,
                                }),
                        }
        
        
        return additional_args






if __name__ == "__main__":
  
    # CODEWRITER TODO - add your process targets to register_functions
     
    register_functions = [
        {
            "process": "static_data_reader",
            "command": df.static_data_reader, # string?
            "timeout": 10,
            "args": [], # repeat from additional_args?
            "kwargs": {}, 
            "abandon": False,
            "group": "POSTPROCESSING",
            "description": "",
            "keep_alive":True
        },
        {
            "process": "stitcher_e",
            "command": mcf.min_cost_flow_online_alt_path,
            "timeout": 10,
            "args": [], # repeat from additional_args?
            "kwargs": {}, 
            "abandon": False,
            "group": "POSTPROCESSING",
            "description": "",
            "keep_alive":True
        },
        {
            "process": "stitcher_w",
            "command": mcf.min_cost_flow_online_alt_path, # string?
            "timeout": 10,
            "args": [], # repeat from additional_args?
            "kwargs": {}, 
            "abandon": False,
            "group": "POSTPROCESSING",
            "description": "",
            "keep_alive":True
        },
        {
            "process": "reconciliation",
            "command": rec.reconciliation_pool, # string?
            "timeout": 10,
            "args": [], # repeat from additional_args?
            "kwargs": {}, 
            "abandon": False,
            "group": "POSTPROCESSING",
            "description": "",
            "keep_alive":True
        },
        {
            "process": "reconciliation_writer",
            "command": rec.write_reconciled_to_db, # string?
            "timeout": 10,
            "args": [], # repeat from additional_args?
            "kwargs": {}, 
            "abandon": False,
            "group": "POSTPROCESSING",
            "description": "",
            "keep_alive":True
        },
    ]
    
    
    
    s = ManagedProcess(register_functions) # name to process?
    
    # start server / process manager
    s.main()
    
    


    