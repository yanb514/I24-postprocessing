#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 12:34:23 2022

@author: yanbing_wang
not tested
"""

import multiprocessing as mp
import os
from i24_configparse import parse_cfg
from i24_sys.ServerControlStub import ServerControlStub

config_path = os.path.join(os.getcwd(),"config")
os.environ["USER_CONFIG_DIRECTORY"] = config_path 
os.environ["user_config_directory"] = config_path
os.environ["my_config_section"] = "TEST"
parameters = parse_cfg("my_config_section", cfg_name = "test_param.config")

# Customized modules
from live_data_feed import live_data_reader # change to live_data_read later
import min_cost_flow as mcf
import reconciliation as rec


# CODEWRITER TODO - add your process targets to register_functions
register_functions = {
                        "live_data_reader": live_data_reader,
                        "stitcher_east": mcf.min_cost_flow_online_alt_path,
                        "stitcher_west": mcf.min_cost_flow_online_alt_path,
                        # "reconciliation": rec.reconciliation_pool,
                        # "reconciliation_writer": rec.write_reconciled_to_db
                      }

# name_to_process = dict([(name, fn) for fn in register_functions])



class ServerControl(ServerControlStub):
    
    def get_additional_args(self):
        # CODEWRITER TODO - Implement any shared variables (queues etc. here)
        # each entry is a tuple (args,kwargs) (list,dict)
        # to be added to process args/kwargs and key is process_name
        # CREATE A MANAGER
        mp_manager = mp.Manager()
        
        
        # Raw trajectory fragment queue
        # -- populated by database connector that listens for updates
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        raw_fragment_queue_e = mp_manager.Queue(maxsize=parameters.raw_trajectory_queue_size) # east direction
        raw_fragment_queue_w = mp_manager.Queue(maxsize=parameters.raw_trajectory_queue_size) # west direction
        
        # Stitched trajectory queue
        # -- populated by stitcher and consumed by reconciliation pool
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        stitched_trajectory_queue = mp_manager.Queue(maxsize=parameters.stitched_trajectory_queue_size) 
        reconciled_queue = mp_manager.Queue(maxsize=parameters.reconciled_trajectory_queue_size)
        
        additional_args = {
                            "live_data_reader": (
                                            (parameters, parameters.raw_collection, 
                                            parameters.range_increment,
                                            raw_fragment_queue_e, raw_fragment_queue_w,
                                            parameters.buffer_time, parameters.min_queue_size,), None),
                            "stitcher_east": (
                                            ("east", raw_fragment_queue_e, stitched_trajectory_queue,
                                            parameters, ), None),
                            "stitcher_west": (
                                            ("west", raw_fragment_queue_w, stitched_trajectory_queue,
                                            parameters, ), None),
                            # "reconciliation": (
                            #             (parameters, stitched_trajectory_queue, reconciled_queue,), None),
                            # "reconciliation_writer": (
                            #             (parameters, reconciled_queue,), None),
                        }
        
        
        return additional_args



if __name__ == "__main__":
  
    s = ServerControl(register_functions)