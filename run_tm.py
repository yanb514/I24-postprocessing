#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 18:06:54 2023

@author: yanbing_wang
This is the postprocessing pipeline for running synthetic data (junyi's)
1. before running this script, ground truth and polluted data has to be written to db
2. parameters.json has to be modified                                                         
"""
from pp1_local import main as pp1_local
from pp1_master import main as pp1_master
from _evaluation.eval_stitcher import main as eval_stitcher
from _evaluation.unsup_statistics1 import main as unsup
from transform import main as transform
import sys
sys.path.insert(0,'../i24_overhead_visualizer')
from time_space import main as time_space
import json


def pipeline(raw_collection="", reconciled_collection=""):
    
    with open("config/parameters.json") as f:
        parameters = json.load(f)
    
    if raw_collection:
        parameters["raw_collection"] = raw_collection
    
    if reconciled_collection:
        parameters["reconciled_collection"] = reconciled_collection
        
    #=== transform raw
    # transform(database_name=parameters["raw_database"], 
    #           collection_name=parameters["raw_collection"])
    
    #=== local compute node processing
    pp1_local(raw_collection=parameters["raw_collection"], 
              reconciled_collection=parameters["reconciled_collection"])
    
    #=== master (cross compute node) processing
    pp1_master(raw_collection=parameters["raw_collection"], 
              reconciled_collection=parameters["reconciled_collection"])
    
    #=== evaluate stitcher
    # eval_stitcher(raw_db=parameters["raw_database"], 
    #               rec_db=parameters["reconciled_database"], 
    #               raw_collection=parameters["raw_collection"], 
    #               rec_collection=parameters["reconciled_collection"])
    
    
    #=== time space ground truth
    # time_space(database_name=parameters["transformed_database"], 
    #             collection_name="tm_900_gt_v4", 
    #             save_path="../figures/")
    
    #=== time space raw
    # time_space(database_name=parameters["transformed_database"], 
    #            collection_name=parameters["raw_collection"], 
    #            save_path="../figures/")
    
    #=== time space reconciled
    # time_space(database_name=parameters["transformed_database"], 
    #             collection_name=parameters["reconciled_collection"], 
    #             save_path="../figures/")
    
    #=== unsupervised statistics for raw
    unsup(database_name=parameters["raw_database"], 
          collection_name=parameters["raw_collection"])
    
    #=== unsupervised statistics for rec
    unsup(database_name=parameters["reconciled_database"], 
          collection_name=parameters["reconciled_collection"])
    return
    
if __name__ == '__main__':
    pipeline()
    
    
    
    