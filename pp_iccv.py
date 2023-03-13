#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 18:06:54 2023
- run local processes in parallel for each compute node
- run a master process              
- transform the reconciled collection                                          
"""
from pp1_local import main as pp1_local
from pp1_master import main as pp1_master
from _evaluation.unsup_statistics1 import main as unsup
from transform import main as transform
import json
import multiprocessing as mp


def pipeline(raw_collection="", reconciled_collection=""):
    
    with open("config/parameters.json") as f:
        parameters = json.load(f)
    
    if raw_collection:
        parameters["raw_collection"] = raw_collection
    
    if reconciled_collection:
        parameters["reconciled_collection"] = reconciled_collection
        
    # SHARED DATA STRUCTURES
    mp_manager = mp.Manager()
    mp_param = mp_manager.dict()
    mp_param.update(parameters)
    
    #=== transform raw
    # transform(database_name=mp_param["raw_database"], 
    #           collection_name=mp_param["raw_collection"])
    
    #=== local compute node processing
    pp1_local(raw_collection=mp_param["raw_collection"], 
              reconciled_collection=mp_param["reconciled_collection"])
    
    #=== master (cross compute node) processing
    pp1_master(raw_collection=mp_param["raw_collection"], 
              reconciled_collection=mp_param["reconciled_collection"])

    
    #=== unsupervised statistics for raw
    # unsup(database_name=mp_param["raw_database"], 
    #       collection_name=mp_param["raw_collection"])
    
    #=== unsupervised statistics for rec
    # unsup(database_name=mp_param["reconciled_database"], 
    #       collection_name=mp_param["reconciled_collection"])
    return
    
if __name__ == '__main__':
    pipeline()
    
    
    
    