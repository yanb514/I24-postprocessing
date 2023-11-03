#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 18:06:54 2023

@author: yanbing_wang
This is the postprocessing pipeline for running synthetic data (junyi's)
1. before running this script, ground truth and polluted data has to be written to db
2. parameters.json has to be modified                                                         
"""
from pp1_all_nodes import main as pp1_all_nodes
from pp1_local import main as pp1_local
from pp1_master import main as pp1_master
from pp_lite import main as pp_lite
from pp_lite_reverse import main as pp_lite_reverse
from _evaluation.eval_stitcher import main as eval_stitcher
from _evaluation.unsup_statistics1 import main as unsup
from _evaluation.unsup_statistics1 import print_res, plot_dyn_dist
from _evaluation.unsup_binned import main as unsup_binned
from transform import main as transform
import json
import sys
sys.path.insert(0,'../i24_visualizer')
sys.path.insert(0,'../trajectory-eval-toolkit-main')
from time_space import main as time_space
from time_space import main_by_lane as time_space_lane

from evaluate_ICCV import main as sup_eval

def pipeline(command, raw_collection="", reconciled_collection=""):
    
    with open("config/parameters.json") as f:
        parameters = json.load(f)
    
    if raw_collection:
        parameters["raw_collection"] = raw_collection
    
    if reconciled_collection:
        parameters["reconciled_collection"] = reconciled_collection

    #=== run 
    if command["pp_lite"]:
        pp_lite(raw_collection=parameters["raw_collection"], 
                  reconciled_collection=parameters["reconciled_collection"])
    
    if command["pp_lite_reverse"]:
        pp_lite_reverse(raw_collection=parameters["raw_collection"], 
                  reconciled_collection=parameters["reconciled_collection"])
        

    if command["pp1_all_nodes"]:
        pp1_all_nodes(raw_collection=parameters["raw_collection"], 
                  reconciled_collection=parameters["reconciled_collection"])

    #=== local compute node processing
    if command["pp1_local"]:
        pp1_local(raw_collection=parameters["raw_collection"], 
                  reconciled_collection=parameters["reconciled_collection"])

    #=== master (cross compute node) processing
    if command["pp1_master"]:
        pp1_master(raw_collection=parameters["raw_collection"], 
                  reconciled_collection=parameters["reconciled_collection"])
    
    #=== transform gt
    if command["transform_gt"]:
        transform(database_name=parameters["raw_database"], 
                  collection_name=parameters["gt_collection"])

    #=== transform raw
    if command["transform_raw"]:
        transform(database_name=parameters["raw_database"], 
              collection_name=parameters["raw_collection"])

    #=== transform rec
    if command["transform_rec"]:
        transform(database_name=parameters["reconciled_database"], 
              collection_name=parameters["reconciled_collection"])
    
    #=== transform temp
    if command["transform_temp"]:
        transform(database_name=parameters["temp_database"], 
                  collection_name=parameters["reconciled_collection"],
                  write_collection_name=parameters["reconciled_collection"]+"_temp")

    #=== evaluate stitcher raw
    if command["eval_stitcher_raw"]:
        eval_stitcher(raw_db=parameters["raw_database"], 
                      rec_db=parameters["raw_database"], 
                      raw_collection=parameters["raw_collection"], 
                      rec_collection=parameters["raw_collection"]   
                      )

    #=== evaluate stitcher temp
    if command["eval_stitcher_temp"]:
        eval_stitcher(raw_db=parameters["raw_database"], 
                      rec_db=parameters["temp_database"], 
                      raw_collection=parameters["raw_collection"], 
                      rec_collection=parameters["reconciled_collection"] 
                      # rec_collection=parameters["raw_collection"]   
                      )
    
    #=== evaluate stitcher rec
    if command["eval_stitcher_rec"]:
        eval_stitcher(raw_db=parameters["raw_database"], 
                      rec_db=parameters["reconciled_database"], 
                      raw_collection=parameters["raw_collection"], 
                      rec_collection=parameters["reconciled_collection"] 
                      # rec_collection=parameters["raw_collection"]   
                      )
        

    #=== supervised metrics raw
    if command["sup_eval_raw"]:
        sup_eval(gt_coll = parameters["gt_collection"],
                 coll_names = [parameters["raw_collection"]],
                 db_list = [parameters["raw_database"]]
                )
    #=== supervised metrics rec
    if command["sup_eval_rec"]:
        sup_eval(gt_coll = parameters["gt_collection"],
                 coll_names = [parameters["reconciled_collection"]],
                 db_list = [parameters["reconciled_database"]]
                )
    #=== supervised metrics temp
    if command["sup_eval_temp"]:
        sup_eval(gt_coll = parameters["gt_collection"],
                 coll_names = [parameters["temp_collection"]],
                 db_list = [parameters["temp_database"]]
                )
        
        
    #=== time space ground truth
    if command["time_space_gt"]:
        time_space(
                    database_name=parameters["transformed_database"], 
                    # database_name=parameters["raw_database"], 
                    collection_name=parameters["gt_collection"], 
                    save_path="../figures/")
    
    #=== time space raw
    if command["time_space_raw"]:
        time_space(
                    database_name=parameters["transformed_database"], 
                    # database_name=parameters["raw_database"], 
                    collection_name=parameters["raw_collection"], 
                    save_path="../figures/")
    
    # === time space reconciled
    if command["time_space_rec"]:
        time_space(
                    database_name=parameters["transformed_database"], 
                    # database_name=parameters["reconciled_database"], 
                    collection_name=parameters["reconciled_collection"], 
                    save_path="../figures/")

    #=== time space temp
    if command["time_space_temp"]:
        time_space(
                    database_name=parameters["transformed_database"], 
                    # database_name=parameters["temp_database"], 
                    collection_name=parameters["reconciled_collection"],
                    save_path="../figures/")
    
    #=== time space by lane
    if command["time_space_lane"]:
        time_space_lane(
                    database_name=parameters["raw_database"], 
                    collection_name=parameters["gt_collection"],
                    labels=["GT"],
                    save_path="../figures/")
        
    #=== time space by lane compare GT vs RAW
    if command["gtraw"]:
        time_space_lane(
                    database_name=[parameters["raw_database"], parameters["raw_database"]], 
                    collection_name=[parameters["gt_collection"], parameters["raw_collection"]],
                    labels = ["GT", "RAW"],
                    save_path="../figures/")
    
    #=== time space by lane compare GT vs RAW
    if command["gtrec"]:
        time_space_lane(
                    database_name=[parameters["raw_database"], parameters["reconciled_database"]], 
                    collection_name=[parameters["gt_collection"], parameters["reconciled_collection"]],
                    labels = ["GT", "REC"],
                    save_path="../figures/")
        
    #=== time space by lane compare GT vs RAW
    if command["all"]:
        time_space_lane(
                    database_name=[parameters["raw_database"], parameters["reconciled_database"], parameters["raw_database"]], 
                    collection_name=[parameters["gt_collection"], parameters["reconciled_collection"], parameters["raw_collection"]],
                    labels = ["GT", "REC", "RAW" ],
                    save_path="../figures/")
        
    #=== unsupervised statistics for GT
    if command["unsup_gt"]:
        res = unsup(database_name=parameters["raw_database"], 
              collection_name=parameters["gt_collection"])
        print_res(res)

    #=== unsupervised statistics for raw
    if command["unsup_raw"]:
        res = unsup(database_name=parameters["raw_database"], 
              collection_name=parameters["raw_collection"],
                   time_sr=command["time_sr"])
        print_res(res)
    
    #=== unsupervised statistics for rec
    if command["unsup_rec"]:
        res = unsup(database_name=parameters["reconciled_database"], 
              collection_name=parameters["reconciled_collection"],
                   time_sr=command["time_sr"])
        print_res(res)
        
    
    #=== plot comparison of distributions raw vs. rec
    if command["compare_distribution"]:
        gt_pkl_name = "{}_{}".format(parameters["raw_database"], parameters["gt_collection"])
        raw_pkl_name = "{}_{}".format(parameters["raw_database"], parameters["raw_collection"])
        rec_pkl_name = "{}_{}".format(parameters["reconciled_database"], parameters["reconciled_collection"])
        res_name_list = [gt_pkl_name, raw_pkl_name, rec_pkl_name]
        plot_dyn_dist(res_name_list=res_name_list, labels = ["GT", "RAW", "REC"], save="../figures/")

    #=== unsupervised statistics (binned) for raw
    if command["unsup_binned_raw"]:
        unsup_binned(db_list=[parameters["raw_database"]], 
              col_list=[parameters["raw_collection"]])
    
    #=== unsupervised statistics (binned) for rec
    if command["unsup_binned_rec"]:
        unsup_binned(db_list=[parameters["reconciled_database"]], 
              col_list=[parameters["reconciled_collection"]])
    return
    
    
if __name__ == '__main__':
    command = {
        "pp_lite":             0,
        "pp_lite_reverse":     0,
        "pp1_all_nodes":       0,
            "pp1_local":       0,
            "pp1_master":      0,
        "transform_gt":        0,
        "transform_raw":       0,
        "transform_rec":       0,
        "transform_temp":      0,
        "eval_stitcher_raw":   0,
        "eval_stitcher_rec":   0,
        "eval_stitcher_temp":  0,

        "sup_eval_raw":        0,
        "sup_eval_rec":        1,
        "sup_eval_temp":       0,
        
        "unsup_gt":            0,
        "unsup_raw":           0, # update time_sr
        "unsup_rec":           0, # update time_sr
            "time_sr" :        0.04,

        "time_space_gt":       0,
        "time_space_raw":      0,
        "time_space_rec":      0,
        "time_space_temp":     0,
        "time_space_lane":     0, # plot GT single collection
            "gtraw":           0, # co mpare two collections
            "gtrec":           0,
            "all":             0,

        "compare_distribution":0,
        "unsup_binned_raw":    0,
        "unsup_binned_rec":    0
    }
    
    pipeline(command)
    
    
    
    