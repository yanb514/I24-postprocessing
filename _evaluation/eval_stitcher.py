#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 11:25:38 2022

@author: yanbing_wang
"""
from i24_database_api import DBClient
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from itertools import islice

def convert_2_dict_mongodb(obj):
    result = {}
    for key, val in obj.items():
        if not isinstance(val, dict):
            result[key] = val
            continue

        for sub_key, sub_val in val.items():
            new_key = '{}.{}'.format(key, sub_key)
            result[new_key] = sub_val
            if not isinstance(sub_val, dict):
                continue

            result.update(convert_2_dict_mongodb(result))
            if new_key in result:
                del result[new_key]

    return result

def clean_raw(raw):
    '''
    clean up raw collection
    raw: raw collection
    update gt_ids field from nested list to a flattened list with no repeated values
    '''
    # check if gt_ids field is already updated (has to be empty or a flattened list instance)
    d = raw.find_one({})
    if "gt_ids" in d: 
        if len(d["gt_ids"]) == 0 or isinstance(d["gt_ids"][0], ObjectId):
            print("collection already cleaned")
            return 
        
    for f in raw.find({}):
        f_id = f["_id"]
        gt_ids_set = set()
        try:
            for l in f["gt_ids"]:
                for gt_id in l:
                    gt_ids_set.add(gt_id) # a fragment may associate to multiple gt_ids
            raw.update_one({"_id": f_id}, { "$set": { 'gt_ids': list(gt_ids_set) } })
        except KeyError:
            raw.update_one({"_id": f_id}, { "$set": { 'gt_ids': [] } })
        except TypeError:
            pass
    print("cleaned raw collection gt_ids field")
            
def plot_traj(veh_ids, dbr, axs = None):
    '''
    Plot fragments with the reconciled trajectory (if col2 is specified)

    Parameters
    ----------
    fragment_list : list of ObjectID fragment _ids
    rec_id: ObjectID (optional)
    '''
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    
    for f_id in veh_ids: 
        f = dbr.find_one({"_id": f_id})
        # print(f_id, "length: ", len(f["timestamp"]), 'gt_ids: ', f["gt_ids"])
        l = str(f_id)[-4:]
        
        axs[0].scatter(f["timestamp"], f["x_position"], s=3, label=l)
        axs[1].scatter(f["timestamp"], f["y_position"], s=3, label=l)
        axs[2].scatter(f["x_position"], f["y_position"], s=3, label=l)
        

    axs[0].set_title("time v x")
    axs[1].set_title("time v y")
    axs[2].set_title("x v y")
    # axs[3].set_title("time v onfidence")
    axs[0].legend()
    return axs

def plot_stitched(rec_ids, rec, raw):
    '''
    plot rec_id and the fragments it stitched together
    '''
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    for rec_id in rec_ids:
        rec_traj = rec.find_one({"_id": rec_id})
        print(rec_traj["fragment_ids"])
        axs = plot_traj(rec_traj["fragment_ids"], raw, axs=axs)
        axs[0].scatter(rec_traj["timestamp"], rec_traj["x_position"], s=1)
        axs[1].scatter(rec_traj["timestamp"], rec_traj["y_position"],  s=1)
        axs[2].scatter(rec_traj["x_position"], rec_traj["y_position"],  s=1)
        # axs[3].scatter(rec_traj["timestamp"], rec_traj["detection_confidence"],  s=1, c='k')
    
    return
    
def test_fragments(raw, stitched, eval=None):
    '''
    raw, stitched are collections that follow raw and stitched schema 
    write results to collection (eval) in evaluation database
    '''   
    if eval is not None:
        d = eval.find_one({})
        if d and "fragments" in d and "id_switches" in d:
            print("already evaluated.")
            return
    
    print("evaluating stitcher results...")    
    st_gt = defaultdict(set) # key: st_id, val (set): gt_ids that correspond to the stitched fragments
    gt_st = defaultdict(set) # key: gt_id, val (set): st_ids that correspond to the gt associated fragments
    no_gt_count = 0
    mul_gt_count = 0
    # save associated st_ids and gt_ids in dictionaries
    for st_doc in stitched.find():
        # corr_gt_ids = set()
        for f_id in st_doc["fragment_ids"]:
            f = raw.find_one({"_id": f_id}) # ObjectId(f_id)
            # TODO: what if f has multiple corresponding gt_ids?
            if len(f["gt_ids"]) == 0:
                no_gt_count+=1
            else:
                if len(f["gt_ids"]) > 1:
                    mul_gt_count+=1
                for gt_id in f["gt_ids"]:
                    st_gt[st_doc["_id"]].add(gt_id)
                    gt_st[gt_id].add(st_doc["_id"])

    # count stuff
    ids_count = 0
    fgmt_count = 0
    ids = defaultdict(set) # record problematic stitching
    fgmt = defaultdict(set)
    
    for st_id, corr_gt_ids in st_gt.items():
        if len(corr_gt_ids) > 1: # a stitched id corresponds to multiple gt_ids, overstitch
            ids_count += len(corr_gt_ids)-1
            [ids[st_id].add(corr_gt_id) for corr_gt_id in corr_gt_ids]
    for gt_id, corr_st_ids in gt_st.items():
        if len(corr_st_ids) > 1: # a gt_id corresponds to multiple stitched ids, understitch
            fgmt_count += len(corr_st_ids)-1
            [fgmt[gt_id].add(corr_st_id) for corr_st_id in corr_st_ids]
    
    for key, val in fgmt.items():
        fgmt[key] = list(val)
    for key, val in ids.items():
        ids[key] = list(val)
    print(f"raw_count: {raw.count_documents({})}, rec_count: {rec.count_documents({})}")
    print(f"no_gt_count: {no_gt_count}, mul_gt_count: {mul_gt_count}")
    print(f"gt_count: {len(gt_st)}, stitched_count: {len(st_gt)}")
    print(f"ids_count: {ids_count}, fgmt_count: {fgmt_count}")
    
    # write result to database
    if eval is not None:
        obj = {"fragments": fgmt, "id_switches": ids}
        obj = convert_2_dict_mongodb(obj)
        
        eval.update_one({"collection": stitched._Collection__name}, 
                        {"$set": obj},
                        upsert = True)
        print("written eval result to db")
    return


#%%

if __name__ == '__main__':
    
    import json
    from bson.objectid import ObjectId
     
    with open("config.json") as f:
        config = json.load(f)

    trajectory_database = "trajectories"
    raw_collection = "demure_wallaby--RAW_GT1"
    rec_collection = "demure_wallaby--RAW_GT1__improvises"
    
    dbc = DBClient(**config)
    raw = dbc.client[trajectory_database][raw_collection]
    rec = dbc.client["reconciled"][rec_collection]
    eval = dbc.client["reconciled"]["evaluation"]
    
    clean_raw(raw)
    test_fragments(raw, rec, eval)
    eval_doc = eval.find_one({"collection": rec_collection})
    fgmt = eval_doc["fragments"]
    ids = eval_doc["id_switches"]
    
    #%%
    # visualize understitch
    for gt_id, corr_st_ids in islice(fgmt.items(), 4,7):  # use islice(d.items(), 3) 
        plot_stitched(corr_st_ids, rec, raw)

    