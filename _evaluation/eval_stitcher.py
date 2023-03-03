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
from datetime import datetime
import matplotlib.dates as md
import json
from bson.objectid import ObjectId
import os

def convert_2_dict_mongodb(obj):
    '''
    convert nested dictionary to "dot" type accepted by mongodb schema
    '''
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

    def flatten(S):
        if S == []:
            return S
        if isinstance(S[0], list):
            return flatten(S[0]) + flatten(S[1:])
        return S[:1] + flatten(S[1:])
        
    for f in raw.find({}):
        f_id = f["_id"]
        try:
            flat = flatten(f["gt_ids"])
        except KeyError:
            pass
        gt_ids_set = set(flat)
        try:
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
        try:
            filter = np.array(f["filter"], dtype=bool)
        except:
            filter = np.array([True]*len(f["timestamp"]))
        # print(f_id, "length: ", len(f["timestamp"]), 'gt_ids: ', f["gt_ids"])
        l = str(f_id)[-4:]
        dates = [datetime.utcfromtimestamp(t) for t in f["timestamp"]]   
        dates=md.date2num(dates)
        x = np.array(f["x_position"])
        y = np.array(f["y_position"])
        axs[0].scatter(dates[filter], x[filter], s=5, marker = "o", label=l)
        axs[1].scatter(dates[filter], y[filter], s=5, marker = "X", label=l)
        axs[2].scatter(x[filter], y[filter], s=5, marker = "X", label=l)
        # axs[3].scatter(dates, f["detection_confidence"], s=5, marker = "X", label=l)
        axs[0].scatter(dates[~filter], x[~filter], s=5, c="lightgrey")
        axs[1].scatter(dates[~filter], y[~filter], s=5, c="lightgrey")
        axs[2].scatter(x[~filter], y[~filter], s=5, c="lightgrey")
        

    axs[0].set_title("time v x")
    axs[1].set_title("time v y")
    axs[2].set_title("x v y")
    # axs[3].set_title("time v confidence")
    axs[0].legend()
    
    return axs

def plot_stitched(rec_ids, rec, raw):
    '''
    plot rec_id and the fragments it stitched together
    '''
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    for rec_id in rec_ids:
        rec_traj = rec.find_one({"_id": rec_id})
        print(rec_traj["fragment_ids"])
        # print(rec_traj["flags"])
        if "post_flag" in rec_traj:
            print("post_flag: ", rec_traj["post_flag"])
        dates = [datetime.utcfromtimestamp(t) for t in rec_traj["timestamp"]]
        dates=md.date2num(dates)
        # axs = plot_traj(rec_traj["fragment_ids"], raw, axs=axs)
        # plt.xticks( rotation=25 )
        axs[0].scatter(dates, rec_traj["x_position"], s=1)
        axs[1].scatter(dates, rec_traj["y_position"],  s=1)
        axs[2].scatter(rec_traj["x_position"], rec_traj["y_position"],  s=1)
        # axs[3].scatter(rec_traj["timestamp"], rec_traj["detection_confidence"],  s=1, c='k')
    
    xfmt = md.DateFormatter('%H:%M:%S')
    for i in range(2):
        axs[i].xaxis.set_major_formatter(xfmt)
        axs[i].tick_params('x', labelrotation=30)
    plt.gcf().subplots_adjust(bottom=0.2)
    return
    
def test_fragments(raw, stitched, eval=None):
    '''
    raw, stitched are collections that follow raw and stitched schema 
    write results to collection (eval) in evaluation database
    '''   
    if eval is not None:
        d = eval.find_one({"collection": stitched._Collection__name})
        if d and "fragments" in d and "id_switches" in d:
            print("already evaluated.")
            fgmt = d["fragments"]
            ids = d["id_switches"]
            print(f"ids_count: {len(ids)}, fgmt_count: {len(fgmt)}")
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
            if len(f["gt_id"]) == 0:
                no_gt_count+=1
            else:
                if len(f["gt_id"]) > 1:
                    mul_gt_count+=1
                for gt_id in f["gt_id"]:
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
    print(f"raw_count: {raw.count_documents({})}, rec_count: {stitched.count_documents({})}")
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


def main(raw_db="transmodeler", rec_db="reconciled", raw_collection=None, rec_collection=None):
    with open(os.path.join(os.environ["USER_CONFIG_DIRECTORY"], "db_param.json")) as f:
        db_param = json.load(f)
        
    # raw_collection = "tm_900_raw_v4" # collection name is the same in both databases
    # rec_collection = "tm_900_raw_v4__1"
    
    dbc = DBClient(**db_param)
    raw = dbc.client[raw_db][raw_collection]
    rec = dbc.client[rec_db][rec_collection]
    eval = dbc.client[rec_db]["evaluation"]
    
    # clean_raw(raw)
    test_fragments(raw, rec, eval)
    return


#%%

if __name__ == '__main__':
    main()

    #%%
    # with open(os.path.join(os.environ["USER_CONFIG_DIRECTORY"], "db_param.json")) as f:
    #     db_param = json.load(f)

    # raw_collection = "tm_900_raw_v4" # collection name is the same in both databases
    # rec_collection = "tm_900_raw_v4__1"
    
    # dbc = DBClient(**db_param)
    # raw = dbc.client["transmodeler"][raw_collection]
    # rec = dbc.client["reconciled"][rec_collection]
    # eval = dbc.client["reconciled"]["evaluation"]
    
    # # clean_raw(raw)
    # test_fragments(raw, rec, eval)
    
    #%%
    # f_ids = [ObjectId('62fd0dc446a150340fcd2195'), ObjectId('62fd0daf46a150340fcd2170'), ObjectId('62fd0dc546a150340fcd2198')]
    # plot_traj(f_ids, raw)

    #%% 
    # rec_ids = [ObjectId('62fd9e4b95f077c66b4d946e'), ObjectId('62fd9e4c95f077c66b4d9471')] # 
    # rec_ids = [ObjectId('6306de96e0c5c896a2a2eec6')]
    # rec_ids = [ObjectId('63fbe697330767fc6c90b084'), ObjectId('63fbe697330767fc6c90b083')]
    # plot_stitched(rec_ids, rec, raw)
    