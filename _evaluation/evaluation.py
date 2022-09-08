#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 11:58:16 2022

@author: yanbing_wang
TODO:
    keep track of what fragments are deleted during the processes
    
"""
import time
import queue
from i24_database_api import DBClient
import i24_logger.log_writer as log_writer
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import pprint
try:
    from .unsup_statistics import _calc_feasibility, UnsupervisedEvaluator
except ImportError:
    from unsup_statistics import _calc_feasibility, UnsupervisedEvaluator

import sys
sys.path.append("..")
from utils.utils_opt import combine_fragments


def update_raw_eval(res, traj_col, eval_col,x_min, x_max):
    '''
    add feasibility information to db
    '''

    eval_col.drop() # reset collection
    
    # update conflicts
    for pair, occurances in res["overlaps"].items():
        id1, id2 = pair
        eval_col.update_one({"_id": id1}, 
                        {"$push": {"conflicts": [id2, occurances]}},
                        upsert = True)
        eval_col.update_one({"_id": id2}, 
                        {"$push": {"conflicts": [id1, occurances],
                                   }},
                        upsert = True)
        
    # update feasibility
    cursor = traj_col.find({})
    for traj in cursor:
        feas = _calc_feasibility(traj, xmin=x_min, xmax=x_max)
        dist, backward, rotation, acceleration, conflict = feas
        eval_col.update_one({"_id": traj["_id"]}, 
                        {"$set": {"feasibility.distance": dist,
                                  "feasibility.backward": backward,
                                  "feasibility.rotation": rotation,
                                  "feasibility.acceleration": acceleration,
                                  "feasibility.conflict": conflict,
                                   }},
                        upsert = True)
          
    print("Updated feasibility for {}".format(traj_col._Collection__name))
        




        
def eval_raw(parameters, db_param):
    '''
    transform to time-index db -> parameters["raw_collection"]
    calculate feasibility score of each trajectory
    calculate conflicts
    write results to database "evaluation" with the same collection name following schema defined in "eval_schema.json" in config
    '''
    while parameters["raw_collection"] == "":
        time.sleep(1)
        

    read_dbc = DBClient(**db_param, database_name = parameters["raw_database"], collection_name = parameters["raw_collection"])
    write_dbc = DBClient(**db_param, database_name = parameters["eval_database"], collection_name = parameters["raw_collection"]) # evaluation collection
         
    if parameters["raw_collection"] in write_dbc.list_collection_names():
        print(parameters["raw_collection"] ," is already evaluated.")    
        return   
                       
    # res has conflicts 

    print("in time evaluation")
    db_param["database_name"] = parameters["raw_database"]
    ue = UnsupervisedEvaluator(db_param, collection_name=parameters["raw_collection"])
    ue.time_evaluate(step=1)
    db_param.pop("database_name") 
    print("writing result to evaluation db...")
    
    # start_time = read_dbc.get_min("first_timestamp")
    # end_time = read_dbc.get_max("last_timestamp")
    x_min = read_dbc.get_min("starting_x")
    x_max = read_dbc.get_max("ending_x")
    
    # calc_feasibility and write to db
    update_raw_eval(ue.res, read_dbc.collection, write_dbc.collection,  x_min, x_max)
    
    return


def eval_reconcile(parameters, db_param):
    '''
    update eval.reconciled_collection["merge"] due to merging
    directly run unsup_statistics on parameters["reconciled_collection"]
    write eval results (feasibility, conflicts) in eval.reconciled_collection[]
    a rec_doc should have merged_ids:[[1,2],[3]]
    fragment_ids:[1,3]
    '''
    logger = log_writer.logger 
    logger.set_name("_eval_reconcile")
    setattr(logger, "_default_logger_extra",  {})
    
    # while parameters["raw_collection"] == "":
    #     time.sleep(1)   
    # while parameters["reconciled_collection"] == "":
    #     time.sleep(1)
    
    dbc = DBClient(**db_param) 
    raw = dbc.client[parameters["raw_database"]][parameters["raw_collection"]]
    rec = dbc.client[parameters["reconciled_database"]][parameters["reconciled_collection"]]
    raw_eval = dbc.client[parameters["eval_database"]][parameters["raw_collection"]]
    rec_eval = dbc.client[parameters["eval_database"]][parameters["reconciled_collection"]]
    
    x_min = raw.find_one(sort=[("starting_x", 1)])["starting_x"]
    x_max = raw.find_one(sort=[("ending_x", -1)])["ending_x"]
    
    
    # reset fields
    rec.update_many({},{"$unset": {"conflicts": "",
                                   "merge": "",
                                   "stitch": "",
                                   "reconcile": "",
                                   "feasibility": "",
                                   } })
    rec_eval.drop()
    
    
    # update merge
    print("Updating eval.merge...")
    for rec_doc in rec.find({}):
        merged_ids = rec_doc["merged_ids"]
        for merged in merged_ids:
            
            cursors = raw.find({"_id": {"$in": merged}})
            xmin = x_max
            xmax = x_min
            first = []
            last = []
            
            for d in cursors:
                xmin = min(xmin, min(d["starting_x"], d["ending_x"]))
                xmax = max(xmax, max(d["starting_x"], d["ending_x"]))
                first.append(d["first_timestamp"])
                last.append(d["last_timestamp"])     
                    
            if len(merged) == 1:
                # not merged with others
                # copy from raw_eval to merge
                _id = merged[0]
                raw_eval_doc = raw_eval.find_one({"_id": _id})
                try:
                    rec_eval.update_one({"_id": _id}, 
                                    {"$set": { 
                                        "merge.merged_ids": merged,
                                        "merge.conflicts": raw_eval_doc["conflicts"],
                                        "merge.feasibility": raw_eval_doc["feasibility"],
                                        "merge.first_timestamp": min(first),
                                        "merge.last_timestamp": max(last),
                                        "merge.xmax": xmax,
                                        "merge.xmin": xmin
                                                }},
                                    upsert = True)
                except KeyError:
                    rec_eval.update_one({"_id": _id}, 
                                    {"$set": { 
                                        "merge.merged_ids": merged,
                                        "merge.feasibility": raw_eval_doc["feasibility"],
                                        "merge.first_timestamp": min(first),
                                        "merge.last_timestamp": max(last),
                                        "merge.xmax": xmax,
                                        "merge.xmin": xmin
                                                }},
                                    upsert = True)
                
            else:
                # recalculate stuff to write to eval.merge
                # find the id that's kept to stitching stage
                _id = [m for m in merged if m in rec_doc["fragment_ids"]][0]
                # update distance score
                distance_score = min(1, (xmax-xmin)/(x_max-x_min))
                
                # update conflicts - remove conflicts of _id if they are merged with _id
                # union of all conflicts with merged_ids minus merged_ids themselves
                all_conf_ids = {}
                feas_list = []
                eval_docs = raw_eval.find({"_id": {"$in": merged}})
                
                for eval_doc in eval_docs:
                    feas_list.append(eval_doc["feasibility"])
                    if "conflicts" in eval_doc: # and eval_doc["_id"] not in merged:
                        for k,v in eval_doc["conflicts"]:
                            if k not in merged:
                                all_conf_ids[k] = v # v is not accurate, becuase we do not count when they conflict
                
                    
                conflicts = [[k,v] for k,v in all_conf_ids.items()]

                # update conflict score
                duration = max(last) - min(first)
                try:
                    time_conflict = max([item[1] for item in conflicts])
                except (KeyError, ValueError):
                    time_conflict = 0
                conflict_score = 1-time_conflict/duration
                
                # feasibility = raw_eval_doc["feasibility"]
                feasibility = {}
                feasibility["rotation"] = sum([d["rotation"] for d in feas_list])/len(feas_list)
                feasibility["acceleration"] = sum([d["acceleration"] for d in feas_list])/len(feas_list)
                feasibility["backward"] = sum([d["backward"] for d in feas_list])/len(feas_list)
                feasibility["conflict"] = conflict_score
                feasibility["distance"] = distance_score
                    
                rec_eval.update_one({"_id": _id}, 
                                {"$set": { 
                                    "merge.merged_ids": merged,
                                    "merge.conflicts": conflicts,
                                    "merge.feasibility": feasibility,
                                    "merge.first_timestamp": min(first),
                                    "merge.last_timestamp": max(last),
                                    "merge.xmax": xmax,
                                    "merge.xmin": xmin
                                            }},
                                upsert = True)
            
    
    # update stitch
    print("Updating eval.stitch...")
    for rec_doc in rec.find({}):
        if len(rec_doc["fragment_ids"]) > 1:
            stitched_ids = rec_doc["fragment_ids"]
            _id = stitched_ids[0]
            cursors = rec_eval.find({"_id": {"$in": stitched_ids}})
            xmin = x_max
            xmax = x_min
            first = []
            last = []
            
            for d in cursors:
                xmin = min(xmin, d["merge"]["xmin"]) # keyerror if d has no merge (copied feasibility)
                xmax = max(xmax, d["merge"]["xmax"])
                first.append(d["merge"]["first_timestamp"])
                last.append(d["merge"]["last_timestamp"])
    
            distance_score = min(1, (xmax-xmin)/(x_max-x_min))
            
            # update conflicts - remove conflicts of _id if they are merged with _id
            # union of all conflicts with merged_ids minus merged_ids themselves
            all_conf_ids = {}
            feas_list = []
            eval_docs = rec_eval.find({"_id": {"$in": stitched_ids}})
            
            for eval_doc in eval_docs:
                feas_list.append(eval_doc["merge"]["feasibility"])
                if "conflicts" in eval_doc["merge"]: # and eval_doc["_id"] not in merged:
                    for k,v in eval_doc["merge"]["conflicts"]:
                        if k not in stitched_ids:
                            all_conf_ids[k] = v # v is not accurate, becuase we do not count when they conflict
            
                
            conflicts = [[k,v] for k,v in all_conf_ids.items()]
    
            # update conflict score
            duration = max(last) - min(first)
            try:
                time_conflict = max([item[1] for item in conflicts])
            except (KeyError, ValueError):
                time_conflict = 0
            conflict_score = 1-time_conflict/duration
            
            # feasibility = raw_eval_doc["feasibility"]
            feasibility = {}
            feasibility["rotation"] = sum([d["rotation"] for d in feas_list])/len(feas_list)
            feasibility["acceleration"] = sum([d["acceleration"] for d in feas_list])/len(feas_list)
            feasibility["backward"] = sum([d["backward"] for d in feas_list])/len(feas_list)
            feasibility["conflict"] = conflict_score
            feasibility["distance"] = distance_score
                
            rec_eval.update_one({"_id": _id}, 
                            {"$set": { 
                                "stitch.fragment_ids": stitched_ids,
                                "stitch.conflicts": conflicts,
                                "stitch.feasibility": feasibility,
                                "stitch.first_timestamp": min(first),
                                "stitch.last_timestamp": max(last),
                                "stitch.xmax": xmax,
                                "stitch.xmin": xmin
                                        }},
                            upsert = True)

    
    # update reconcile
    print("Updating eval.reconcile")
    db_param["database_name"] = parameters["reconciled_database"]
    ue = UnsupervisedEvaluator(db_param, collection_name=parameters["reconciled_collection"])
    ue.time_evaluate(step=1)
    db_param.pop("database_name") 
    
    
    # calc_feasibility and write to db
    
    # update conflicts
    for pair, occurances in ue.res["overlaps"].items():
        id1, id2 = pair
        # have to update to rec collection first, because calc_feasibility uses traj["conflicts"] information
        rec.update_one({"_id": id1}, 
                        {"$push": {"conflicts": [id2, occurances]}},
                        upsert = True)
        rec.update_one({"_id": id2}, 
                        {"$push": {"conflicts": [id1, occurances],
                                   }},
                        upsert = True)
        
        rec_eval.update_one({"_id": id1}, 
                        {"$push": {"reconcile.conflicts": [id2, occurances]}},
                        upsert = True)
        rec_eval.update_one({"_id": id2}, 
                        {"$push": {"reconcile.conflicts": [id1, occurances],
                                   }},
                        upsert = True)
        
    # update feasibility in eval.reconcile
    cursor = rec.find({})
    for traj in cursor:
        feas = _calc_feasibility(traj, xmin=x_min, xmax=x_max)
        dist, backward, rotation, acceleration, conflict = feas
        rec_eval.update_one({"_id": traj["_id"]}, 
                        {"$set": {"reconcile.feasibility.distance": dist,
                                  "reconcile.feasibility.backward": backward,
                                  "reconcile.feasibility.rotation": rotation,
                                  "reconcile.feasibility.acceleration": acceleration,
                                  "reconcile.feasibility.conflict": conflict,
                                   }},
                        upsert = True)
        rec.update_one({"_id": traj["_id"]}, 
                        {"$set": {"feasibility.distance": dist,
                                  "feasibility.backward": backward,
                                  "feasibility.rotation": rotation,
                                  "feasibility.acceleration": acceleration,
                                  "feasibility.conflict": conflict,
                                   }},
                        upsert = True)
          
    print("Updated feasibility for {}".format(rec._Collection__name))

    return



def report(eval_raw, eval_rec):
    '''
    - total number of docs at each stage
        - # pre-filtered
        - # merged
        - # stitched
        - # feasible trajectories to start with
        - # removed to resolve conflicts
        - # feasible trajectories left
    '''
    res = {}
    # total number of raw
    num_raw = eval_raw.count_documents({})
    
    # count after merging - merged_ids have more than 1 element (NOT ACCURATE!)
    num_m = eval_rec.count_documents( {"merge": {"$exists": True}} )
    
    # count after stitching
    num_st = eval_rec.count_documents( {"reconcile": {"$exists": True}} )
    
    # count after stitching
    num_rec = eval_rec.count_documents( {"reconcile": {"$exists": True}} )
    
    # count conflict raw
    num_conf_raw = eval_raw.count_documents({"conflicts": {"$exists": True}})
    
    # count conflict merge
    num_conf_m = eval_rec.count_documents({"merge.conflicts": {"$exists": True}})
    
    # count conflict stitch
    num_conf_st = eval_rec.count_documents({"stitch.conflicts": {"$exists": True}})
    
    # count conflict reconcile
    num_conf_rec = eval_rec.count_documents({"reconcile.conflicts": {"$exists": True}})
    
    # write to res
    res["number_of_raw"] = num_raw
    res["number_of_merged"] = num_m
    res["number_of_stitched"] = num_st
    res["number_of_reconciled"] = num_rec
    res["number_of_conflicts_raw"] = num_conf_raw
    res["number_of_conflicts_merged"] = num_conf_m
    res["number_of_conflicts_stitched"] = num_conf_st
    res["number_of_conflicts_reconciled"] = num_conf_rec
    
    pprint.pprint(res)
    return


def plot_stage_hist(rec_collection, db_param, field_name_list="all"):
    '''
    field_name_list: list of field names
    '''
    dbc = DBClient(**db_param, database_name = "evaluation", collection_name = rec_collection)
    eval_raw = dbc.db[rec_collection.split("__")[0]]
    eval_rec = dbc.collection
    
    if field_name_list == "all":
        field_name_list = ["acceleration", "rotation", "backward", "conflict", "distance"]
    
    fig, ax = plt.subplots(1,len(field_name_list), figsize = (20,5))
    i = 0
    bins = 100
    
    for field_name in field_name_list:
        print("Getting data for {}".format(field_name))
        raw_data = []
        for eval in eval_raw.find({}):
            raw_data.append(eval["feasibility"][field_name])
        data = defaultdict(list)
        for eval in eval_rec.find({}):
            data["merge"].append(eval["merge"]["feasibility"][field_name])
            
            try:
                data["reconcile"].append(eval["reconcile"]["feasibility"][field_name])
                try: # docs that have "reconcile" field are the merged ones, retrieve info from "stitch", if not successful, then retrive from "merge"
                    data["stitch"].append(eval["stitch"]["feasibility"][field_name])
                except KeyError:
                    data["stitch"].append(eval["merge"]["feasibility"][field_name])
            except KeyError:
                pass
            
        # plot
        # print(len(raw_data), len(data["merge"]), len(data["stitch"]), len(data["reconcile"]))
        ax[i].hist(raw_data, alpha=0.3, bins=bins, label = "raw")
        ax[i].hist(data["merge"], bins=bins, alpha=0.3, label = "merge")
        ax[i].hist(data["stitch"], bins=bins, alpha=0.3,label = "stitch")
        ax[i].hist(data["reconcile"], bins=bins, alpha=0.3, label = "reconcile")
        ax[i].set_title(field_name)
        ax[i].legend()
        i += 1
        
    
            
def conflict_graph(collection):
    '''
    visualize the relationship between conflicts
    '''
    
    G = nx.DiGraph()
    queries = collection.find({"conflicts":{"$exists": True}})
    # queries = collection.find({"$or": [{"feasibility.distance":{"$lte":0.7}},
    #                                   {"conflicts":{"$exists":True}}]})
    
    for traj in queries:
        # if traj["feasibility"]["distance"] < 0.7:
        # if len(traj["merged_ids"]) > len(traj["fragment_ids"]):
        #     color = "red"
        # else:
        
        color = "#210070"
        G.add_node(traj["_id"], weight=len(traj["timestamp"]), color=color)
        try:
            edges = traj["conflicts"]
            dur = traj["last_timestamp"]-traj["first_timestamp"]
            for nei, edge_weight in edges:
                if not collection.find_one({"_id": nei}):
                    continue
                G.add_edge(traj["_id"], nei, weight=edge_weight/dur)
        except KeyError: # no conflicts
            pass
    
    # clean edges to keep only the larger weight of bi-direction
    remove_list = []
    for u,v,w in G.edges(data="weight"):
        try:
            if w > G.edges[(v,u)]["weight"]:
                remove_list.append((v,u))
            else:
                remove_list.append((u,v))
        except KeyError:
            pass
    G.remove_edges_from(remove_list)

    # viz_graph(G, collection._Collection__name)
    return G


def delete_conflict(rec_collection, eval_collection):
    '''
    delete nodes from the highest degree first until no edges left in G
    '''
    G = conflict_graph(rec_collection)
    cntr = 0
    to_remove = []
    while G.number_of_edges() > 0:
        # sort by degree
        sorted_d = sorted(G.degree, key=lambda x: x[1], reverse=True) # [(node, degree)]
        highest = sorted_d[0][0]
        
        # delete the node of the highest degree
        G.remove_node(highest)
        cntr += 1
        to_remove.append(highest)
        
    print(f"Deleted {cntr} nodes to resolve all conflicts")
    
    # viz_graph(G, collection._Collection__name)   
    # remove from database collection
    # TODO: delete "conflicts" field as well
    query = {"_id": {"$in": to_remove}}
    d1 = rec_collection.delete_many(query)
    # d2 = collection.update_many({"conflicts": {"$exists":True}},
    #                         {"$pull": {"conflicts": {"$elemMatch": 
    #                                                  {"0": {"$in": to_remove}}}},
    #                          })
    d2 = rec_collection.update_many({},{"$unset": {"conflicts": "",
                                   } })
    d3 = eval_collection.update_many({},{"$unset": {"reconcile.conflicts": "",
                                    } })
    
    print(d1.deleted_count, " documents deleted, ", d2.modified_count, " documents modified.")
    
    return G, to_remove





def viz_graph(G, collection_name):
    # visualize the graph
    fig, ax = plt.subplots(figsize=(12, 12))
    # Generate layout for visualization
    # pos = nx.kamada_kawai_layout(G)
    pos = nx.spring_layout(G, weight=0.00001)
    # Visualize graph components
    edgewidth = [e[2]*10 for e in G.edges(data="weight")]
    nodesize = [v for _,v in G.nodes(data="weight")]
    nodecolor = [v for _,v in G.nodes(data="color")]
    
    nx.draw_networkx_edges(G, pos, alpha=0.7, width=edgewidth, edge_color="m")
    nx.draw_networkx_nodes(G, pos, node_size=nodesize, node_color=nodecolor, alpha=0.5)
    # label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
    # nx.draw_networkx_labels(G, pos, font_size=6, bbox=label_options)


    # Title/legend
    font = {"fontname": "Helvetica", "color": "k", "fontweight": "bold", "fontsize": 10}
    ax.set_title("Conflict graph for {}".format(collection_name), font)
    # Change font color for legend
    font["color"] = "k"
    
    ax.text(
        0.80,
        0.10,
        "edge width = conflicting time/traj duration",
        horizontalalignment="center",
        transform=ax.transAxes,
        fontdict=font,
    )
    ax.text(
        0.80,
        0.08,
        "node size = traj length",
        horizontalalignment="center",
        transform=ax.transAxes,
        fontdict=font,
    )
    ax.text(
        0.80,
        0.06,
        "# nodes: {}, # edges: {}".format(len(G.nodes), len(G.edges)),
        horizontalalignment="center",
        transform=ax.transAxes,
        fontdict=font,
    )
    ax.text(
        0.80,
        0.04,
        "red nodes: short tracks (<0.7)",
        horizontalalignment="center",
        transform=ax.transAxes,
        fontdict=font,
    )
    
    # Resize figure for label readibility
    ax.margins(0.1, 0.05)
    fig.tight_layout()
    plt.axis("off")
    plt.show()

    
               

if __name__ == "__main__":
    # print("not implemented")
    import json
    import os
    
    with open("../config/parameters.json") as f:
        parameters = json.load(f)
    
    parameters["raw_collection"] = "organic_forengi--RAW_GT2"
    parameters["reconciled_collection"] = "organic_forengi--RAW_GT2__protests"
    
    with open(os.path.join(os.environ["USER_CONFIG_DIRECTORY"], "db_param.json")) as f:
        db_param = json.load(f)
        
    raw = DBClient(**db_param, database_name = "trajectories", collection_name=parameters["raw_collection"] )
    rec = DBClient(**db_param, database_name = "reconciled", collection_name=parameters["reconciled_collection"] )
    raw_eval = DBClient(**db_param, database_name = "evaluation", collection_name=parameters["raw_collection"] )
    rec_eval = DBClient(**db_param, database_name = "evaluation", collection_name=parameters["reconciled_collection"] )
    
    rec.transform()
    eval_reconcile(parameters, db_param)
    # plot_stage_hist(parameters["reconciled_collection"], db_param)
    G = conflict_graph(rec.collection)
    viz_graph(G, parameters["reconciled_collection"])
    G, to_remove = delete_conflict(rec.collection, rec_eval.collection)
    viz_graph(G, parameters["reconciled_collection"])
    
    report(raw_eval.collection, rec_eval.collection)
    
                
   

