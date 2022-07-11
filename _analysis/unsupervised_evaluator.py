#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 21:45:29 2022

@author: yanbing_wang
Get the statistics of a collection
- tmin/tmax/xymin/xymax/# trajectories

Compare raw and reconciled (unsupervised)
- what fragments are filtered out
- unmatched fragments
- y deviation
- speed distribution
- starting / ending x distribution
- collision
- length, width, height distribution
- density? flow?

Examine problematic stitching
- plot a list of fragments
- plot the reconciled trajectories

Statics output write to
- DB (?)
- file
- log.info(extra={})
"""

from i24_database_api.db_reader import DBReader
from i24_database_api.db_writer import DBWriter
from i24_configparse import parse_cfg
import matplotlib.pyplot as plt
import os
import warnings
from bson.objectid import ObjectId
import pprint

class UnsupervisedEvaluator():
    
    def __init__(self, config, collection1, collection2=None):
        '''
        Parameters
        ----------
        config : Dictionary
            store all the database-related parameters.
        collection1 : str
            Collection name.
        collection2 : str, optional
            Collection name. The default is None.
        '''
        self.col1_name = collection1
        self.config = config
        self.col1 = DBReader(config, collection_name=collection1)
        if collection2: # collection2 is optional
            self.col2 = DBReader(config, collection_name=collection2)
            self.col2_name = collection2
        else:
            self.col2 = None
            
        print("connected to pymongo client")
            
    def get_collection_info(self):
        """
        To set the bounds on query and on visualization
        """
        col1 = self.col1
        self.col1_info = {
            "count": col1.count(),
            "min_first_time": col1.get_min("first_timestamp"),
            "max_last_time": col1.get_max("last_timestamp"),
            "min_last_time": col1.get_min("last_timestamp"),
            "max_first_time": col1.get_max("first_timestamp"),
            "min_x": min(col1.get_min("starting_x"), col1.get_min("ending_x"),col1.get_max("starting_x"), col1.get_max("ending_x")),
            "max_x": max(col1.get_max("starting_x"), col1.get_max("ending_x"),col1.get_min("starting_x"), col1.get_min("ending_x"))
            }  
        if self.col2:
            col2 = self.col2
            self.col2_info = {
                "count": col2.count(),
                "min_first_time": col2.get_min("first_timestamp"),
                "max_last_time": col2.get_max("last_timestamp"),
                "min_last_time": col2.get_min("last_timestamp"),
                "max_first_time": col2.get_max("first_timestamp"),
                "min_x": min(col2.get_min("starting_x"), col2.get_min("ending_x"),col2.get_max("starting_x"), col2.get_max("ending_x")),
                "max_x": max(col2.get_max("starting_x"), col2.get_max("ending_x"),col2.get_min("starting_x"), col2.get_min("ending_x"))
                } 
        else:
            self.col2_info = None
            
            
        
    def delete_collection(self, collection_list):
        '''
        delete (reset) collections in list
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for col in collection_list:
                dbw = DBWriter(self.config, collection_name = col, schema_file=None)
                # dbw.reset_collection() # This line throws OperationFailure, not sure how to fix it
                dbw.collection.drop()
                if col not in dbw.db.list_collection_names():
                    print("Collection {} is successfully deleted".format(col))
                    
        
    def get_random(self, collection_name):
        '''
        Return a random document from a collection
        '''
        dbr = DBReader(self.config, collection_name=collection_name)
        import random
        doc = dbr.collection.find()[random.randrange(dbr.count())]
        return doc
    
        
    
    def plot_fragments(self, fragment_list,rec_id=None):
        '''
        Plot fragments with the reconciled trajectory (if col2 is specified)

        Parameters
        ----------
        fragment_list : list of ObjectID fragment _ids
        rec_id: ObjectID (optional)
        '''
        
        plt.figure()
        if self.col2 and rec_id:
            d = self.col2.find_one("_id", rec_id)
            plt.scatter(d["timestamp"], d["x_position"], c="r", s=0.2, label="reconciled")
        for f_id in fragment_list:
            f = self.col1.find_one("_id", f_id)
            plt.scatter(f["timestamp"], f["x_position"], c="b", s=0.5, label="raw")
        plt.legend()
            
        
    def fragment_length_dist(self):
        '''
        Get the distribution for the #fragments that matched to one trajectory
        '''
        if self.col2 is None:
            print("Collection 2 must be specified")
            return
        pipeline = [{'$project':{ 'count': { '$size':'$fragment_ids'} } }, 
                       { '$group' : {'_id':'$count', 'count':{'$sum':1} } },
                       { '$sort'  : {'count': -1 } } ]
        cur = self.col2.collection.aggregate(pipeline)
        pprint.pprint(list(cur))

    
    def evaluate(self):
        '''
        1. filtered fragments (not in reconciled fragment_ids)
        2. lengths of those fragments (should be <3) TODO: Some long documents are not matched
        3. 
        '''
        if self.col2 is None:
            print("Collection 2 must be specified")
            return
        
        # find all unmatched fragments
        res = self.col1.collection.aggregate([
               {
                  '$lookup':
                     {
                       'from': self.col2_name,
                       'localField': "_id",
                       'foreignField': "fragment_ids",
                        'pipeline': [
                            { '$project': { 'count': 1}  }, # dummy
                        ],
                       'as': "matched"
                     }
                },
                {
                 '$match': {
                     "matched": { '$eq': [] } # select all unmatched fragments
                   }
                }
            ] )
        self.res = list(res)
        print("{} out of {} fragments are not in reconciled ".format(len(self.res), self.col1.count()))
        
        # Get the length distribution of these unmatched fragments
        import math
        f_ids = [d['_id'] for d in self.res] # get all the ids
        pipeline = [{'$project':{ 'length': { '$size':'$timestamp'} } },
                    { '$match': { "_id": {'$in': f_ids } } },
                       { '$group' : {'_id':'$length', 'count':{'$sum':1}} },
                       { '$sort'  : {'count': -1 } } ]
        cur = self.col1.collection.aggregate(pipeline)
        dict = {d["_id"]: math.log(d["count"]) for d in cur}
        plt.bar(dict.keys(), dict.values(), width = 2, color='g')
        plt.xlabel("Lengths of documents")
        plt.ylabel("Count (log-scale)")
        plt.title("Unmatched fragments length distribution in log")
        
        # pprint.pprint(list(cur))
        
    def get_stats(self):
        '''
        get the mean, median and standard deviations for
        - length/width/height
        - starting_x, ending_x
        - 
        '''
        # direction-agnostic filters
        pipeline1 = [{'$group': 
                     {'_id':'null', 
                      'avg_starting_x': {'$avg':"$starting_x"},
                      'avg_ending_x': {'$avg':"$ending_x"},
                      'stdDev_starting_x': {'$stdDevPop':"$starting_x"},
                      'stdDev_ending_x': {'$stdDevPop':"$ending_x"},
                      } 
                     }]
            
        pipeline2 = [{'$group': 
                         {'_id':'null', # no grouping 
                          'avg_starting_x': {'$avg':"$starting_x"},
                          'avg_ending_x': {'$avg':"$ending_x"},
                          'avg_veh_length': {'$avg':"$length"},
                          'avg_veh_width': {'$avg':"$width"},
                          'avg_veh_height': {'$avg':"$height"},
                          'stdDev_starting_x': {'$stdDevPop':"$starting_x"},
                          'stdDev_ending_x': {'$stdDevPop':"$ending_x"},
                          } 
                     }]
            
        # pipeline1_e = [{
        #             '$match': { 'direction': { '$eq : 18 } }
        #             '$group': 
        #                  {'_id':'null', # no grouping 
        #                   'avg_starting_x': {'$avg':"$starting_x"},
        #                   'avg_ending_x': {'$avg':"$ending_x"},
        #                   'avg_veh_length': {'$avg':"$length"},
        #                   'avg_veh_width': {'$avg':"$width"},
        #                   'avg_veh_height': {'$avg':"$height"},
        #                   'stdDev_starting_x': {'$stdDevPop':"$starting_x"},
        #                   'stdDev_ending_x': {'$stdDevPop':"$ending_x"},
        #                   } 
        #              }]

        # direction-specific filters
        
        

        res1 = self.col1.collection.aggregate(pipeline1)
        pprint.pprint(list(res1))
        
        res2 = self.col2.collection.aggregate(pipeline2)
        pprint.pprint(list(res2))
        



if __name__ == '__main__':


    cwd = os.getcwd()
    cfg = "../config"
    config_path = os.path.join(cwd,cfg)
    os.environ["user_config_directory"] = config_path
    os.environ["my_config_section"] = "TEST"
    db_param = parse_cfg("my_config_section", cfg_name = "test_param.config")
    
    ue = UnsupervisedEvaluator(db_param, "batch_5_07072022", "batch_nll_modified")
    # fragment_list = [ObjectId('62c713dfc77930b8d9533454'), ObjectId('62c713fbc77930b8d9533462')]
    # rec_id = ObjectId("62c730078b650aa00a3b925f")
    # ue.plot_fragments(fragment_list=fragment_list,rec_id=None)
    
    
    
    # ue.get_collection_info()
    # ue.fragment_length_dist()
    # ue.evaluate()
    # ue.get_stats()
    
    ue.delete_collection(["tracking_v1_stitched", "tracking_v1_nll_modified"])
