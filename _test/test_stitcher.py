#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:26:36 2022

@author: yanbing_wang
"""
import os
import sys
import queue
from i24_database_api.db_reader import DBReader
from i24_database_api.db_writer import DBWriter
from i24_configparse.parse import parse_cfg
sys.path.append('../')
from stitcher import stitch_raw_trajectory_fragments
import min_cost_flow as mcf
from collections import defaultdict
import bson
import time

import unittest
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

class T(unittest.TestCase):
    

    @classmethod
    def setUpClass(cls):
        """ get_some_resource() is slow, to avoid calling it for each test use setUpClass()
            and store the result as class variable
        """
        super(T, cls).setUpClass()
        
        # get parameters
        cwd = os.getcwd()
        cfg = "../config"
        config_path = os.path.join(cwd,cfg)
        os.environ["user_config_directory"] = config_path
        parameters = parse_cfg("TEST", cfg_name = "test_param.config")
        
        # read to queue
        gt_ids = [i for i in range(100,150)]
        fragment_queue,actual_gt_ids = mcf.read_to_queue(gt_ids=gt_ids, gt_val=25, lt_val=45, parameters=parameters)
        s1 = fragment_queue.qsize()
        print("{} fragments, actual_gt_ids: {} ".format(s1, len(actual_gt_ids)))

            
        raw = DBReader(host=parameters.default_host, port=parameters.default_port, 
                                   username=parameters.readonly_user, password=parameters.default_password,
                                   database_name=parameters.db_name, collection_name=parameters.raw_collection)
        gt = DBReader(host=parameters.default_host, port=parameters.default_port, 
                                   username=parameters.readonly_user, password=parameters.default_password,
                                   database_name=parameters.db_name, collection_name=parameters.gt_collection)

        stitched = DBWriter(host=parameters.default_host, port=parameters.default_port, 
                username=parameters.default_username, password=parameters.default_password,
                database_name=parameters.db_name, collection_name=parameters.stitched_collection,
                server_id=1, process_name=1, process_id=1, session_config_id=1, schema_file=None)
        stitched.collection.drop()
        
        stitched_reader = DBReader(host=parameters.default_host, port=parameters.default_port, 
                                   username=parameters.readonly_user, password=parameters.default_password,
                                   database_name=parameters.db_name, collection_name=parameters.stitched_collection)
        print("connected to stitched collection")
        


        # save variables to self
        cls.parameters = parameters
        cls.raw = raw
        cls.gt = gt
        cls.stitched = stitched
        cls.stitched_reader = stitched_reader
        cls.fragment_queue = fragment_queue
        cls.gt_ids = actual_gt_ids
        
        cls.run_stitcher(cls) # start stitching!
        
        
    def run_stitcher(self):
        parameters = self.parameters
        
        # make a queue
        stitched_trajectories_queue = queue.Queue()
        s1 = self.fragment_queue.qsize()
        # run stitcher, write to stitched collection
        t1 = time.time()
        # mcf.min_cost_flow_online(self.fragment_queue, stitched_trajectories_queue, parameters)
        mcf.min_cost_flow_batch(self.fragment_queue, stitched_trajectories_queue, parameters)
        # stitch_raw_trajectory_fragments("west", self.fragment_queue, stitched_trajectories_queue, parameters)
        t2 = time.time()
        print("run time: {:.2f}".format(t2-t1))
        print("{} fragments stitched to {} trajectories".format(s1, stitched_trajectories_queue.qsize()))
        
    
    # @unittest.skip("demonstrating skipping")
    def test_stitched_size(self):
        '''
        Make sure the total number of stitched trajectories equals the ground truth trajectories
        Note that passing this test does not mean the stitcher works correctly, but failing this test means that the stitcher is definitely incorrect.
        '''
        time.sleep(2)
        self.assertEqual(self.stitched.collection.count_documents({}), len(self.gt_ids), "Total number of stitched documents incorrect.")
 
    
    # @unittest.skip("demonstrating skipping")
    def test_fragments(self):
        '''
        Count the number of fragments (under-stitch) from the output of the stitcher
        '''        
        gt_id_st_fgm_ids = defaultdict(set) # key: (int) gt_id, val: (set) corresponding stitcher fragment_ids
        not_in_stitched = set()
        
        for gt_id in self.gt_ids:
            gt_doc = self.gt.find_one("ID", gt_id)
            fragment_obj_ids = gt_doc["fragment_ids"] # the ground truth fragment_ids associated with a gt ID
            for _id in fragment_obj_ids:
                if _id in self.fragment_set:
                    # find out what the stitcher thinks that fragment_id corresponds to
                    try:
                        corr_stitched_id = self.stitched_reader.collection.find_one({"fragment_ids": str(_id)})["_id"]
                        gt_id_st_fgm_ids[gt_id].add(corr_stitched_id) # count how many times corr_stitched_id is different from gt_id
                    except:
                        # find_one result is empty, a particular fragment is not stitched
                        not_in_stitched.add(_id)
                        pass 
                    
        
        # Compute fragments
        # if fragments associated to the same gt_id appear in multiple stitched_id, count as fragments
        FRAG = 0
        for gt_id in gt_id_st_fgm_ids:
            FRAG += len(gt_id_st_fgm_ids[gt_id])-1
        
        self.assertEqual(FRAG, 0, "Stitcher produces {} fragments!".format(FRAG))
        self.assertEqual(len(not_in_stitched), 0, "Fragments cannot be found in stitched collection!")
        return
        
    # @unittest.skip("demonstrating skipping")
    def test_ids(self):
        '''
        Count the number of times of overstitching (ID-switches) of the stitcher
        '''
        st_id_gt_fgm_ids = defaultdict(set) # key: (int) stitched_traj_id, val: (set) corresponding gt_ids
                
        for stitched_doc in self.stitched.collection.find({}):
            fragment_ids = stitched_doc["fragment_ids"]
            stitched_id = stitched_doc["_id"]
            for fragment_id in fragment_ids: # fragment_id is str object id
                # corr_gt_id = fragment_id // self.base  
                corr_gt_id = self.gt.collection.find_one({"fragment_ids": bson.objectid.ObjectId(fragment_id)})["_id"]
                st_id_gt_fgm_ids[stitched_id].add(corr_gt_id)
        
        # Compute ID switches
        IDS = 0
        for pred_id in st_id_gt_fgm_ids:
            IDS += len(st_id_gt_fgm_ids[pred_id])-1

        self.assertEqual(IDS, 0, "Stitcher produces {} ID switches!".format(IDS))




if __name__ == '__main__':
    unittest.main()


# %% test items
# 1. total number of stitched trajectories vs. ground truth

