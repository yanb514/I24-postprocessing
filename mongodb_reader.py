#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:54:26 2022

@author: wangy79
read data from Mongodb
"""

from pymongo import MongoClient
import pymongo
import urllib.parse
import time



class DataReader():
    
    def __init__(self, LOGIN, MODE, vis=False):
        '''
        mode: 'test', 'dev' or 'deploy'
        collection_name: 'raw_trajectories' if in 'data association' mode, or all trajectories if in 'vis' mode
        '''
        # connect to MongoDB with MongoDB URL
        username = urllib.parse.quote_plus(LOGIN['username'])
        password = urllib.parse.quote_plus(LOGIN['password'])
        client = MongoClient('mongodb://%s:%s@10.2.218.56' % (username, password))
        
        # get a database
        self.db = client.trajectories
        # list collection names:
        
        self.raw = getattr(self.db, "raw_trajectories")
        self.gt = getattr(self.db, "ground_truth_trajectories")
        
        # Create index
        indices = ['first_timestamp', 'last_timestamp', 'starting_x', 'ending_x']  
        self._create_index('raw_trajectories', indices)
        self._create_index('ground_truth_trajectories', indices)
        
        
    def _create_index(self, collection_name, all_keys):
        '''
        add all_keys to indices to collection_name, if not already existed
        '''
        collection = getattr(self.db, collection_name)
        existing_keys = collection.index_information().keys()
        
        for key in all_keys:
            if key not in existing_keys:
                collection.create_index([(key, pymongo.ASCENDING)])
        return
        
    def _get_first(self, collection_name, index_name):
        '''
        get the first document from MongoDB by index_name
        TODO: make this code easier using find_one()
        '''
        collection = getattr(self, collection_name)
        cur = collection.find({}).sort(index_name, pymongo.ASCENDING).limit(1)
        for car in cur:
            return car
        
    def _get_last(self, collection_name, index_name):
        '''
        get the last document from MongoDB by index_name
        TODO: make this code easier using find_one()
        '''
        collection = getattr(self, collection_name)
        cur = collection.find({}).sort(index_name, pymongo.DESCENDING).limit(1)
        for car in cur:
            return car
    
    def _get(self, collection_name, index_name, index_value):
        collection = getattr(self, collection_name)
        return collection.find_one({index_name: index_value})
        
    def _is_empty(self, collection_name):
        # TODO: test this
        collection = getattr(self, collection_name)
        return collection.count() == 0
        
    def _get_keys(self, collection_name): 
        collection = getattr(self, collection_name)
        oneKey = collection.find().limit(1)
        for key in oneKey:
            return key.keys()
        
    
    
if __name__ == "__main__":
    # connect to MongoDB with MongoDB URL
    login_info = {'username': 'i24-data',
                  'password': 'mongodb@i24'}
    mode = 'test'
    dr = DataReader(login_info, mode)
    
    car = dr._get("gt","ID", 1)
    print(car['ID'])
    
    for fragment_id in car['fragment_ids']:
        print(fragment_id)
        raw = dr._get("raw", "_id", fragment_id)
        print(raw['ID'])
    




