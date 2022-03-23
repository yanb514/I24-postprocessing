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



class DataReader():
    
    def __init__(self, login_info, mode, collection_name=None):
        '''
        mode: 'data_association' or 'vis'
        collection_name: 'raw_trajectories' if in 'data association' mode, or all trajectories if in 'vis' mode
        '''
        # connect to MongoDB with MongoDB URL
        username = urllib.parse.quote_plus(login_info['username'])
        password = urllib.parse.quote_plus(login_info['password'])
        client = MongoClient('mongodb://%s:%s@10.2.218.56' % (username, password))
        
        # get a database
        self.db = client.trajectories
        self.gt_mode = False
        
        if mode == 'data_association':
            self.collection = getattr(self.db, collection_name)
            if hasattr(self.db, 'ground_truth_trajectories'):
                self.gt_collection = self.db.ground_truth_trajectories
                self.gt_mode = True
                
        elif mode == 'vis':
            # run visualization tool
            return
        
    def _get_first(self, index_name):
        '''
        get the first document from MongoDB by index_name
        TODO: make this code easier using find_one()
        '''
        cur = self.collection.find({}).sort(index_name, pymongo.ASCENDING).limit(1)
        for car in cur:
            return car
        
    def _get_last(self, index_name):
        '''
        get the last document from MongoDB by index_name
        TODO: make this code easier using find_one()
        '''
        cur = self.collection.find({}).sort(index_name, pymongo.DESCENDING).limit(1)
        for car in cur:
            return car
    
    def _get(self, index_name, index_value):
        return self.collection.find_one({index_name: index_value})
        
        
        
        
        
        
        
    
if __name__ == "__main__":
    # connect to MongoDB with MongoDB URL
    login_info = {'username': 'i24-data',
                  'password': 'mongodb@i24'}
    collection_name = 'raw_trajectories'
    mode = 'data_association'
    dr = DataReader(login_info, mode, collection_name)
    
    car_42 = dr._get("ID", 42)
    
    last_car = dr._get_last('last_timestamp')
    print(last_car['_id'])
    
    first_car = dr._get_first('first_timestamp')
    print(first_car['ID'])
    