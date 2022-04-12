#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:54:26 2022

@author: wangy79
read/write data from/to Mongodb

Resources: 
    [pymongo cheatsheet] https://sysadmins.co.za/mongodb-cheatsheet-with-pymongo/
"""

from pymongo import MongoClient
import pymongo
import urllib.parse
import parameters
from bson.objectid import ObjectId
import matplotlib.pyplot as plt

class DataReader:
    '''
    A DataReader object for query
    '''
    def __init__(self, collection_name, login_info=parameters.login_info):
        client = self.get_client(login_info)
        self.col = client.trajectories[collection_name]
        
    def get_client(self, login_info):
        '''
        :param login_info: a dictionary with username and password
        :return database
        '''
        # connect to MongoDB with MongoDB URL
        username = urllib.parse.quote_plus(login_info['username'])
        password = urllib.parse.quote_plus(login_info['password'])
        client = MongoClient('mongodb://%s:%s@10.2.218.56' % (username, password))
        return client

    def get_first(self, index_name):
        '''
        get the first document from MongoDB by index_name
        '''
        return self.col.find_one(sort=[(index_name, pymongo.ASCENDING)])
        
    def get_last(self, index_name):
        '''
        get the last document from MongoDB by index_name        
        '''
        return self.col.find_one(sort=[(index_name, pymongo.DESCENDING)])
    
    def find_one(self, index_name, index_value):
        return self.col.find_one({index_name: index_value})
        
    def is_empty(self):
        return self.count() == 0
        
    def get_keys(self): 
        oneKey = self.col.find().limit(1)
        for key in oneKey:
            return key.keys()
        
    def create_index(self, indices):
        existing_indices = self.col.index_information().keys()
        for index in indices:
            if index+"_1" not in existing_indices and index+"_-1" not in existing_indices:
                self.col.create_index(index)     
        return
    
    def get_range(self, index_name, start, end): 
#        return self.col.find({
#            index_name : { "$in" : [start, end]}}).sort(index_name, pymongo.ASCENDING)
        return self.col.find({
            index_name : { "$gte" : start, "$lt" : end}}).sort(index_name, pymongo.ASCENDING)
    
    def count(self):
        return self.col.count_documents({})
    
    def get_min(self, index_name):
        return self.get_first(index_name)[index_name]
    
    def get_max(self, index_name):
        return self.get_last(index_name)[index_name]
    
    def exists(self, index_name, value):
        return self.col.count_documents({index_name: value }, limit = 1) != 0
   
        

    
class DataWriter:
    '''
    A DataReader object for query
    '''
    def __init__(self, collection_name, login_info=parameters.login_info):
        client = self.get_client(login_info)
        self.col = client.trajectories[collection_name]
        
    def get_client(self, login_info):
        '''
        :param login_info: a dictionary with username and password
        :return database
        '''
        # connect to MongoDB with MongoDB URL
        username = urllib.parse.quote_plus(login_info['username'])
        password = urllib.parse.quote_plus(login_info['password'])
        client = MongoClient('mongodb://%s:%s@10.2.218.56' % (username, password))
        return client

    def insert(self, doc):
        # TODO: schema enforcement?
        self.col.insert_one(doc) #inserts traj, which is a dictionary, into col
        return
                
    def insert_batch(self, queue):
        '''
        batch insert using non-blocking implementation (Motor package)
        queue: multiprocessing queue
        TODO: finish
        '''
#        while True:
#            doc = queue.get()
        return

    
    
        
if __name__ == "__main__":
    # connect to MongoDB with MongoDB URL
    
    raw = DataReader("raw_trajectories_one")
    gt = DataReader("ground_truth_one")
    raw.create_index(["first_timestamp","last_timestamp","starting_x","ending_x", "ID"])
    gt.create_index(["first_timestamp","last_timestamp","starting_x","ending_x", "ID"])
    
    # get stats
    print("# trajectories (raw): {}".format(raw.count()))
    print("Time range (raw): {:.2f}-{:.2f}".format(raw.get_min("first_timestamp"), raw.get_max("last_timestamp")))
    print("ID range (raw): {}-{}".format(raw.get_min("ID"), raw.get_max("ID")))
    print("Start x range (raw): {:.2f}-{:.2f}".format(raw.get_min("starting_x"), raw.get_max("starting_x")))
    print("End x range (raw): {:.2f}-{:.2f}".format(raw.get_min("ending_x"), raw.get_max("ending_x")))
    
    print("# trajectories (gt): {}".format(gt.count()))
    print("Time range (gt): {:.2f}-{:.2f}".format(gt.get_min("first_timestamp"), gt.get_max("last_timestamp")))
    print("ID range (gt): {}-{}".format(gt.get_min("ID"), gt.get_max("ID")))
    print("Start x range (gt): {:.2f}-{:.2f}".format(gt.get_min("starting_x"), gt.get_max("starting_x")))
    print("End x range (gt): {:.2f}-{:.2f}".format(gt.get_min("ending_x"), gt.get_max("ending_x")))
    
    
    # check for fragment id
#    gt_doc = gt.find_one("ID",102)
#    for raw_id in gt_doc["fragment_ids"]:
#        raw_doc = raw.find_one("_id", raw_id)
#        print(raw_doc["ID"])
    
#    car = dr.find_one("_id", ObjectId("6243e7cf734f2333efbab466"))
#    cur = dr.col.find({}).limit(5)
    
#    for doc in cur:
#        print(doc["ID"])
#        plt.figure()
#        plt.scatter(doc["timestamp"], doc["x_position"])
 
    
    
    



