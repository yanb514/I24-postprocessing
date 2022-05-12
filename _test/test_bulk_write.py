#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 13:30:48 2022

@author: yanbing_wang
"""
import multiprocessing as mp
# from threading import Thread
# from db_writer import DBWriter
import db_parameters
import time
import sys
import random
import math
import pymongo
import urllib

def main_proc(queue):
    print("main started")
    num_tasks = 100
    # while True:
    #     try:
    #         # simulate a CPU-intensive task for 1 sec
    #         startTime = time.time()
    #         while time.time() - startTime < 1:
    #             math.factorial(100) 
    #         t0 = time.time()
    #         for i in range(num_tasks):
    #             queue.put({"key": i, 'rand': [random.random() for j in range(40)],
    #                    'message': 'a message '*100})
    #         t1 = time.time()
    #         print("Time to insert to queue: ", t1-t0)
    #     except KeyboardInterrupt:
    #         sys.exit()
 
def writer_proc( queue):
    print("writer started")
    # username=db_parameters.DEFAULT_USERNAME
    # password=db_parameters.DEFAULT_PASSWORD
    # host=db_parameters.DEFAULT_HOST
    # username = urllib.parse.quote_plus(username)
    # password = urllib.parse.quote_plus(password)
    # client = pymongo.MongoClient('mongodb://%s:%s@%s' % (username, password, host))
                      
    # collection = client.db["test_collection"]
    # collection.drop() 
    
    # t0 = time.time()
    # while True:
    #     doc = queue.get(True)
    #     collection.insert_one(doc)
        
    #     print("Final docs: {}, other proc: {:.2f}, insertion time: {:.2f}".format(collection.count_documents({}), t0,t0))


def main():
    mpq = mp.Queue()
    p1 = mp.Process(target = main_proc, args=(mpq,), daemon = False)
    p2 = mp.Process(target = writer_proc, args=(mpq,), daemon = False)
    p1.start()
    p2.start()


if __name__ == '__main__':
    # the manager level
    main()
    
        
    