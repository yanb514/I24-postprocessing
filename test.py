#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:38:44 2022

@author: yanbing_wang
"""

from pymongo import MongoClient
import urllib


if __name__ == '__main__':
    
    # url = 'http://viz-dev.isis.vanderbilt.edu:5991/upload?type=jane_viz'
    
    # files = {'upload_file': open("/Users/yanbing_wang/Documents/Research/I24/figures/conf.png",'rb')}
    # ret = requests.post(url, files=files, allow_redirects=True)
    # print(ret)
    # if ret.status_code == 200:
    #     print('Uploaded!')
    
    
    username = urllib.parse.quote_plus('readonly')
    password = urllib.parse.quote_plus('mongodb@i24')
    client = MongoClient('mongodb://%s:%s@10.80.4.91' % (username, password))
    db = client["trajectories"] # put database name here
    col = db["ICCV_2023_scene2_TRACKLETS"] # put collection name here
    
    # print the id of the first 5 documents
    # col.find({}) returns a running cursor of the collection
    cnt = 0
    for doc in col.find({}):
        # do something with the document
        print(doc["_id"])
        cnt += 1
        if cnt > 5:
            break
        