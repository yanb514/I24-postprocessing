#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:38:44 2022

@author: yanbing_wang
"""
from i24_database_api import DBClient
import os
import json
import sys


def main(database_name="", collection_name=""):
    with open(os.environ["USER_CONFIG_DIRECTORY"]+"/db_param.json") as f:
        db_param = json.load(f)
        
    # TRANSFORM
    dbc = DBClient(**db_param, database_name = database_name)
    dbc.client["transformed_beta"][collection_name].drop()
    dbc.transform2(read_collection_name = collection_name, chunk_size=50)
    del dbc


if __name__ == '__main__':
    
    main(sys.argv[1], sys.argv[2])
    
    