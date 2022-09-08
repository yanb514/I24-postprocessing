#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 09:45:53 2022

@author: yanbing_wang
"""
from i24_database_api import DBClient

if __name__ == "__main__":
    import json
    import os
    
    with open("../config/parameters.json") as f:
        parameters = json.load(f)
    
    with open(os.path.join(os.environ["USER_CONFIG_DIRECTORY"], "db_param.json")) as f:
        db_param = json.load(f)
        
    
    dbc = DBClient(**db_param, database_name = parameters["eval_database"])
    for col in dbc.list_collection_names():
        if "organic_forengi" in col:
            r = input("Delete {}? {} documents. Y/n".format(col, dbc.db[col].count_documents({})))
            if r == "Y":
                dbc.db[col].drop()