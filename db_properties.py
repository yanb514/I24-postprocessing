#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:02:50 2022

@author: yanbing_wang
"""
from i24_database_api.db_reader import DBReader
from i24_configparse.parse import parse_cfg
import os


cwd = os.getcwd()
cfg = "./config"
config_path = os.path.join(cwd,cfg)
os.environ["user_config_directory"] = config_path
parameters = parse_cfg("DEBUG", cfg_name = "test_param.config")


collection_name = "ground_truth_two"
dbr = DBReader(host=parameters.default_host, port=parameters.default_port, username=parameters.readonly_user,   
               password=parameters.default_password,
               database_name=parameters.db_name, collection_name=collection_name)

# dbr.create_index(["ID", "first_timestamp", "last_timestamp", "starting_x", "ending_x"])
print("collection name: ", collection_name)
print("number of traj: ", dbr.count())

print("min ID: ", dbr.get_min("ID"))
print("max ID: ", dbr.get_max("ID"))

print("min start time: ", dbr.get_min("first_timestamp"))
print("max start time: ", dbr.get_max("first_timestamp"))

print("min end time: ", dbr.get_min("last_timestamp"))
print("max end time: ", dbr.get_max("last_timestamp"))

print("min start x: ", dbr.get_min("starting_x"))
print("max start x: ", dbr.get_max("starting_x"))

print("min ending x: ", dbr.get_min("ending_x"))
print("max ending x: ", dbr.get_max("ending_x"))

