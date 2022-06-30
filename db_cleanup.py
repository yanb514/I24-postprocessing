#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 13:22:12 2022

@author: yanbing_wang
"""
import os
import signal
import sys

import i24_logger.log_writer as log_writer
from i24_database_api.db_writer import DBWriter
from i24_database_api.db_reader import DBReader
from i24_configparse import parse_cfg


config_path = os.path.join(os.getcwd(),"config")
os.environ["user_config_directory"] = config_path
os.environ["my_config_section"] = "TEST"
parameters = parse_cfg("my_config_section", cfg_name = "test_param.config")


# Reset collection
reconciled_schema_path = os.path.join(os.environ["user_config_directory"],parameters.reconciled_schema_path)
dbw = DBWriter(parameters, collection_name = "stitched_three", schema_file=reconciled_schema_path)
raw = DBReader(parameters, collection_name="garbage_dump_2")

print(dbw.db.list_collection_names())
dbw.reset_collection() # This line throws OperationFailure, not sure how to fix it
print("stitched_three" in dbw.db.list_collection_names())

