#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 16:13:43 2022

@author: yanbing_wang
"""

# stitcher algorithm parameters
TIME_WIN = 50
THRESH = 3
VARX = 0.05 # TODO: unit conversion (now it's based on meter)
VARY = 0.03

IDLE_TIME = 5 # if tail_time of a path has not changed after IDLE_TIME, then write to database

# For writing raw trajectories as Fragment objects
# change first "ID" to "_id" to query by ObjectId
WANTED_DOC_FIELDS = ["ID", "ID","timestamp","x_position","y_position","direction","last_timestamp","last_timestamp", "first_timestamp"]
FRAGMENT_ATTRIBUTES = ["id","ID","t","x","y","dir","tail_time","last_timestamp","first_timestamp"]

# data feed parameter
RANGE_INCREMENT = 50 # seconds interval to batch queue and refill raw_trajectories_queue
