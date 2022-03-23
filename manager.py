#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:56:43 2022

@author: wangy79
MODE: 
    'test': test using synthetic data with ground truth
    'dev': development mode on real data, no GT
    'deploy': deploy on real testbed with streaming data
"""
import queue
from utils.mongodb_reader import DataReader
from utils.mongodb_handler import DataHandler
from utils.log_handler import LogHandler
from utils.data_association import spatial_temporal_match_online
from utils.rectification import elastic_net

## specify parameters
MODE = 'test'
TIME_OUT = 50 # gracefully shutdown if db has not been updated in TIME_OUT seconds

# database parameters
login_info = {
        'username': 'i24-data',
        'password': 'mongodb@i24'
        }
db_name = 'raw_trajectories'
db_params = {
        'LOGIN': login_info,
        'MODE': MODE,
        'DB': db_name
        }

# data association parameters
da_params = {
        'TIME_WIN': XX,
        'THRESH': XX,
        'VARX': XX,
        'VARY': xx
        }

# rectification parameters
re_params = {
        'LAM1_X':XX,
        'LAM1_Y':XX,
        'LAM2_X':XX,
        'LAM2_Y':XX,
        'PH': xx,
        'IH': xx
        }


## create an intermediary data queue 
data_q = queue.Queue() # TODO: make the q global?
log_q = queue.Queue()
dr = DataReader(**db_params) 
while True: # TODO: while not TIME_OUT
    data_q.put(dr._get_first('last_timestamp')) # get the trajectories ordered in last_timestamp, read one traj each loop
    data_q, log_q = spatial_temporal_match_online(data_q, log_q, **da_params) # run data association and update data_q, log_q
    data_q, log_q = elastic_net(data_q, log_q, **re_params) # run trajectory rectification in parallel


