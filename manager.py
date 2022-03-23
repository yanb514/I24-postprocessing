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
from mongodb_reader import DataReader
from utils.data_association import methodxx
from utils.rectification import elastic_net

# specify parameters
MODE = 'test'

# database parameters
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


