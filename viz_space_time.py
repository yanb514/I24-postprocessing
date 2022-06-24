#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:22:38 2022

@author: yanbing_wang
"""

from i24_database_api.db_reader import DBReader
import matplotlib.pyplot as plt
import numpy as np
import cmd
import json
import os

class SpaceTimePlot():
    """
    Create a time-space diagram for a specified time-space window
    Query from database
    Plot on matplotlib
    """
    
    def __init__(self, config):
        """
        Initializes an Overhead Traffic VIsualizer object
        
        Parameters
        ----------
        config : object
        """
        self.dbr = DBReader(host=config["host"], 
                       port=config["port"], 
                       username=config["username"], 
                       password=config["password"], 
                       database_name=config["database_name"], 
                       collection_name=config["collection_name"])

        # build multikey index
        dbr.collection.createIndex( { "timestamp":  1 } )
        
    def plot_time_space(self, tmin, tmax, xmin, xmax):
        
        docs = self.dbr.find({})
        
    
    
if __name__=="__main__":
    
    config_path = 
    stp = SpaceTimePlot(config)
    