#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 19:43:04 2023

@author: yanbing_wang
- examine saved pickle file
"""
import _pickle as pickle


if __name__ == "__main__":
    
    file_name = "data/trajectories_ICCV_2023_scene1_b.pkl"
    with open(file_name,"rb") as f:
        res = pickle.load(f)