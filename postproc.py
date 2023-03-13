#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 14:44:19 2022

@author: yanbing_wang
"""
import sys
import pp1_all_nodes




def __process_entry__(trackID='', postID=''):
    
    pp1_all_nodes.main(raw_collection = trackID, reconciled_collection = postID)
    
    return



if __name__ == "__main__":

    if (len(sys.argv) == 2) or (len(sys.argv) == 3):
        print('Postprocessing for I-24 Motion')
        
        # 'online' run
        if (len(sys.argv) == 2):
            __process_entry__(sys.argv[1], sys.argv[1])

        # 'offline' run
        if (len(sys.argv) == 3):
            __process_entry__(sys.argv[1], sys.argv[2]) 
            
    else: 
        print('Not enough arguments')