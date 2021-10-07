# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 11:34:52 2021

@author: wangy79
Dashboard items:
    1. distribution of the frames that each track is in scene
    2. distribution of speed, space gap, time headway
    3. number of crashes vs. frame
    4. outcome after each processing step:
        # ID, distributions
        - naive filter
        - NN merging
        - object stitching
    5. make use of vehicle class information
    6. DA metrics:
        - # misclassification (manual count)
            - false positive tracks (double counts)
            - false negative tracks (should be a new track)
        - # unique ID vs. real ID counts (manual)
        - crashes before and after DA
"""
import utils
import utils_optimization as opt
import data_association as da
import pathlib
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # read & rectify each camera df individually
    data_path = pathlib.Path().absolute().joinpath('../3D tracking')
    tform_path = pathlib.Path().absolute().joinpath('../tform')
    
    df = utils.read_data(data_path.joinpath('p1c2_small_stitch.csv'))
    
    # %% lane distribution
    # count number of unique objects in each lane
    df = utils.assign_lane(df)
    x = df.groupby('lane').ID.nunique()
    plt.bar(x.index,x.values)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Lane index')
    plt.ylabel('ID count')
    plt.title('Lane distribution')
     
    # %% Make time-space diagram
    df = utils.assign_lane(df)
    lane_idx = 1
    lane = df[df['lane']==lane_idx]
    groups = lane.groupby('ID')
    for carid, group in groups:
        x = group['Frame #'].values
        y1 = group['bbr_x'].values
        y2 = group['fbr_x'].values
        plt.fill_between(x,y1,y2,label=carid)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Frame #')
    plt.ylabel('x (m)')
    plt.title('Time-space diagram for lane {}'.format(lane_idx))
    print('{} unique IDs in lane {}'.format(len(lane['ID'].unique()),lane_idx))

    # %% count overlaps
    lane_idx = 3
    lane = df[df['lane']<=lane_idx]
    overlaps = da.count_overlaps(lane)
    