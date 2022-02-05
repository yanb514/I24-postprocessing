# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 13:03:59 2021
Clean up ground truth data (status: https://docs.google.com/spreadsheets/d/1BQwHaPUAT2V6C-czkjbrOMlsg5bcyRzcz5L_1_Blavk/edit#gid=0)

GT
1. img to space conversion
1.5 add x,y etc.
2. DA
3. plot time space diagram for each lane
4. save updated files

@author: wangy79
"""

import utils
import utils_optimization as opt
import data_association as da
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import utils_vis as vis
import numpy.linalg as LA
import pandas as pd

if __name__ == "__main__":
    # read & rectify each camera df individually
    data_path = r"E:\I24-postprocess\0616-dataset-alpha\FOR ANNOTATORS"
    tform_path = r"C:\Users\wangy79\Documents\I24_trajectory\manual-track-labeler-main\DATA\tform"
    
    # %% read data into one single df
    camera_list = ["p1c2","p1c3","p1c4","p1c5"]
    sequence = 0
    df_list = []
    df_prev = None
    for camera in camera_list:
        file_path = data_path+r"\rectified_{}_{}_track_outputs_3D.csv".format(camera, sequence)
        df_single = utils.read_data(file_path)
        df_single = df_single[df_single["Frame #"]<1000]
        if isinstance(df_prev, pd.core.frame.DataFrame):
            df_single = da.assign_unique_id(df_prev, df_single)
        df_single = utils.img_to_road(df_single, tform_path, camera)
        df_single["x"] = (df_single["bbr_x"]+df_single["bbl_x"])/2
        df_single["y"] = (df_single["bbr_y"]+df_single["bbl_y"])/2
        df_prev = df_single
        df_list.append(df_single)
    df = pd.concat(df_list, axis=0, ignore_index=True)
    
    # %% run data association
    print('Before DA: ', len(df['ID'].unique()), 'cars')
    df = da.stitch_objects(df)
    print('After stitching: ', len(df['ID'].unique()), 'cars')
    df.to_csv(data_path+r"\p1_all_gt.csv", index = False)
    
    # %% visualize
    vis.plot_lane_distribution(df)
    for lane in [1,2,3,4,7,8,9,10]:
        vis.plot_time_space(df, lanes=[lane])