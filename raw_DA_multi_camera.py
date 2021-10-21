# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 13:03:59 2021
Clean up multi-camera data (status: https://docs.google.com/spreadsheets/d/1BQwHaPUAT2V6C-czkjbrOMlsg5bcyRzcz5L_1_Blavk/edit#gid=0)

raw
1. preprocess
2. DA
3. plot time space diagram for each lane

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
    data_path = r"E:\I24-postprocess\0616-dataset-alpha\3D tracking"
    tform_path = r"C:\Users\wangy79\Documents\I24_trajectory\manual-track-labeler-main\DATA\tform"
    
    # %% read data into one single df
    camera_list = ["p1c2","p1c3","p1c4","p1c5"]
    sequence = 0
    df = pd.DataFrame()
    for camera in camera_list:
        file_path = data_path+r"\{}_{}_3D_track_outputs.csv".format(camera, sequence)
        df_single = utils.preprocess(file_path, tform_path, skip_row = 0)
        df_single = df_single[df_single["Frame #"]<1000]
        if len(df)>0:
            df_single = da.assign_unique_id(df, df_single)
        df_single = utils.img_to_road(df_single, tform_path, camera)
        df_single["x"] = (df_single["bbr_x"]+df_single["bbl_x"])/2
        df_single["y"] = (df_single["bbr_y"]+df_single["bbl_y"])/2
        df = pd.concat([df,df_single], axis=0, ignore_index=True)
    df.to_csv(data_path+r"\p1_all_preprocess.csv", index = False)
    
    # %% run data association
    path = data_path+r"\p1_all_preprocess.csv"
    df = utils.read_data(path)
    print('Before DA: ', len(df['ID'].unique()), 'cars')
    df = da.stitch_objects(df)
    print('After stitching: ', len(df['ID'].unique()), 'cars')
    # df.to_csv(data_path+r"\DA\p1_all_DA_iou.csv", index = False)
    
    # %% visualize
    temp = df[df["Frame #"]>4800]
    vis.plot_lane_distribution(temp)
    lane_list = [1,2,3,4,7,8,9,10]
    for lane in lane_list:
        vis.plot_time_space(temp, lanes=[lane])