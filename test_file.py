# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 16:49:19 2021

@author: wangy79
"""
import utils
import utils_optimization as opt
import data_association as da
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import utils_vis as vis

if __name__ == "__main__":
    # read & rectify each camera df individually
    data_path = r"E:\I24-postprocess\June_5min\Automatic 3D (uncorrected)"
    tform_path = r"C:\Users\wangy79\Documents\I24_trajectory\manual-track-labeler-main\DATA\tform"
    
    # %% read data preprocess and save
    camera_name, sequence = "p1c4", "0"
    file_path = data_path+"\{}_{}_track_outputs_3D.csv".format(camera_name, sequence)
    df = utils.read_data(file_path)
    df = utils.preprocess(file_path, tform_path, skip_row = 0)
    # df = df[df['Frame #']<1800]
    print('Before DA: ', len(df['ID'].unique()), 'cars')
    df = da.stitch_objects(df)
    print('After stitching: ', len(df['ID'].unique()), 'cars')
    df.to_csv("E:\I24-postprocess\June_5min\DA\{}_{}.csv".format(camera_name, sequence), index = False)
    
    # %% rectify
    df = utils.read_data("E:\I24-postprocess\June_5min\DA\{}_{}.csv".format(camera_name, sequence))
    df = opt.rectify(df)
    df.to_csv(r"E:\I24-postprocess\June_5min\rectified\{}_{}.csv".format(camera_name, sequence))
    
    # %% assign unique IDs to objects in each camera after DA on each camera independently
    df2 = utils.read_data(data_path.joinpath('p1c2_small_stitch.csv'))
    df3 = utils.read_data(data_path.joinpath('p1c3_small_stitch.csv'))
    df3 = da.assign_unique_id(df2,df3)
    
    # %% 
    # try removing outliers first
    pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
    Y1 = np.array(car[pts])	
    y_avg = (Y1[:,3]+Y1[:,5])/2
    l = np.abs(Y1[:,0]-Y1[:,2])
    w = np.abs(Y1[:,1]-Y1[:,7])
    invalid = [False]*len(Y1)
    for meas in [y_avg]:
        valid = meas[~np.isnan(meas)]
        if len(valid) > 5:
            q1, q3 = np.percentile(valid,[10,90])
            invalid = invalid | (meas<q1) | (meas>q3)
    Y1[invalid,:] = np.nan
	
    vis.plot_track(Y1)
  