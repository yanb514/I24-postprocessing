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
    data_path = r"E:\I24-postprocess\3D tracking"
    tform_path = r"C:\Users\wangy79\Documents\I24_trajectory\manual-track-labeler-main\DATA\tform"
    
    # %% read data preprocess and save
    camera_name, sequence = "p1c4", "0"
    # file_path = data_path+"\{}_{}_track_outputs_3D.csv".format(camera_name, sequence)
    file_path = data_path+r"\record_51_{}_00000_3D_track_outputs.csv".format(camera_name)
    # df = utils.read_data(file_path)
    df = utils.preprocess(file_path, tform_path, skip_row = 0)
    # df = df[df['Frame #']<1800]
    print('Before DA: ', len(df['ID'].unique()), 'cars')
    df = da.stitch_objects(df)
    print('After stitching: ', len(df['ID'].unique()), 'cars')
    df.to_csv(data_path+"\DA\{}.csv".format(camera_name), index = False)
    
    # %% rectify
    df = utils.read_data(r"E:\I24-postprocess\3D tracking\DA\{}.csv".format(camera_name))
    df = opt.rectify(df)
    df.to_csv(r"E:\I24-postprocess\3D tracking\rectified\{}.csv".format(camera_name),index=False)
    
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

    #%% get GPS speed
    gps_path = r"E:\I24-postprocess\gps_0806"
    file_path = gps_path+r"\2021-08-06-11-40-19_2T3MWRFVXLW056972_GPS_Messages.csv"
    gps = utils.read_data(file_path)
    gps = gps[gps["Gpstime"]>0]
    gps = gps.reset_index(drop=True)
    # plot
    plt.scatter(gps.Long.values, gps.Lat.values,s=1)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    # %% calculate distance traveld and speed
    dist = []
    speed = []
    for i,row in gps.iterrows():
        try:
            d,_,_ = utils.euclidean_distance(gps.iloc[i-1].Lat, gps.iloc[i-1].Long, row.Lat, row.Long)
            dt = row.Gpstime-gps.iloc[i-1].Gpstime
        except:
            d = 0
            dt = 0.1
        v = d/dt
        speed.append(v)
        dist.append(d)
    dist[0] = 0
    speed[0] = 0
    gps["dist"] = dist
    gps["speed"] = speed
    # calculate relataive coords given origin 0,0 -> 36.005437, -86.610796
    
    