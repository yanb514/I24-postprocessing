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
import numpy.linalg as LA
import pandas as pd

# %%
if __name__ == "__main__":
    # MC tracking
    data_path = r"E:\I24-postprocess\MC_tracking"
    file_path = data_path+r"\reinterpolated_3D_tracking_outputs.csv"
    df= utils.read_data(file_path)
    df= utils.remove_wrong_direction_df(df)
    # read & rectify each camera df individually
    # data_path = r"E:\I24-postprocess\0616-dataset-alpha\3D tracking"
    # tform_path = r"C:\Users\wangy79\Documents\I24_trajectory\manual-track-labeler-main\DATA\tform"
    # camera_name, sequence = "p1c2", "0"
    
    #%% preprocess MC
    df = utils.preprocess_MC(file_path, "")
    # assign frame idx TODO: may have issues
    df['Frame #'] = df.groupby("Timestamp").ngroup()
    plt.scatter(df["Frame #"].values, df["Timestamp"].values, s=0.1)
    df.to_csv(data_path+r"\MC_reinterpolated.csv", index = False)
    
    # preprocess individual camera
    # file_path = data_path+"\{}_{}_3D_track_outputs.csv".format(camera_name, sequence)
    # # file_path = r"E:\I24-postprocess\0616-dataset-alpha\FOR ANNOTATORS\rectified_p1c2_0_track_outputs_3D.csv"
    # df = utils.preprocess(file_path, tform_path, skip_row = 0)
    # df.to_csv(data_path+r"\{}_{}.csv".format(camera_name, sequence), index = False)
    
    # %% data association
    df = utils.read_data(data_path+r"\MC_reinterpolated.csv")
    df= utils.remove_wrong_direction_df(df)
    df = df[(df["Frame #"]<2000)]
    print('Before DA: ', len(df['ID'].unique()), 'cars', len(df))
    df = da.stitch_objects_jpda(df,2.5, mc=True)
    df= utils.remove_wrong_direction_df(df)
    print('After stitching: ', len(df['ID'].unique()), 'cars', len(df))
    df = df.groupby("ID").apply(utils.connect_track).reset_index(drop=True)
    # df.to_csv(data_path+r"\DA\{}_{}.csv".format(camera_name, sequence), index = False)
    df.to_csv(data_path+r"\DA\MC_jpda.csv", index = False)
    
    #%%
    dfda = utils.read_data(data_path+r"\DA\MC_jpda.csv")
    # %%
    def detect_outlier_position_by_fft(signal, threshold_freq=0.1,
                                   frequency_amplitude=.001):
        signal = signal.copy()
        mask = np.isfinite(signal)
        signal = np.interp(signal.index, signal.index[mask], signal[mask])
        fft_of_signal = np.fft.fft(signal)
        outlier = np.max(signal) if abs(np.max(signal)) > abs(np.min(signal)) else np.min(signal)
        if np.any(np.abs(fft_of_signal[threshold_freq:]) > frequency_amplitude):
            index_of_outlier = np.where(signal == outlier)
            return index_of_outlier[0]
        else:
            return None
    
    #%% visualize
    temp = test[(test["Frame #"]<1500)&(test["Frame #"]>1000)]
    # temp = utils.remove_wrong_direction_df(temp)
    for lane_idx in [1,2,3,4,7,8,9,10]:
        vis.plot_time_space(temp, lanes=[lane_idx], time="frame", space="x")
        # vis.plot_time_space(test, lanes=[lane_idx], time="frame", space="y")

    # %% rectify
    # df = utils.read_data(data_path+r"\DA\{}_{}.csv".format(camera_name, sequence))
    df = utils.read_data(data_path+r"\DA\MC_jpda.csv")
    # df = df[df["Frame #"]<1000]
    # df = df[(df["ID"]>=2700) & (df["ID"]<2800)] # 2785 is too long
    df = opt.rectify(df)
    df.to_csv(data_path+r"\rectified\MC.csv", index = False)
    
    # %% test time-space matching
    df = utils.read_data(data_path+r"\MC_reinterpolated.csv")
    df = df[(df["Frame #"]<2100)&(df["Frame #"]>900)]
    df= utils.remove_wrong_direction_df(df)
    # test, groupList,DX,DY = da.time_space_matching(df, 5, 0.1)
    test = da.stitch_objects_jpda(df,3, mc=True)
    # test = da.stitch_objects_gnn(df,2.5, mc=True)
    # test = da.stitch_objects_bm(df,2.5, mc=True)
    
    # %% explore maha distance
    track1 = df[df["ID"].isin([464,483,489,496,507])]
    track1["ID"] = 464
    track1 = utils.connect_track(track1).reset_index(drop=True)
    track2= track1.copy()
    track2 = opt.rectify_single_camera(track2, (1,0,0,0.1,0.1,0))
    pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
    Y1 = np.array(track1[pts])
    Y2 = np.array(track2[pts])
    diff = Y1-Y2
    # array([3.45759796, 0.18503484, 3.54686592, 0.40411873, 3.56373424,
       # 0.2956523 , 3.45051538, 0.21322893]) stdev
    plt.figure()
    for i in range(len(diff)):
        for j in range(4):
            plt.scatter(diff[i][j],diff[i][2*j+1])
    # %%
    ed = []
    for i in range(len(diff)):
        if Y1[i][0] != np.nan:
            ed.append(dist_score(Y1[i],Y2[i],'ed'))
    maha = []
    for i in range(len(diff)):
        if Y1[i][0] != np.nan:
            maha.append(dist_score(Y1[i],Y2[i],'maha'))   
            
   
    #%%
    id1,id2=360,373
    car1=df[df["ID"]==id1]
    car2=df[df["ID"]==id2]
    d = {id1:car1}
    x,d = da.predict_tracks_df(d)
    vis.plot_track_compare(car1,car2)
    pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
    # box1 = np.array(car1[pts]) 
    box1=np.array(d[id1].tail(1)[pts])
    box2 = np.array(car2[pts])[1]
    print(da.dist_score(box1,box2,'xyw'))
        # %%
    temp = df[df["Frame #"]<300]
    lens = []
    ids = []
    groups = temp.groupby("ID")
    for carid, group in groups:
        ids.append(carid)
        lens.append(group["Frame #"].iloc[-1]-group["Frame #"].iloc[0])
    vis.plot_lane_distribution(temp)
    for lane in [1,2,3,4,7,8,9,10]:   
        vis.plot_time_space(temp, lanes=[lane])
        
   
    
    # %% post processing
    df = utils.read_data(data_path+r"\rectified\{}_{}_l1.csv".format(camera_name, sequence))
    df = utils.post_process(df)
    df.to_csv(data_path+r"\rectified\{}_{}_l1_post.csv".format(camera_name, sequence), index = False)
     
    # %% diagnose rectification on single cars
    dfda = utils.read_data(data_path+r"\DA\{}_{}.csv".format(camera_name, sequence))
    df = utils.read_data(data_path+r"\rectified\{}_{}.csv".format(camera_name, sequence))
    
    # %%individual cars
    car1 = da.df[da.df["ID"]==594]
    # car2 = da.df[da.df["ID"]==463]
    # vis.plot_track_compare(car1,car2)
    vis.plot_track_df(car1)
    #%%
    import utils_optimization as opt
    carid = 2
    carda = da.df[da.df["ID"]==carid]
    car = carda.copy()
    car = opt.rectify_single_camera(car,  (1,0,0,0.2,0,0))
    # car = df[df["ID"]==carid]
    vis.plot_track_compare(carda, car)
    vis.dashboard([carda, car],["raw","rectified"])
    # plt.title(carid)
    Y1 = np.array(carda[['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']])
    Yre = np.array(car[['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']])
    diff = Y1-Yre
    score = np.nanmean(LA.norm(diff,axis=1))
    print(score)
    #%% slow car
    slow = df[df["speed"]<20]
    
    # %% assign unique IDs to objects in each camera after DA on each camera independently
    df2 = utils.read_data(data_path.joinpath('p1c2_small_stitch.csv'))
    df3 = utils.read_data(data_path.joinpath('p1c3_small_stitch.csv'))
    df3 = da.assign_unique_id(df2,df3)
    
    # %% visualization
    gt_path = r"E:\I24-postprocess\0616-dataset-alpha\FOR ANNOTATORS\p1c24_gt.csv"
    DA_path = r"E:\I24-postprocess\0616-dataset-alpha\3D tracking\DA\p1c24.csv"
    # gt = utils.read_data(gt_path)
    da = utils.read_data(DA_path)
    # gt = gt[gt["Frame #"]<200]
    da = da[da["Frame #"]<200]
    # vis.plot_lane_distribution(gt)
    vis.plot_lane_distribution(da)
    # vis.plot_time_space(gt)
    vis.plot_time_space(da)
    
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
    speed[0] = 0
    gps["speed"] = speed
    
    # %% plot speed
    plt.figure()
    plt.scatter(gps.Gpstime.values, gps.speed.values,s=1)
    plt.xlabel("Time")
    plt.ylabel("Speed (m/s)")
    plt.title("Speed")
    plt.ylim([0,50])
    # calculate relataive coords given origin 0,0 -> 36.005437, -86.610796
    
    