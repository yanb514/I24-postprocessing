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

if __name__ == "__main__":
    # read & rectify each camera df individually
    data_path = r"E:\I24-postprocess\0616-dataset-alpha\3D tracking"
    tform_path = r"C:\Users\wangy79\Documents\I24_trajectory\manual-track-labeler-main\DATA\tform"
    camera_name, sequence = "p1c2", "0"
    
    # %% read data preprocess
    
    file_path = data_path+"\{}_{}_3D_track_outputs.csv".format(camera_name, sequence)
    # file_path = r"E:\I24-postprocess\0616-dataset-alpha\FOR ANNOTATORS\rectified_p1c2_0_track_outputs_3D.csv"
    df = utils.preprocess(file_path, tform_path, skip_row = 0)
    # df = utils.img_to_road(df, tform_path, camera_name)
    # df["camera"] = camera_name
    # df["x"] = np.array((df["bbr_x"].values+df["bbl_x"].values)/2)
    # df["y"] = np.array((df["bbr_y"].values+df["bbl_y"].values)/2)
    df.to_csv(data_path+r"\{}_{}.csv".format(camera_name, sequence), index = False)
    # df = df[df['Frame #']<1800]

    
    # %% data association
    # df = utils.read_data(data_path+"\{}_{}.csv".format(camera_name, sequence))
    df = utils.read_data(data_path+r"\{}_{}.csv".format(camera_name, sequence))
    
#%%
    df = df[(df["Frame #"]<=1200)]
    print('Before DA: ', len(df['ID'].unique()), 'cars', len(df))
    df = da.stitch_objects(df,2.5, 0.2)
    print('After stitching: ', len(df['ID'].unique()), 'cars', len(df))
    df.to_csv(data_path+r"\DA\{}_{}.csv".format(camera_name, sequence), index = False)
    # df.to_csv(data_path+r"\{}_{}_gtda.csv".format(camera_name, sequence), index = False)
    
    #%%
    def plot_track(x, y, x_id, y_id, camera, frame_id = 0, length=15,width=4):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(length,width))
    
        for i in range(len(x)):
            coord = x[i,:]
            coord = np.reshape(coord,(-1,2)).tolist()
            coord.append(coord[0]) #repeat the first point to create a 'closed loop'
            xs, ys = zip(*coord) #lon, lat as x, y
            plt.plot(xs,ys, c='r', label='pred' if i==0 else '')#alpha=i/len(D)
            plt.text(xs[0], ys[0], str(x_id[i]), fontsize=8)
            plt.scatter(x[i,2],x[i,3],color='black') # 
    
        for i in range(len(y)):
            coord = y[i,:]
            coord = np.reshape(coord,(-1,2)).tolist()
            coord.append(coord[0]) #repeat the first point to create a 'closed loop'
            xs, ys = zip(*coord) #lon, lat as x, y
            plt.plot(xs,ys, c='b', label='meas' if i==0 else '')#alpha=i/len(D)
            plt.text(xs[0], ys[0], str(y_id[i]), fontsize=8)
            plt.scatter(y[i,2],y[i,3],color='black') # 
            
        plt.xlabel('meter')
        plt.ylabel('meter')
        xmin,xmax,ymin,ymax = utils.get_camera_range(camera)
        plt.xlim([180,260])
        plt.ylim([ymin,ymax])
        plt.title(frame_id)
        plt.legend()
        plt.show() 
        return
    
    plot_track(np.vstack(x), y, curr_id, frame["ID"].values, ["p1c3"], frame["Frame #"].iloc[0])
    #%%
    # df = newdf
    frames = df.groupby("Frame #")
    ids = set()
    for f_id, frame in frames:
        groups = frame.groupby("ID")
        for carid, group in groups:
            if len(group)>1:
                ids.add(carid)
                
    #%%# plot time space diagram (4 lanes +1 direction)
    df = utils.read_data(data_path+r"\DA\{}_{}.csv".format(camera_name, sequence))
    temp = df[df['ID'].isin([360.0, 387.0])]
    
    plt.figure()
    
    colors = ["blue","orange","green","red"]
    groups = temp.groupby('ID')
    j = 0
    for carid, group in groups:
        x = group['Frame #'].values
        y1 = group['bbr_x'].values
        y2 = group['fbr_x'].values
        plt.fill_between(x,y1,y2,alpha=0.5,color = colors[j%4], label="id {}".format(carid))
        j += 1
    try:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    except:
        pass
    plt.xlabel('Frame #')
    plt.ylabel('x (m)')
    plt.title('Time-space diagram')  
        
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
        
    # %% rectify
    df = utils.read_data(data_path+r"\DA\{}_{}.csv".format(camera_name, sequence))
    # df = df[(df["ID"]>=2700) & (df["ID"]<2800)] # 2785 is too long
    df = opt.rectify(df)
    df.to_csv(data_path+r"\rectified\{}_{}.csv".format(camera_name, sequence), index = False)
    
    # %% post processing
    df = utils.read_data(data_path+r"\rectified\{}_{}_l1.csv".format(camera_name, sequence))
    df = utils.post_process(df)
    df.to_csv(data_path+r"\rectified\{}_{}_l1_post.csv".format(camera_name, sequence), index = False)
     
    # %% diagnose rectification on single cars
    dfda = utils.read_data(data_path+r"\DA\{}_{}.csv".format(camera_name, sequence))
    df = utils.read_data(data_path+r"\rectified\{}_{}.csv".format(camera_name, sequence))
    
    # %%individual cars
    import utils_optimization as opt
    carid = 88
    carda = dfda[dfda["ID"]==carid]
    car = carda.copy()
    car = opt.rectify_single_camera(car,  (1,0,0,0.1,0.1,0))
    # car = df[df["ID"]==carid]
    vis.plot_track_compare(carda, car)
    utils.dashboard([carda, car])
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
    
    