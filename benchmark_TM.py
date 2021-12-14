# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 21:03:14 2021

@author: wangy79
benmark using TransModeler (TM) simulation data
"""
import utils
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin,cos
import utils_optimization as opt
import utils_vis as vis
from tqdm import tqdm
import random

pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    # deal with boundary effect
    y_smooth[:box_pts] = y_smooth[box_pts]
    y_smooth[-box_pts:] = y_smooth[-box_pts]
    return y_smooth

def plot_time_space(df, lanes=[1], time="Time", space="Distance", ax=None, show =True):
        
    # plot time space diagram (4 lanes +1 direction)
    # if ax is None:
    #     fig, ax = plt.subplots()      
    
    colors = ["blue","orange","green","red","purple"]
    for i,lane_idx in enumerate(lanes):
        fig, ax = plt.subplots(1,1, figsize=(5,5), facecolor='w', edgecolor='k')
        lane = df[df['lane']==lane_idx]
        groups = lane.groupby('ID')
        j = 0
        for carid, group in groups:
            x = group[time].values             
            y1 = group['bbr_x'].values
            y2 = group['fbr_x'].values
            ax.fill_between(x,y1,y2,alpha=0.5,color = colors[j%len(colors)], label="{}".format(carid))
            # ax.plot(x,y1,color = colors[j%len(colors)], label="{}".format(carid))
            j += 1
        # try:
        #     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # except:
        #     pass

        ax.set_xlabel(time)
        ax.set_ylabel(space)
        ax.set_title('Lane {}'.format(lane_idx)) 
    return None if show else ax

def standardize(df):
    '''
    Standardize the units to International Units
    meter, m/s, rad etc.
    '''
    # distance is in feet -> m (/3.281)
    # time: sec
    # speed mph -> m/s (/2.237)
    df["Distance"] /= 3.281
    df["Speed"] /= 2.237
    df = df.drop(columns=['Latitude', 'Longitude', 'Segment', "Offset", "Heading", "Mileage"])
    # df = df.rename(columns={"Distance":"x"})
    df["Distance"] = df["Distance"].values - min(df["Distance"].values)
    
    return df

def calc_state(df):
    '''
    1. get y-axis based on lane idx, 1=HOV
    2. get smooth y-axis
    3. calc vx, vy, theta (heading angle)
    4. estimate l,w  based on vehicle classes
    '''
    veh = {1:"sedan",
           2:"sedan",
           3:"sedan",
           4:"SUV",
           5:"truck",
           6:"trailer",
           7:"bus"}
    dim = {"sedan": [4.5, 1.7], # dimension are in [width, length]
           "SUV": [4.5, 1.9],
           "truck": [8, 2],
           "bus": [12, 2.2],
           "trailer": [12, 2.2]}
    y = (np.arange(0,12*4,12)+6)/3.281 # get y position (meter) from lane idx, assuming the lane width = 12 ft
    y = y[::-1]
    y_arr = [y[i-1] for i in df.Lane.values]
    cls_arr = df.Class.values # convert vehicle class index to actual class name
    for i,c in enumerate(cls_arr):
        try:
            cls_arr[i] = int(c)
        except:
            cls_arr[i] = 1
    veh_arr = [veh[i] for i in cls_arr]
    l_arr = [dim[i][0] for i in veh_arr]
    w_arr = [dim[i][1] for i in veh_arr]
    
    # write information to dataframe
    df["Class"] = veh_arr
    df["Width"] = w_arr
    df["Length"] = l_arr
    df["y"] = y_arr
    df["Time"] = df["Time"].values - min(df["Time"].values)
        
    return df

def resample_single(car):
    '''
    resample from 1hz to 30hz
    '''
    if len(car)<3: # ignore short trajectories
        return None
    time = car.Time.values
    newtime = np.arange(time[0], time[-1]+1/30, 1/30)# to 30hz
    d = car.Distance.values 
    dir = np.sign(d[-1]-d[0]) # travel direction
    v = np.abs(np.diff(d))    # differentiate distance to get speed
    v = np.hstack([v,v[-1]])  
    y = car.y.values
    vy = np.diff(y)           # y-component speed
    vy = np.hstack([vy, vy[-1]])
    theta = np.arccos(dir)-np.arcsin(vy/v) # heading angle
    
    thetare = np.interp(newtime, time, theta) # linear interpolate to 30hz
    vre = np.interp(newtime, time, v)
    
    new_index = pd.Index(newtime, name="Time")
    car = car.set_index("Time").reindex(new_index).reset_index()
    # interpolate / fill nan
    pts_fixed = ["ID","Class","Dir","Width","Length"] # simply copy these values during upsampling
    car["Time"] = newtime
    car["theta"] = thetare
    car["speed"] = vre
    car["direction"] = dir
    # copy the rest column
    car[pts_fixed] = car[pts_fixed].interpolate(method='pad')
         
    return car
  
def generate_meas(car):
    '''
    Generate footprint measurements (bbr_x, bbr_y, etc.) from state information
    '''
    w = car.Width.values
    l = car.Length.values
    x0 = car.Distance.values[0]
    y0 = car.y.values[0]
    theta=car.theta.values
    theta = smooth(theta, 10) # TODO: smoothing window to be testeds
    v=car.speed.values
    Yre,x,y,a = opt.generate(w,l,x0, y0, theta, v, outputall=True)
    car["x"] = x
    car["y"] = y
    car["acceleration"] = a
    car["theta"] = theta
    car.loc[:,pts] = Yre 
    return car
    
def preprocess(df):
    '''
    1. up sample from 1hz to 30hz
    2. Generate measurements from states
    3. standardize csv format
    '''
    tqdm.pandas()
    print("Up sampling data...")
    df = df.groupby('ID').apply(resample_single).reset_index(drop=True) 
    tqdm.pandas()
    print("Generating measurements...")
    df = df.groupby('ID').apply(generate_meas).reset_index(drop=True) 
    df = utils.assign_lane(df)
    
    # standardize for csv reader
    df = df.rename(columns={"Time":"Timestamp", "Class": "Object class", "Width":"width", "Length":"length"})
    df['Frame #'] = np.round(df["Timestamp"].values*30).astype(int)
    col = ['Frame #', 'Timestamp', 'ID', 'Object class', 'BBox xmin','BBox ymin','BBox xmax','BBox ymax',
            'vel_x','vel_y','Generation method',
            'fbrx','fbry','fblx','fbly','bbrx','bbry','bblx','bbly','ftrx','ftry','ftlx','ftly','btrx','btry','btlx','btly',
            'fbr_x','fbr_y','fbl_x','fbl_y','bbr_x','bbr_y','bbl_x','bbl_y',
            'direction','camera','acceleration','speed','x','y','theta','width','length','height',"lane"]
    df = df.reindex(columns=col)

    df = df.sort_values(by=['Frame #','ID']).reset_index(drop=True)         
    return df

def pollute_car(car, AVG_CHUNK_LENGTH, OUTLIER_RATIO):
    '''
    AVG_CHUNK_LENGTH: Avg length (in # frames) of missing chunk mask
    OUTLIER_RATIO: ratio of bbox in each trajectory are outliers (noisy measurements)
    Assume each trajectory is manually chopped into 0.01N fragments, where N is the length of the trajectory
        Mark the IDs of the fragments as xx000, e.g., if the GT ID is 9, the fragment IDs obtained from track 9 are 9000, 9001, 9002, etc.
        This is assign a unique ID to each fragments.
        
    '''
    car=car.reset_index(drop=True)
    l = car["length"].iloc[0]
    w = car["width"].iloc[0]
    id = car["ID"].iloc[0] # original ID
    
    # mask chunks
    n_chunks = int(len(car)*0.01)
    for index in sorted(random.sample(range(0,len(car)),n_chunks)):
        to_idx = max(index, index+AVG_CHUNK_LENGTH+np.random.normal(0,20)) # The length of missing chunks follow Gaussian distribution N(AVG_CHUNK_LENGTH, 20)
        car.loc[index:to_idx, pts] = np.nan # Mask the chunks as nan to indicate missing detections
        if id>=1000: id+=1 # assign unique IDs to fragments
        else: id*=1000
        car.loc[to_idx:, ["ID"]] = id
        
    # add outliers (noise)
    outlier_idx = random.sample(range(0,len(car)),int(OUTLIER_RATIO*len(car))) # randomly select 0.01N bbox for each trajectory to be outliers
    for idx in outlier_idx:
        noise = np.random.multivariate_normal([0,0,0,0,0,0,0,0], np.diag([0.3*l, 0.3*w]*4)) # add noises to each outlier box
        car.loc[idx, pts] += noise
    car.loc[outlier_idx, ["Generation method"]] = "outlier"
    return car

def pollute(df, AVG_CHUNK_LENGTH, OUTLIER_RATIO):
    print("Downgrading data...")
    df = df.groupby('ID').apply(pollute_car, AVG_CHUNK_LENGTH, OUTLIER_RATIO).reset_index(drop=True)
    # df = applyParallel(df.groupby("ID"), pollute_car).reset_index(drop=True)
    return df

# %%
if __name__ == "__main__":
    data_path = r"E:\I24-postprocess\benchmark\TM_trajectory.csv"
    df = pd.read_csv(data_path, nrows=5000)

    print(len(df))
    df = standardize(df)
    df = calc_state(df)
    df = preprocess(df)
    
    # you can select some time-space window such that the trajectory lengths are similar (run plot_time_space to visualize)
    df = df[df["x"]>1000]
    df = df[df["Frame #"]>1000]
    
    df.to_csv(r"E:\I24-postprocess\benchmark\TM_1000_GT.csv", index=False) # save the ground truth data
    #%%
    df = pollute(df, AVG_CHUNK_LENGTH=30, OUTLIER_RATIO=0.2) # manually perturb (downgrade the data)
    df.to_csv(r"E:\I24-postprocess\benchmark\TM_1000_pollute.csv", index=False) # save the downgraded data
    print("saved.")
    # %% visualize in time-space diagram
    plot_time_space(df, lanes=[1], time="Frame #", space="x", ax=None, show =True)
    
    # %% examine an individual track by its ID
    car = df[df["ID"]==13]
    vis.plot_track_df(car[80:180])
    