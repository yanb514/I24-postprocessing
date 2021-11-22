# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 21:03:14 2021

@author: wangy79
benmark using TransModeler simulation data
"""
import utils
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from numpy import sin,cos
import utils_optimization as opt
import utils_vis as vis
from tqdm import tqdm

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
        try:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        except:
            pass

        ax.set_xlabel(time)
        ax.set_ylabel(space)
        ax.set_title('Lane {}'.format(lane_idx)) 
    return None if show else ax

def standardize(df):
    # distance is in feet -> m (/3.281)
    # time: sec
    # speed mph -> m/s (/2.237)
    df["Distance"] /= 3.281
    df["Speed"] /= 2.237
    df = df.drop(columns=['Latitude', 'Longitude', 'Segment', "Offset", "Heading", "Mileage"])
    # df = df.rename(columns={"Distance":"x"})
    df["Distance"] = df["Distance"].values - min(df["Distance"].values)
    
    # resample 1hz to 30hz
    return df

def calc_state(df):
    '''
    1. get y-axis based on lane idx, 1=HOV
    2. get smooth y-axis
    3. calc vx, vy, theta
    4. estimate l,w  based on vehicle class
    '''
    veh = {1:"sedan",
           2:"sedan",
           3:"sedan",
           4:"SUV",
           5:"truck",
           6:"trailer",
           7:"bus"}
    dim = {"sedan": [4.5, 1.7],
           "SUV": [4.5, 1.9],
           "truck": [8, 2],
           "bus": [12, 2.2],
           "trailer": [12, 2.2]}
    y = (np.arange(0,12*4,12)+6)/3.281
    y = y[::-1]
    y_arr = [y[i-1] for i in df.Lane.values]
    cls_arr = df.Class.values
    for i,c in enumerate(cls_arr):
        try:
            cls_arr[i] = int(c)
        except:
            cls_arr[i] = 1
    veh_arr = [veh[i] for i in cls_arr]
    l_arr = [dim[i][0] for i in veh_arr]
    w_arr = [dim[i][1] for i in veh_arr]
    df["Class"] = veh_arr
    df["Width"] = w_arr
    df["Length"] = l_arr
    df["y"] = y_arr
    df["Time"] = df["Time"].values - min(df["Time"].values)
        
    return df

def resample_single(car):
    '''
    resample from 1hz to 30hz
    car is each car
    '''
    if len(car)<3:
        return None
    time = car.Time.values
    newtime = np.arange(time[0], time[-1]+1/30, 1/30)# to 30hz
    d = car.Distance.values
    dir = np.sign(d[-1]-d[0])
    v = np.abs(np.diff(d))
    v = np.hstack([v,v[-1]])
    y = car.y.values
    vy = np.diff(y)
    vy = np.hstack([vy, vy[-1]])
    theta = np.arccos(dir)-np.arcsin(vy/v)
    
    thetare = np.interp(newtime, time, theta)
    vre = np.interp(newtime, time, v)
    
    new_index = pd.Index(newtime, name="Time")
    car = car.set_index("Time").reindex(new_index).reset_index()
    # interpolate / fill nan
    pts_fixed = ["ID","Class","Dir","Width","Length"]
    car["Time"] = newtime
    car["theta"] = thetare
    car["speed"] = vre
    car["direction"] = dir
    # copy the rest column
    car[pts_fixed] = car[pts_fixed].interpolate(method='pad')
         
    return car
  
def generate_meas(car):
    pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
    w = car.Width.values
    l = car.Length.values
    x0 = car.Distance.values[0]
    y0 = car.y.values[0]
    theta=car.theta.values
    theta = smooth(theta, 50) # TODO: window to be testeds
    v=car.speed.values
    Yre,x,y,a = opt.generate(w,l,x0, y0, theta, v, outputall=True)
    car["x"] = x
    car["y"] = y
    car["acceleration"] = a
    car["theta"] = theta
    car.loc[:,pts] = Yre 
    return car
    
def preprocess(df):
    tqdm.pandas()
    df = df.groupby('ID').apply(resample_single).reset_index(drop=True) 
    tqdm.pandas()
    df = df.groupby('ID').apply(generate_meas).reset_index(drop=True) 
    df = utils.assign_lane(df)
    
    # standardize for csv reader
    df = df.rename(columns={"Time":"Timestamp", "Class": "Object class", "Width":"width", "Length":"length"})
    df['Frame #'] = np.array(df["Timestamp"].values*30, dtype=int)
    col = ['Frame #', 'Timestamp', 'ID', 'Object class', 'BBox xmin','BBox ymin','BBox xmax','BBox ymax',
            'vel_x','vel_y','Generation method',
            'fbrx','fbry','fblx','fbly','bbrx','bbry','bblx','bbly','ftrx','ftry','ftlx','ftly','btrx','btry','btlx','btly',
            'fbr_x','fbr_y','fbl_x','fbl_y','bbr_x','bbr_y','bbl_x','bbl_y',
            'direction','camera','acceleration','speed','x','y','theta','width','length','height',"lane"]
    df = df.reindex(columns=col)
    # df = df[col]
    return df


# %%
if __name__ == "__main__":
    data_path = r"E:\I24-postprocess\benchmark\TM_trajectory.csv"
    df = pd.read_csv(data_path, nrows=1000)
    # df = df[:, 3000:4000]
    df = standardize(df)
    df = calc_state(df)
    df = preprocess(df)
    
    df.to_csv(r"E:\I24-postprocess\benchmark\TM_synth_1000.csv", index=False)
    # %%
    # plot_time_space(df, lanes=[1,2,3,4], time="Frame #", space="x", ax=None, show =True)
    # car = df[df["ID"]==16]
    
    # plt.plot(car.Time.values, car.theta.values)
    # plt.figure()
    # plt.plot(car.Time.values, car.y.values)