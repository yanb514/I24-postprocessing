import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin,cos
import cv2
import csv
import sys
import re
import glob
from tqdm import tqdm
import utils_optimization as opt
import data_association as da
from functools import partial
import itertools
import os
from multiprocessing import Pool, cpu_count


# read data
def read_data(file_name, skiprows = 0, index_col = False):     
#      path = pathlib.Path().absolute().joinpath('tracking_outputs',file_name)
    df = pd.read_csv(file_name, skiprows = skiprows,index_col = index_col) # error_bad_lines=False
    df = df.rename(columns={"GPS lat of bbox bottom center": "lat", "GPS long of bbox bottom center": "lon", 'Object ID':'ID'})
    return df
    
def remove_wrong_direction(car):
    direction = car["direction"].iloc[0]
    if np.mean(car["bbr_y"].values) < 18.5: # +1 direction   
        actual_direction = 1
    else:
        actual_direction = -1
    if actual_direction==direction:
        return car
    else:
        return None
    
def remove_wrong_direction_df(df):
    direction_x = np.sign(df["fbr_x"].values-df["bbr_x"].values).copy() # should be same as ys
    direction_y = np.sign(df["fbr_y"].values-df["fbl_y"].values).copy() # should be opposite to ys

    ys = df["y"].values.copy()

        
    ys[ys<18.5] = 1
    ys[ys>=18.5] = -1
    valid_x = direction_x == ys
    valid_y = direction_y*(-1) == ys
    df = df[np.logical_and(valid_x, valid_y)].reset_index(drop=True)
    return df
    
def reorder_points(df):
    '''
        make sure points in the road-plane do not flip
    '''
    pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
    Y = np.array(df[pts]).astype("float")
    xsort = np.sort(Y[:,[0,2,4,6]])
    ysort = np.sort(Y[:,[1,3,5,7]])
    try:
        if df['direction'].values[0]== 1:
            Y = np.array([xsort[:,0],ysort[:,0],xsort[:,2],ysort[:,1],
            xsort[:,3],ysort[:,2],xsort[:,1],ysort[:,3]]).T
        else:
            Y = np.array([xsort[:,2],ysort[:,2],xsort[:,0],ysort[:,3],
            xsort[:,1],ysort[:,0],xsort[:,3],ysort[:,1]]).T
    except np.any(xsort<0) or np.any(ysort<0):
        print('Negative x or y coord, please redefine reference point A and B')
        sys.exit(1)

    df.loc[:,pts] = Y
    return df

def filter_width_length(df):
    '''
    filter out bbox if their width/length is 2 std-dev's away
    '''
    pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']

    Y = np.array(df[pts]).astype("float")
    
    # filter outlier based on width    
    w1 = np.abs(Y[:,3]-Y[:,5])
    w2 = np.abs(Y[:,1]-Y[:,7])
    m = np.nanmean([w1,w2])
    s = np.nanstd([w1,w2])
    # print(m,s)
    outliers =    np.logical_or(abs(w1 - m) > 2 * s,abs(w2 - m) > 2 * s)
    # print('width outlier:',np.count_nonzero(outliers))
    Y[outliers,:] = np.nan
    
    # filter outlier based on length
    l1 = np.abs(Y[:,0]-Y[:,2])
    l2 = np.abs(Y[:,4]-Y[:,6])
    m = np.nanmean([l1,l2])
    s = np.nanstd([l1,l2])
    # print(m,s)
    outliers =    np.logical_or(abs(l1 - m) > 2 * s,abs(l2 - m) > 2 * s)
    # print('length outlier:',np.count_nonzero(outliers))
    Y[outliers,:] = np.nan
    
    # write into df
    df.loc[:,pts] = Y
    return df
    
def filter_short_track(df):

    Y1 = df['bbrx']
    N = len(Y1) 
    notNans = np.count_nonzero(~np.isnan(df['bbrx']))

    if (notNans <= 3) or (N <= 3):
        # print('track too short: ', df['ID'].iloc[0])
        return False
    return True
    

def findLongestSequence(car, k=0):
    '''
    keep the longest continuous frame sequence for each car
    # https://www.techiedelight.com/find-maximum-sequence-of-continuous-1s-can-formed-replacing-k-zeroes-ones/    
    '''
    A = np.diff(car['Frame #'].values)
    A[A!=1]=0
    
    left = 0        # represents the current window's starting index
    count = 0        # stores the total number of zeros in the current window
    window = 0        # stores the maximum number of continuous 1's found
                    # so far (including `k` zeroes)
 
    leftIndex = 0    # stores the left index of maximum window found so far
 
    # maintain a window `[left…right]` containing at most `k` zeroes
    for right in range(len(A)):
 
        # if the current element is 0, increase the count of zeros in the
        # current window by 1
        if A[right] == 0:
            count = count + 1
 
        # the window becomes unstable if the total number of zeros in it becomes
        # more than `k`
        while count > k:
            # if we have found zero, decrement the number of zeros in the
            # current window by 1
            if A[left] == 0:
                count = count - 1
 
            # remove elements from the window's left side till the window
            # becomes stable again
            left = left + 1
 
        # when we reach here, window `[left…right]` contains at most
        # `k` zeroes, and we update max window size and leftmost index
        # of the window
        if right - left + 1 > window:
            window = right - left + 1
            leftIndex = left
 
    # print the maximum sequence of continuous 1's
#      print("The longest sequence has length", window, "from index",
#          leftIndex, "to", (leftIndex + window - 1))
    return car.iloc[leftIndex:leftIndex + window - 1,:]
    
def preprocess(file_path, tform_path, skip_row = 0):
    '''
    preprocess for one single camera data
    skip_row: number of rows to skip when reading csv files to dataframe
    '''
    print('Reading data...')
    df = read_data(file_path,skip_row)
    if (df.columns[0] != 'Frame #'):
        df = read_data(file_path,9)
    if 'Object ID' in df:
        df = df.rename(columns={"Object ID": "ID"})
    if "veh rear x" in df:
        df = df.rename(columns={"veh rear x": "x", "veh center y":"y"})
    if 'frx' in df:
        df = df.rename(columns={"frx":"fbr_x", "fry":"fbr_y", "flx":"fbl_x", "fly":"fbl_y","brx":"bbr_x","bry":"bbr_y","blx":"bbl_x","bly":"bbl_y"})
    
    print('Total # cars before preprocessing:', len(df['ID'].unique()), len(df))
    camera_name = find_camera_name(file_path)
    # preprocess to make units international metrics
    if np.mean(df.y.values) > 40: 
        print("Converting units to meter...")
        cols_to_convert = ["speed","x","y","width","length","height"]
        pts = ["fbr_x","fbr_y","fbl_x","fbl_y","bbr_x","bbr_y","bbl_x","bbl_y"]
        df[cols_to_convert] = df[cols_to_convert] / 3.281
    if "bbr_x" not in df or np.mean(df.bbr_y.values) > 40:
        print("convert ft to m")
        df = img_to_road(df, tform_path, camera_name)
        
    df["x"] = np.array((df["bbr_x"].values+df["bbl_x"].values)/2)
    df["y"] = np.array((df["bbr_y"].values+df["bbl_y"].values)/2)
    
    print('Interpret missing timestamps...')
    frames = [min(df['Frame #']),max(df['Frame #'])]
    times = [min(df['Timestamp']),max(df['Timestamp'])]
    if np.isnan(times).any(): # if no time is recorded
        print('No timestamps values')
        p = np.poly1d([1/30,0]) # frame 0 corresponds to time 0, fps=30
    else:
        z = np.polyfit(frames,times, 1)
        p = np.poly1d(z)
    df['Timestamp'] = p(df['Frame #'])
    
    # print('Constrain x,y range by camera FOV')
    # if 'camera' not in df:
    #     df['camera'] = find_camera_name(file_path)
    # xmin, xmax, ymin, ymax = get_camera_range(df['camera'].dropna().unique())  
    # print(xmin, xmax, ymin, ymax)
    # pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
    # df.loc[(df['bbr_x'] < xmin) | (df['bbr_x'] > xmax), pts] = np.nan # 
    # df.loc[(df['bbr_y'] < ymin) | (df['bbr_y'] > ymax), pts] = np.nan # 
    # print('Total # car:', len(df['ID'].unique()), len(df))
    
    # print('Filtering out tailing place holders...')
    # df = df.groupby('ID').apply(remove_tailing_place_holders).reset_index(drop=True)
    # print('Total # car:', len(df['ID'].unique()), len(df))
    
    # print('Get the longest continuous frame chuck...')
    # df = df.groupby('ID').apply(findLongestSequence).reset_index(drop=True)
    # print('Total # car:', len(df['ID'].unique()),len(df))
    
    print('Get x direction...')
    df = get_x_direction(df)
    print('Total # car:', len(df['ID'].unique()),len(df))
    
    # print("Remove wrong direction...") # for MOTA accuracy only
    # df = df.groupby("ID").apply(remove_wrong_direction).reset_index(drop=True)
    # print('Total # car:', len(df['ID'].unique()),len(df))
    
    # print("reorder points...")
    # df = df.groupby("ID").apply(reorder_points).reset_index(drop=True)
    # print('Total # car:', len(df['ID'].unique()),len(df))
    
    # print('filter width length:', len(df['ID'].unique()))
    # df = df.groupby("ID").apply(filter_width_length).reset_index(drop=True)
    
    df = df.sort_values(by=['Frame #','Timestamp']).reset_index(drop=True) 
    print('Total # car:', len(df['ID'].unique()),len(df))
    return df

def preprocess_MC(file_path, skip_row = 0):
    '''
    preprocess for MC tracking data
    10/28/2021
    '''
    print('Reading data...')
    df = read_data(file_path,skip_row)
    if (df.columns[0] != 'Frame #'):
        df = read_data(file_path,9)
    if isinstance(df["Frame #"].iloc[0], str):
        print("Getting frame #...")
        df['Frame #'] = df.groupby("Timestamp").ngroup()
    if 'Object ID' in df:
        df = df.rename(columns={"Object ID": "ID"})
    if "veh rear x" in df:
        df = df.rename(columns={"veh rear x": "x", "veh center y":"y"})
    if 'frx' in df:
        df = df.rename(columns={"frx":"fbr_x", "fry":"fbr_y", "flx":"fbl_x", "fly":"fbl_y","brx":"bbr_x","bry":"bbr_y","blx":"bbl_x","bly":"bbl_y"})
    
    print('Total # cars before preprocessing:', len(df['ID'].unique()), len(df))

    if np.mean(df.y.values) > 40: 
        print("Converting units to meter...")
        cols_to_convert = ["speed","x","y","width","length","height"]
        pts = ["fbr_x","fbr_y","fbl_x","fbl_y","bbr_x","bbr_y","bbl_x","bbl_y"]
        df[cols_to_convert+pts] = df[cols_to_convert+pts] / 3.281
        
    df["x"] = np.array((df["bbr_x"].values+df["bbl_x"].values)/2)
    df["y"] = np.array((df["bbr_y"].values+df["bbl_y"].values)/2)

    
    print('Get x direction...')
    df = get_x_direction(df)
    print('Total # car:', len(df['ID'].unique()),len(df))
    
    df = df.sort_values(by=['Frame #','Timestamp']).reset_index(drop=True) 
    
    print("Assign lane ID...")
    df = assign_lane(df)
    
    
    return df


def remove_tailing_place_holders(car):
    notnan = ~np.isnan(np.sum(np.array(car[['bbr_x']]),axis=1))
    if np.count_nonzero(notnan)>0:
        start = np.where(notnan)[0][0]
        end = np.where(notnan)[0][-1]
        car = car.iloc[start:end+1]
    return car
    
def find_camera_name(file_path):
    camera_name_regex = re.compile(r'p(\d)*c(\d)*')
    camera_name = camera_name_regex.search(str(file_path))
    return camera_name.group()



def get_x_direction(df):
    return df.groupby("ID").apply(ffill_direction).reset_index(drop=True)

def ffill_direction(df):
    bbrx = df['bbr_x'].values
    notnan = ~np.isnan(bbrx)
    bbrx = bbrx[notnan]
    y = (df['bbr_y'].values[notnan]+df['bbl_y'].values[notnan])/2
    
    if (len(bbrx)<=1):
        if np.mean(y)<18:
            sign = 1
        else:
            sign = -1    
    else:
        sign = np.sign(bbrx[-1]-bbrx[0])
        
    df = df.assign(direction = sign)
    return df



def get_homography_matrix(camera_id, tform_path):
    '''
    camera_id: pxcx
    read from Derek's new transformation file
    '''
    # find and read the csv file corresponding to the camera_id
    # tf_file = glob.glob(str(tform_path) + '/' + camera_id + "*.csv")
    # tf_file = tf_file[0]
    # tf = pd.read_csv(tf_file)
    # M = np.array(tf.iloc[-3:,0:3], dtype=float)
    
    transform_path = glob.glob(str(tform_path) + '/' + camera_id + "*.csv")[0]
    # get transform from RWS -> ImS
    keep = []
    with open(transform_path,"r") as f:
        read = csv.reader(f)
        FIRST = True
        for row in read:
            if FIRST:
                FIRST = False
                continue
                    
            if "im space"  in row[0]:
                break
            
            keep.append(row)
    pts = np.stack([[float(item) for item in row] for row in keep])
    im_pts = pts[:,:2] # * 2.0
    lmcs_pts = pts[:,2:]
    H,_ = cv2.findHomography(im_pts,lmcs_pts)
# Minv = np.linalg.inv(M)
    return H
    
def img_to_road(df,tform_path,camera_id,ds=1):
    '''
    ds: downsample rate
    '''
    M = get_homography_matrix(camera_id, tform_path)
    for pt in ['fbr','fbl','bbr','bbl']:
        img_pts = np.array(df[[pt+'x', pt+'y']]) # get pixel coords
        img_pts = img_pts/ds # downsample image to correctly correspond to M
        img_pts_1 = np.vstack((np.transpose(img_pts), np.ones((1,len(img_pts))))) # add ones to standardize
        road_pts_un = M.dot(img_pts_1) # convert to gps unnormalized
        road_pts_1 = road_pts_un / road_pts_un[-1,:][np.newaxis, :] # gps normalized s.t. last row is 1
        road_pts = np.transpose(road_pts_1[0:2,:])/3.281 # only use the first two rows, convert from ft to m
        df[[pt+'_x', pt+'_y']] = pd.DataFrame(road_pts, index=df.index)
    return df

def euclidean_distance(lat1, lon1, lat2, lon2):
# https://math.stackexchange.com/questions/29157/how-do-i-convert-the-distance-between-two-lat-long-points-into-feet-meters
    r = 6371000
    lat1,lon1 = np.radians([lat1, lon1])
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    theta = (lat1+lat2)/2
    dx = r*cos(theta)*(lon2-lon1)
    dy = r*(lat2-lat1)
    d = np.sqrt(dx**2+dy**2)
    # d = r*np.sqrt((lat2-lat1)**2+(cos(theta)**2*(lon2-lon1)**2))
    return d,dx,dy
    
def transform_pt_array(point_array,M):
    """
    Applies 3 x 3  image transformation matrix M to each point stored in the point array
    """
    
    original_shape = point_array.shape
    
    num_points = int(np.size(point_array,0)*np.size(point_array,1)/2)
    # resize array into N x 2 array
    reshaped = point_array.reshape((num_points,2))   
    
    # add third row
    ones = np.ones([num_points,1])
    points3d = np.concatenate((reshaped,ones),1)
    
    # transform points
    tf_points3d = np.transpose(np.matmul(M,np.transpose(points3d)))
    
    # condense to two-dimensional coordinates
    tf_points = np.zeros([num_points,2])
    tf_points[:,0] = tf_points3d[:,0]/tf_points3d[:,2]
    tf_points[:,1] = tf_points3d[:,1]/tf_points3d[:,2]
    
    tf_point_array = tf_points.reshape(original_shape)
    
    return tf_point_array    
    
def img_to_road_box(img_pts_4,tform_path,camera_id):
    '''
    the images are downsampled
    img_pts: N x 8
    '''
    M = get_homography_matrix(camera_id, tform_path)
    print(img_pts_4.shape)
    road_pts_4 = np.empty([len(img_pts_4),0])
    for i in range(4):
        img_pts = img_pts_4[:,2*i:2*i+1]
        print(img_pts.shape)
        img_pts = img_pts/2 # downsample image to correctly correspond to M
        img_pts_1 = np.vstack((np.transpose(img_pts), np.ones((1,len(img_pts))))) # add ones to standardize
        road_pts_un = M.dot(img_pts_1) # convert to gps unnormalized
        road_pts_1 = road_pts_un / road_pts_un[-1,:][np.newaxis, :] # gps normalized s.t. last row is 1
        road_pts = np.transpose(road_pts_1[0:2,:])/3.281 # only use the first two rows, convert from ft to m
        road_pts_4 = np.hstack([road_pts_4, road_pts])
    return road_pts
    
def get_xy_minmax(df):

    if isinstance(df, pd.DataFrame):
        Y = np.array(df[['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']])
    else:
        Y = df
    notNan = ~np.isnan(np.sum(Y,axis=-1))
    Yx = Y[:,[0,2,4,6]]
    # print(np.where(Yx[notNan,:]==Yx[notNan,:].min()))
    Yy = Y[:,[1,3,5,7]]
    return Yx[notNan,:].min(),Yx[notNan,:].max(),Yy[notNan,:].min(),Yy[notNan,:].max()
    
def extend_prediction(car, args):
    '''
    extend the dynamics of the vehicles that are still in view
    constant acceleration model
    '''
    # print(car['ID'].iloc[0])
    xmin, xmax, minFrame, maxFrame = args
    dir = car['direction'].iloc[0]

    xlast = car['x'].iloc[-1]    
    xfirst = car['x'].iloc[0]
    
    pts_fixed = ["ID","Object class"]
    
    if (dir == 1) & (xlast < xmax):#
        car = forward_predict(car,xmin,xmax,'xmax',maxFrame)
        car[pts_fixed] = car[pts_fixed].interpolate(method='pad')
    if (dir == -1) & (xlast > xmin):
        car = forward_predict(car,xmin,xmax,'xmin', maxFrame)
        car[pts_fixed] = car[pts_fixed].interpolate(method='pad')
    if (dir == 1) & (xfirst > xmin):#tested
        car = backward_predict(car,xmin,xmax,'xmin', minFrame)
        car[pts_fixed] = car[pts_fixed].interpolate(method='pad')
    if (dir == -1) & (xfirst < xmax):
        car = backward_predict(car,xmin,xmax,'xmax', minFrame)
        car[pts_fixed] = car[pts_fixed].interpolate(method='pad')
    
    return car
        

def forward_predict(car,xmin,xmax,target, maxFrame):
    '''
    stops at maxFrame
    '''
    # print(car['ID'].iloc[0])
    # lasts
    framelast = car['Frame #'].values[-1]
    if framelast >= maxFrame:
        return car
    ylast = car['y'].values[-1]
    xlast = car['x'].values[-1]
    # vlast = car['speed'].values[-1]
    vlast = np.nanmean(car['speed'].values)
    # vlast = np.max(car['speed'].values)
    # alast = car['acceleration'].values[-1]
    if vlast < 1:
        return car
    
    w = car['width'].values[-1]
    l = car['length'].values[-1]
    h = car['height'].values[-1]
    dir = car['direction'].values[-1]
    # thetalast = np.mean(car['theta'].values)
    thetalast = np.arccos(dir)
    dt = 1/30

    # v = []
    x = [xlast]
    xfinal=xlast
    if target=='xmax':
        while xfinal < xmax:# and vlast > 0: 
            # vlast = vlast + alast*dt
            xfinal = xfinal + vlast*dt*cos(thetalast)
            x.append(xfinal)
            if len(x) > 1000:
                break
            # v.append(vlast)


    else:
        while xfinal > xmin:# and vlast > 0: # tested
            # vlast = vlast + alast*dt
            xfinal = xfinal + vlast*dt*cos(thetalast)
            x.append(xfinal)
            if len(x) > 1000:
                break
            # v.append(vlast)

    
    x = np.array(x)
    theta = np.ones(x.shape) * thetalast
    v = np.ones(x.shape) * vlast
    tlast = car['Timestamp'].values[-1]
    timestamps = np.linspace(tlast+dt, tlast+dt+dt*len(v), len(v), endpoint=False)

    # compute positions
    Yre,x,y,a = opt.generate(w,l,xlast+dt*vlast*cos(theta[0]),ylast+dt*vlast*sin(theta[0]),theta,v,outputall=True)
    
    frames = np.arange(framelast+1,framelast+1+len(x))
    pos_frames = frames<=maxFrame
    pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
    car_ext = {'Frame #': frames[pos_frames],
                'x':x[pos_frames],
                'y':y[pos_frames],
                'bbr_x': 0,
                'bbr_y': 0,
                'fbr_x': 0,
                'fbr_y': 0,
                'fbl_x': 0,
                'fbl_y': 0,
                'bbl_x': 0,
                'bbl_y': 0,
                'speed': v[pos_frames],
                'theta': theta[pos_frames],
                'width': w,
                'length':l,
                'height':h,
                'ID': car['ID'].values[-1],
                'direction': dir,
                'acceleration': 0,
                'Timestamp': timestamps[pos_frames],
                'Generation method': 'Extended'
                }
    car_ext = pd.DataFrame.from_dict(car_ext)
    car_ext[pts] = Yre[pos_frames,:]
    return pd.concat([car, car_ext], sort=False, axis=0)    
    

def backward_predict(car,xmin,xmax,target,minFrame):
    '''
    backward predict up until frame 0
    '''
    # first
    # print(car['ID'].iloc[0])
    framefirst = car['Frame #'].values[0]
    if framefirst <= minFrame:
        return car
    yfirst = car['y'].values[0]
    xfirst = car['x'].values[0]
    vfirst = np.nanmean(car['speed'].values)
    # vfirst = np.max(car['speed'].values)
    # vfirst = car['speed'].values[0]
    # afirst = car['acceleration'].values[1]

    if vfirst < 1:# or afirst < -5 or afirst > 5:
        return car
    # thetafirst = np.nanmean(car['theta'].values)
    w = car['width'].values[-1]
    l = car['length'].values[-1]
    h = car['height'].values[-1]
    dt = 1/30
    dir = car['direction'].values[-1]
    thetafirst = np.arccos(dir)
    # v = []
    x = []
    y = []

    if target=='xmax': # dir=-1
        while xfirst < xmax:# and vfirst > 0:    
            # vfirst = vfirst - afirst*dt
            xfirst = xfirst - vfirst*dt*cos(thetafirst)
            yfirst = yfirst - vfirst*dt*sin(thetafirst)
            x.insert(0,xfirst)
            y.insert(0,yfirst)
            if len(x) > 1000:
                break
            # v.insert(0,vfirst)
    else:
        while xfirst > xmin:# and vfirst > 0: 
            # vfirst = vfirst - afirst*dt
            xfirst = xfirst - vfirst*dt*cos(thetafirst)
            yfirst = yfirst - vfirst*dt*sin(thetafirst)
            x.insert(0,xfirst)
            y.insert(0,yfirst)
            if len(x) > 1000:
                break
            # v.insert(0,vfirst)
    
    # v = np.array(v)
    
    x = np.array(x)
    y = np.array(y)
    v = np.ones(x.shape)*vfirst
    theta = np.ones(v.shape) * thetafirst
    # a = np.ones(v.shape) * afirst
    tfirst = car['Timestamp'].values[0]
    timestamps = np.linspace(tfirst-dt-dt*len(v), tfirst-dt, len(v), endpoint=False)

    # compute positions
    xa = x + w/2*sin(theta)
    ya = y - w/2*cos(theta)
    xb = xa + l*cos(theta)
    yb = ya + l*sin(theta)
    xc = xb - w*sin(theta)
    yc = yb + w*cos(theta)
    xd = xa - w*sin(theta)
    yd = ya + w*cos(theta)
#     Yre = np.stack([xa,ya,xb,yb,xc,yc,xd,yd],axis=-1)        
    
    frames = np.arange(framefirst-len(x),framefirst)
    pos_frames = frames>=minFrame
    # discard frame# < 0
    car_ext = {'Frame #': frames[pos_frames],
                'x':x[pos_frames],
                'y':y[pos_frames],
                'bbr_x': xa[pos_frames],
                'bbr_y': ya[pos_frames],
                'fbr_x': xb[pos_frames],
                'fbr_y': yb[pos_frames],
                'fbl_x': xc[pos_frames],
                'fbl_y': yc[pos_frames],
                'bbl_x': xd[pos_frames], 
                'bbl_y': yd[pos_frames],
                'speed': v[pos_frames],
                'theta': thetafirst,
                'width': w,
                'length':l,
                'height':h,
                'ID': car['ID'].values[0],
                'direction': dir,
                'acceleration': 0,#a[pos_frames],
                'Timestamp': timestamps[pos_frames],
                'Generation method': 'Extended'
                }
    car_ext = pd.DataFrame.from_dict(car_ext)
    return pd.concat([car_ext, car], sort=False, axis=0)
    

    
def overlap_score(car1, car2):
    '''
    apply after rectify, check the overlap between two cars
    '''
    end = min(car1['Frame #'].iloc[-1],car2['Frame #'].iloc[-1])
    start = max(car1['Frame #'].iloc[0],car2['Frame #'].iloc[0])
    
    if end <= start: # if no overlaps
        return 999
    car1 = car1.loc[(car1['Frame #'] >= start) & (car1['Frame #'] <= end)]
    car2 = car2.loc[(car2['Frame #'] >= start) & (car2['Frame #'] <= end)]

    pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
    Y1 = np.array(car1[pts])
    Y2 = np.array(car2[pts])
    return np.sum(np.linalg.norm(Y1-Y2,2, axis=1))/(len(Y1))
    


def get_id_rem(df,SCORE_THRESHOLD):
    '''
    get all the ID's to be removed due to overlapping
    '''
    groups = df.groupby('ID')
    groupList = list(groups.groups)
#     nO = len(groupList)
    comb = itertools.combinations(groupList, 2)
    id_rem = [] # ID's to be removed

    for c1,c2 in comb:
        car1 = groups.get_group(c1)
        car2 = groups.get_group(c2)
    #      score = overlap_score(car1, car2)
        score = da.IOU_score(car1,car2)
#         IOU.append(score)
        if score > SCORE_THRESHOLD:
            # remove the shorter track
            if len(car1)>= len(car2):
                id_rem.append(c2)
            else:
                id_rem.append(c1)
    return id_rem



# delete repeated measurements per frame per object
del_repeat_meas = lambda x: x.head(1) if np.isnan(x['bbr_x'].values).all() else x[~np.isnan(x['bbr_x'].values)].head(1)

# x: df of measurements of same object ID at same frame, get average
def average_meas(x):
    mean = x.head(1)
    pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
    Y = np.array(x[pts])
    Y = np.nanmean(Y, axis=0)
    mean.loc[:,pts] = Y
    return mean
    
def del_repeat_meas_per_frame(framesnap):
    framesnap = framesnap.groupby('ID').apply(del_repeat_meas)
    return framesnap
    
def preprocess_multi_camera(data_path, tform_path):
    
    df = pd.DataFrame()
    for root,dirs,files in os.walk(str(data_path), topdown = True):
        for file in files:
            if file.endswith(".csv"):
                file_name = data_path.joinpath(file)
                camera_name = find_camera_name(file_name)
                print('*** Reading ',camera_name,'...')
                df1 = read_data(file_name,9)
                if 'Object ID' in df1:
                    df1.rename(columns={"Object ID": "ID"})
                print(len(df1['ID'].unique()))
                print('Transform from image to road...')
                
                df1 = img_to_road(df1, tform_path, camera_name)
                print('Deleting unrelavent columns...')
                df1 = df1.drop(columns=['BBox xmin','BBox ymin','BBox xmax','BBox ymax','vel_x','vel_y','lat','lon'])
                df1 = df1.assign(camera=camera_name)
                df = pd.concat([df, df1])
        break
        
    # MUST SORT!!! OTHERWISE DIRECTION WILL BE WRONG
    df = df.sort_values(by=['Frame #','Timestamp']).reset_index(drop=True) 
    print('sorted.')
        
    print('Get x direction...')
    df = get_x_direction(df)
    
    print('Interpret missing timestamps...')
    frames = [min(df['Frame #']),max(df['Frame #'])]
    times = [min(df['Timestamp']),max(df['Timestamp'])]
    z = np.polyfit(frames,times, 1)
    p = np.poly1d(z)
    df['Timestamp'] = p(df['Frame #'])
    print('Get the longest continuous frame chuck...')
    df = df.groupby('ID').apply(findLongestSequence).reset_index(drop=True)
    return df
    
def preprocess_data_association(df,file,tform_path):
    '''
    stitch objects based on their predicted trajectories
    associate objects based on obvious overlaps
    '''
    tqdm.pandas()
    # associate based on overlaps (IOU measure)
    # parent = associate_overlaps(df)
    # df['ID'] = df['ID'].apply(lambda x: parent[x] if x in parent else x)  
    # print('After assocating overlaps: ', len(df['ID'].unique()), 'cars')
    print('cars: ', len(df['ID'].unique()), 'cars',len(df))
    df = da.remove_overlaps(df)
    print('After removing overlaps: ', len(df['ID'].unique()), 'cars',len(df))

    # stitch based on prediction (weighted distance measure)
    print('Before DA: ', len(df['ID'].unique()), 'cars')
    parent = da.stitch_objects(df)
    df['ID'] = df['ID'].apply(lambda x: parent[x] if x in parent else x)
    print('After stitching: ', len(df['ID'].unique()), 'cars, checking for overlaps...')
    
    
    print('Get the longest continuous frame chunk...')
    df = df.groupby('ID').apply(findLongestSequence).reset_index(drop=True)
    df = applyParallel(df.groupby("Frame #"), del_repeat_meas_per_frame).reset_index(drop=True)
    # added the following noticed error in current version
    camera = find_camera_name(file)
    df = img_to_road(df, tform_path,camera)
    df = df.groupby("ID").apply(reorder_points).reset_index(drop=True)
    return df
    
def applyParallel(dfGrouped, func, args=None):
    with Pool(cpu_count()) as p:
        if args is None:
            ret_list = list(tqdm(p.imap(func, [group for name, group in dfGrouped]), total=len(dfGrouped.groups)))
        else:# if has extra arguments
            ret_list = list(tqdm(p.imap(partial(func, args=args), [group for name, group in dfGrouped]), total=len(dfGrouped.groups)))
    return pd.concat(ret_list)    
    
def get_camera_range(camera_id_list):
    '''
    return xmin, xmax, ymin, ymax (in meter)
    '''
    ymin = -5
    ymax = 45
    xminfinal = 2000
    xmaxfinal = 0
    
    for camera_id in camera_id_list:
    
        if camera_id=='p1c1':
            xmin = 0
            xmax = 520
        elif camera_id=='p1c2':
            xmin = 400
            xmax = 740
        elif camera_id=='p1c3':
            xmin = 600
            xmax = 800
        elif camera_id=='p1c4':
            xmin = 600
            xmax = 800
        elif camera_id=='p1c5':
            xmin = 660
            xmax = 960
        elif camera_id=='p1c6':
            xmin = 760
            xmax = 1200
        elif camera_id=='p2c1':
            xmin = 400
            xmax = 900
        elif camera_id=='p2c2':
            xmin = 600
            xmax = 920
        elif camera_id=='p2c3':
            xmin = 700
            xmax = 1040
        elif camera_id=='p2c4':
            xmin = 700
            xmax = 1040
        elif camera_id=='p2c5':
            xmin = 780
            xmax = 1060
        elif camera_id=='p2c6':
            xmin = 800
            xmax = 1160
        elif camera_id=='p3c1':
            xmin = 800
            xmax = 1400
        elif camera_id=='p3c2':
            xmin = 1150
            xmax = 1450
        elif camera_id=='p3c3':
            xmin = 1220
            xmax = 1600
        elif camera_id=='p3c4':
            xmin = 1220
            xmax = 1600
        elif camera_id=='p3c5':
            xmin = 1350
            xmax = 1800
        elif camera_id=='p3c6':
            xmin = 1580
            xmax = 2000
        elif camera_id=='all':
            xmin = 0
            xmax = 2000
        else:
            print('no camera ID in get_camera_range')
            return
        xminfinal = min(xminfinal, xmin)
        xmaxfinal = max(xmaxfinal, xmax)
        
    return xminfinal/3.281, xmaxfinal/3.281, ymin, ymax
    
def post_process(df):
    # print('remove overlapped cars...')
    # id_rem = get_id_rem(df, SCORE_THRESHOLD=0) # TODO: untested threshold
    # df = df.groupby(['ID']).filter(lambda x: (x['ID'].iloc[-1] not in id_rem))
    print('cap width at 2.59m...')
    df = df.groupby("ID").apply(width_filter).reset_index(drop=True)
    
    # print('remove overlaps, before: ', len(df['ID'].unique()))
    # df = da.remove_overlaps(df)
    # print('remove overlaps, after: ', len(df['ID'].unique()))
    
    print('extending tracks to edges of the frame...')
    camera = df['camera'].iloc[0]
    xmin, xmax, ymin, ymax = get_camera_range([camera])
    minFrame = min(df['Frame #'])
    maxFrame = max(df['Frame #'])
    print(xmin, xmax)
    args = (xmin, xmax, minFrame, maxFrame)
    tqdm.pandas()
    # df = df.groupby('ID').apply(extend_prediction, args=args).reset_index(drop=True)
    df = applyParallel(df.groupby("ID"), extend_prediction, args=args).reset_index(drop=True)
    
    print('standardize format for plotter...')
    # if ('lat' in df):
        # df = df.drop(columns=['lat','lon'])
    df = df[['Frame #', 'Timestamp', 'ID', 'Object class', 'BBox xmin','BBox ymin','BBox xmax','BBox ymax',
            'vel_x','vel_y','Generation method',
            'fbrx','fbry','fblx','fbly','bbrx','bbry','bblx','bbly','ftrx','ftry','ftlx','ftly','btrx','btry','btlx','btly',
            'fbr_x','fbr_y','fbl_x','fbl_y','bbr_x','bbr_y','bbl_x','bbl_y',
            'direction','camera','acceleration','speed','x','y','theta','width','length','height']]
    return df

def get_camera_x(x):
    x = x * 3.281 # convert to feet
    if x < 640:
        camera = 'p1c2'
    elif x < 770:
        camera = 'p1c3'
    elif x < 920:
        camera = 'p1c5'
    else:
        camera = 'p1c6'
    return camera
    
def road_to_img(df, tform_path):
    # TODO: to be tested
    if 'camera_post' not in df:
        df['camera_post'] = df[['x']].apply(lambda x: get_camera_x(x.item()), axis = 1)
    groups = df.groupby('camera_post')
    df_new = pd.DataFrame()
    for camera_id, group in groups:
        M = get_homography_matrix(camera_id, tform_path)
        Minv = np.linalg.inv(M)
        for pt in ['fbr','fbl','bbr','bbl']:
            road_pts = np.array(df[[pt+'_x', pt+'_y']]) * 3.281
            road_pts_1 = np.vstack((np.transpose(road_pts), np.ones((1,len(road_pts)))))
            img_pts_un = Minv.dot(road_pts_1)
            img_pts_1 = img_pts_un / img_pts_un[-1,:][np.newaxis, :]
            img_pts = np.transpose(img_pts_1[0:2,:])*2
            group[[pt+'x',pt+'y']] = pd.DataFrame(img_pts, index=df.index)
        df_new = pd.concat([df_new, group])
    return df_new
    

def width_filter(car):
# post processing only
# filter out width that's wider than 2.59m
# df is the df of each car
    
    w = car['width'].values[-1]
    l = car['length'].values[-1]
    notNan = np.count_nonzero(~np.isnan(np.sum(np.array(car[['bbr_x']]),axis=1)))
    if (w < 2.59) & (notNan == len(car)):
        return car
    theta = car['theta'].values
#     dt=1/30
    
    
    if w > 2.59:
        w = 2.59
    # a = car['acceleration'].values
    # a = np.nan_to_num(a) # fill nan with zero
    v = car['speed'].values
    x = car['x'].values
    y = car['y'].values
    # dir = car['direction'].values[0]
    # for i in range(1,len(car)):
        # v[i] = v[i-1] + a[i-1] * dt
        # x[i] = x[i-1] + dir*v[i-1] * dt
        # y[i] = y[i-1]
    # compute new positions
    xa = x + w/2*sin(theta)
    ya = y - w/2*cos(theta)
    xb = xa + l*cos(theta)
    yb = ya + l*sin(theta)
    xc = xb - w*sin(theta)
    yc = yb + w*cos(theta)
    xd = xa - w*sin(theta)
    yd = ya + w*cos(theta)
    
    car['width'] = w
    car['x'] = x
    car['y'] = y
    car['bbr_x'] = xa
    car['bbr_y'] = ya
    car['fbr_x']= xb
    car['fbr_y']= yb
    car['fbl_x']= xc
    car['fbl_y']= yc
    car['bbl_x']= xd
    car['bbl_y']= yd
    car['speed']= v
    
    return car

def connect_track(car):
    '''
    check if a track has missing frames in the middle
    if yes, create missing data place holders for those missing frames
    otherwise do nothing
    '''

    if car["Frame #"].iloc[-1]-car["Frame #"].iloc[0]+1 > len(car):
        # print("connect:", car["ID"].iloc[0])

        frames = np.arange(car["Frame #"].iloc[0],car["Frame #"].iloc[-1]+1)
        new_index = pd.Index(frames, name="Frame #")
        car = car.set_index("Frame #").reindex(new_index).reset_index()
        # interpolate / fill nan
        pts_fixed = ["ID","Object class","direction","width","length","height"]
        car["Timestamp"] = car.Timestamp.interpolate()
        
        # copy the rest column
        car[pts_fixed] = car[pts_fixed].interpolate(method='pad')
        
        # car["Object class"] = car["Object class"].interpolate(method='pad')
        # car["camera"] = car["camera"].interpolate(method='pad')
        # car["direction"] = car["direction"].interpolate(method='pad')

    return car
    

def assign_lane(df):
    '''
    assign lane number to each measurement box based on y
    assume lane width is 12ft
    lane 1 is the right-most lane in +1 direction
    '''
    if 'y' in df:
        y = df['y'].values
    else:
        y = (df['bbr_y'].values + df['bbl_y'].values)/2
    # if y is nan, this current version will assign that to lane 11
    lane_edges = np.arange(0,12*11,12)/3.281 # 10 lanes in total
    lane_idx = np.digitize(y, lane_edges)
    df['lane'] = lane_idx
    df.loc[df.lane == 11, "lane"] = np.nan
    return df

def calc_dynamics_car(car):
    # dt = np.diff(car["Timestamp"].values)
    direction = car['direction'].iloc[0]
    if len(car)<3:
        return
    dt = np.array([1/30]*(len(car)-1))
    dx = np.diff(car.x.values)
    vx = dx/dt*direction
    vx = np.append(vx, vx[-1])
    dy = np.diff(car.y.values)
    vy = dy/dt
    vy = np.append(vy, vy[-1])
    v = np.sqrt(vx**2+vy**2)
    theta = np.arccos(direction) - np.arcsin(vy/v)
    a = np.diff(v)/dt
    a = np.append(a, a[-1])
    car.loc[:,"speed"] = v
    car.loc[:,"acceleration"] = a
    car.loc[:,"theta"] = theta
    return car

def constant_speed(car):
    temp = car[~car["bbr_x"].isna()]
    if len(temp)<2:
        return None
    v_bbr = (max(temp.bbr_x.values)-min(temp.bbr_x.values))/(max(temp.Timestamp.values)-min(temp.Timestamp.values))
    v_fbr = (max(temp.fbr_x.values)-min(temp.fbr_x.values))/(max(temp.Timestamp.values)-min(temp.Timestamp.values))
    avgv = (v_bbr+v_fbr)/2
    car["speed"] = avgv if avgv<50 else np.nan
    return car    

def calc_dynamics(df):
    df = df.groupby("ID").apply(calc_dynamics_car).reset_index(drop=True)
    return df

# path compression
def find(parent, i):
	if (parent[i] != i):
		parent[i] = find(parent, parent[i])
	return parent[i]

def compress(parent, groupList):	
	for i in groupList:
		find(parent, i)
	return parent  

def mark_outliers(car):
    '''
    mark outlier by comparing x,y with constant-velocity x and y
    mark as outlier_xy
    '''
    # get data
    x = car.x.values
    y = car.y.values
    frames = car["Frame #"].values
    
    # constant velocity
    notnan = ~np.isnan(x)
    # x = x[notnan] # remove rows with missing values (dim = mx8)
    # y = y[notnan]
    # frames = frames[notnan]

    fitx = np.polyfit(frames[notnan],x[notnan],1)
    fity = np.polyfit(frames[notnan],y[notnan],1)
    xhat = np.polyval(fitx, frames)
    yhat = np.polyval(fity, frames)
   
    xdiff = x-xhat
    ydiff = y-yhat
    
    # get 10,90% quantile
    q1x = np.nanquantile(xdiff, 0.1)
    q2x = np.nanquantile(xdiff, 0.9)
    q1y = np.nanquantile(ydiff, 0.1)
    q2y = np.nanquantile(ydiff, 0.9)
    
    # get outlier indices
    outliers = np.logical_or.reduce((xdiff<q1x, xdiff>q2x, ydiff<q1y, ydiff>q2y))
    car["Generation method"][outliers]='outlier1'
    
    return car