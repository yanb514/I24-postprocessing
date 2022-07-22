# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 09:25:43 2021

@author: wangy79
This utils file contains functions that compute

"""
import numpy as np
# from shapely.geometry import Polygon
# from scipy.stats import variation
import numpy.linalg as LA

def iou_ts(a,b):
    """
    Description
    -----------
    Calculates intersection over union for track a and b in time-space diagram

    Parameters
    ----------
    a : 1x8
    b : 1x8
    Returns
    -------
    iou - float between [0,1] 
    """
    a,b = np.reshape(a,(1,-1)), np.reshape(b,(1,-1))
    if np.isnan(np.sum(a)+np.sum(b)):
        return 0
    p = Polygon([(a[0,2*i],a[0,2*i+1]) for i in range(4)])
    q = Polygon([(b[0,2*i],b[0,2*i+1]) for i in range(4)])
    intersection_area = p.intersection(q).area
    union_area = min(p.area, q.area)
    iou = float(intersection_area/union_area)
      
    return iou
    
def get_invalid(df, ratio=0.4):
    '''
    valid: length covers more than RATIO percent of the FOV
    invalid: length covers less than 10% of FOV, or
            crashes with any valid tracks
    undetermined: tracks that are short but not overlaps with any valid tracks
    '''
    
    xmin, xmax = min(df["x"].values),max(df["x"].values)
    fmin, fmax = min(df["Frame #"].values),max(df["Frame #"].values)
    groups = df.groupby("ID")
    # s = ['bbr_x','bbr_y','fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
    
    valid = {}
    collision = set()
    for carid, group in groups:
        frames = group["Frame #"].values
        if (max(group.x.values)-min(group.x.values)>ratio*(xmax-xmin)) or ((fmin<=min(frames)<=fmin+2) or (fmax-2<=max(frames)<=fmax)): # long tracks and boundary tracks
            first = group.head(1)
            last = group.tail(1)
            x0, x1 = max(first.bbr_x.values[0],first.fbr_x.values[0]),min(first.bbr_x.values[0],first.fbr_x.values[0])
            x2, x3 = min(last.bbr_x.values[0],last.fbr_x.values[0]),max(last.bbr_x.values[0],last.fbr_x.values[0])
            y0, y1 = max(first.bbr_y.values[0],first.bbl_y.values[0]),min(first.bbr_y.values[0],first.bbl_y.values[0])
            y2, y3 = min(last.bbr_y.values[0],last.bbl_y.values[0]),max(last.bbr_y.values[0],last.bbl_y.values[0])
            t0,t1 = min(frames), max(frames)
            valid[carid] = [np.array([t0,x0,t0,x1,t1,x2,t1,x3]),np.array([t0,y0,t0,y1,t1,y2,t1,y3])]
            
    # check crash within valid
    valid_list = list(valid.keys())
    for i,car1 in enumerate(valid_list):
        bx,by = valid[car1]
        for car2 in valid_list[i+1:]:
            ax,ay = valid[car2]
            ioux = iou_ts(ax,bx)
            iouy = iou_ts(ay,by)
            if ioux > 0 and iouy > 0: # trajectory overlaps with a valid track
                if bx[4]-bx[0] > ax[4]-ax[0]: # keep the longer track    
                    collision.add(car2)
                else:
                    collision.add(car1)
    valid = set(valid.keys())
    valid = valid-collision
    # print("Valid tracklets: {}/{}".format(len(valid),len(groupList)))

    return valid, collision

def get_lane_change(df):
    # check lane-change vehicles
    groups = df.groupby("ID")
    multiple_lane = set()
    for carid, group in groups:
        if group.lane.nunique()>1:
            if np.abs(np.max(group[["bbr_y","bbl_y"]].values)-np.min(group[["bbr_y","bbl_y"]].values)) > 12/3.281:
                multiple_lane.add(carid)
    return multiple_lane

def get_multiple_frame_track(df):
    groups = df.groupby("ID")
    multiple_frame = {}
    for carid, group in groups:
        count = len(group) - group["Frame #"].nunique()
        if count > 0:
            multiple_frame[carid] = count
    return multiple_frame

def mark_outliers_car(car):
    '''
    Identify outliers based on y,w,l
    mark as outliers but not removing
    '''
    my,sy = np.nanmedian(car.y.values), np.nanstd(car.y.values)
    mw,sw = np.nanmedian(car.width.values), np.nanstd(car.width.values)
    ml,sl = np.nanmedian(car.length.values), np.nanstd(car.length.values)
    
    valid = np.logical_and.reduce(((my-sy*2)<=car.y.values, car.y.values<=(my+2*sy), 
                           (mw-sw*2)<=car.width.values, car.width.values<=(mw+2*sw), 
                           (ml-sl*2)<=car.length.values, car.length.values<=(ml+2*sl)))
    car["Generation method"][~valid]='outlier'
    return car
    
def get_x_covered(df, ratio=True):
    def xrange(car):
        return max(car.x.values)-min(car.x.values)
    xranges = {}
    groups = df.groupby("ID")
    for carid,group in groups:
        xranges[carid] = xrange(group)
    if ratio:
        fov = max(df.x.values)-min(df.x.values)
        for key in xranges:
            xranges[key] /= fov
    return xranges

def get_variation(df, col):
    '''
    return a dictionary with key=carid, value=coefficient of variation (var/mean)
    '''
    output = {}
    groups = df.groupby("ID")
    for carid,group in groups:
        var = variation(group[col].values, axis=0)
        if np.isnan(var):
            continue
        output[carid] = var
    return output
    
def get_correction_score(df_raw, df_rec):
    groups_raw = df_raw.groupby("ID")
    groups_rec = df_rec.groupby("ID")
    
    pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
    
    scores = {}
    for carid, rec in groups_rec:
        # rec = np.array(rec[pts]).astype("float")
        try:
            raw = groups_raw.get_group(carid)
            b = rec["Frame #"].isin(raw["Frame #"].values)
            diff = np.array(raw[pts]).astype("float")-np.array(rec.loc[b,:][pts]).astype("float")
            score = np.nanmean(LA.norm(diff,axis=1))
            scores[carid] = score
        except:
            pass
    return scores


