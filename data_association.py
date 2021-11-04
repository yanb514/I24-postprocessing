# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 10:45:24 2021

@author: wangy79
Data association class consists of 
- data preprocess
    - timestamp & frame # interpolation
    - unit conversion
    - get direction
    - get lane info
    
- object stitching
    - GNN: global nearest neighbor (online)
    - BM: bipartite matching (online)
    - JPDA: joint probablistic data association (online)
    - TSM: Time-space matching (offline)
    
- data postprocess
    - iteratively remove and save valid tracks
    - output invalid (fragmented / wrong direction / overlapped) tracks
    - outlier removal
    - connect tracks
    
- visualization
    - time-space diagrams for original data and DA'ed data
    - # invalid tracks removed
    - spacing distribution B&A

"""
import numpy as np
import utils
import itertools
from shapely.geometry import Polygon
import pandas as pd
import collections
import utils_vis as vis
import scipy
import matplotlib.pyplot as plt


class Data_Association():
    
    def __init__(self, data_path, params = None):
        '''
        params = {"method": "gnn",
          "start": 0,
          "end": 1000,
          "lanes": []
          "plot_start": 0, # for plotting tracks in online methods
          "plot_end": 10,
          "preprocess": True
          }
        '''
        
        self.params = params 
        if params["preprocess"]:
            self.df = utils.preprocess_MC(data_path)
        else:
            self.df = utils.read_data(data_path)
        self.df = self.df[(self.df["Frame #"] >= params["start"]) & (self.df["Frame #"] <= params["end"])]
        if len(params["lanes"]) > 0:
            self.df = self.df[self.df["lane"].isin(params["lanes"])]
        self.original = self.df.copy()
    
    
    def dist_score(B, B_data, DIST_MEAS='xyw', DIRECTION=True):
        '''
        compute euclidean distance between two boxes B and B_data
        B: predicted bbox location ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
        B_data: measurement
        '''
        B = np.reshape(B,(-1,8))
        B_data = np.reshape(B_data,(-1,8))
        
        # check sign
        if DIRECTION==True:
            if (np.sign(B[0,[2]]-B[0,[0]])!=np.sign(B_data[0,[2]]-B_data[0,[0]])) : # if not the same x direction
                return 99
        
        diff = B-B_data
        diff = diff[0]
        
        if DIST_MEAS == 'xy':
            # return np.linalg.norm(B-B_data,2) # RMSE
            mae_x = np.mean(np.abs(diff[[0,2,4,6]])) 
            mae_y = np.mean(np.abs(diff[[1,3,5,7]])) 
            return (mae_x + mae_y)/2
    
        # weighted x,y displacement, penalize y more heavily
        elif DIST_MEAS == 'xyw':
            alpha = 0.2
            mae_x = np.mean(np.abs(diff[[0,2,4,6]])) 
            mae_y = np.mean(np.abs(diff[[1,3,5,7]])) 
            # return alpha*np.linalg.norm(B[[0,2,4,6]]-B_data[[0,2,4,6]],2) + (1-alpha)*np.linalg.norm(B[[1,3,5,7]]-B_data[[1,3,5,7]],2)
            return alpha*mae_x + (1-alpha)*mae_y
        
        # mahalanobis distance
        elif DIST_MEAS == 'maha':
            alpha = (1/3.5)**2
            beta = (1/0.27)**2
            d2 = 0
            for i in range(4):
                d2 += np.sqrt(alpha*diff[i]**2+beta*diff[2*i+1]**2)
            return d2/4
        
        # euclidean distance
        elif DIST_MEAS == 'ed':
            d2 = 0
            for i in range(4):
                d2 += np.sqrt(diff[i]**2+diff[2*i+1]**2)
            return d2/4
        else:
            return

    def iou(self,a,b,DIRECTION=True,AREA=False):
        """
        Description
        -----------
        Calculates intersection over union for all sets of boxes in a and b
    
        Parameters
        ----------
        a : tensor of size [8,3] 
            bounding boxes in relative coords
        b : array of size [8,3] 
            bounding boxes in relative coords
            ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
        Returns
        -------
        iou - float between [0,1] if a, b are valid boxes, -1 otherwise
            average iou for a and b
        """
        a,b = np.reshape(a,(1,-1)), np.reshape(b,(1,-1))

        # if has invalid measurements
        if np.isnan(sum(sum(a))) or np.isnan(sum(sum(b))):
            if AREA==True:    
                return 0,-1,-1
            else: return 0
        
        ax = np.sort(a[0,[0,2,4,6]])
        ay = np.sort(a[0,[1,3,5,7]])
        bx = np.sort(b[0,[0,2,4,6]])
        by = np.sort(b[0,[1,3,5,7]])

        area_a = (ax[3]-ax[0]) * (ay[3]-ay[0])
        area_b = (bx[3]-bx[0]) * (by[3]-by[0])
        
        if DIRECTION==True:
            if (np.sign(a[0,[2]]-a[0,[0]])!=np.sign(b[0,[2]]-b[0,[0]])):# if not the same x / y direction
                if AREA==True:    
                    return -1,area_a,area_b
                else:
                    return -1
        
        minx = max(ax[0], bx[0]) # left
        maxx = min(ax[2], bx[2])
        miny = max(ay[1], by[1])
        maxy = min(ay[3], by[3])
        
        intersection = max(0, maxx-minx) * max(0,maxy-miny)
        # union = area_a + area_b - intersection + 1e-06
        union = min(area_a,area_b)
        iou = intersection/union
        if AREA==True:
            return iou,area_a,area_b
        else: return iou
        
        
    def predict_tracks_df(self, tracks):
        '''
        tracks: [dictionary]. Key: car_id, value: df
        if a track has only 1 frame, assume 25m/s
        otherwise do constant-velocity one-step-forward prediction
        Return: 
            x: last predicted position: array of n_car x 8
            tracks: updated dictionary
        '''
        x = []
        pts = ['bbr_x','bbr_y','fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
        v = 30 #m/s
        mpf = v/30
        
        for car_id, track_df in tracks.items():
            # direction = track_df["direction"].iloc[0]
            # direction = np.sign(track_df["fbr_x"].iloc[-1]-track_df["bbr_x"].iloc[-1])
            track = np.array(track_df[pts])
            direction = np.sign(track[-1][2]-track[-1][0])
            
            if len(track)>1:   # average speed
    
                frames = np.arange(0,len(track))
                fit = np.polyfit(frames,track,1)
                est_speed = np.mean(fit[0,[0,2,4,6]])
                x_pred = np.polyval(fit, len(track))
                if abs(est_speed)<mpf/2 or (np.sign(est_speed)!=direction) or (abs(x_pred[0]-x_pred[2])<1): # too slow
                    x_pred = track[-1,:] + direction* np.array([mpf,0,mpf,0,mpf,0,mpf,0])
    
            else:
                x_pred = track[-1,:] + direction* np.array([mpf,0,mpf,0,mpf,0,mpf,0])
            x_pred = np.reshape(x_pred,(1,-1))
            x.append(x_pred) # prediction next frame, dim=nx8
            new_row = pd.DataFrame(x_pred, columns=pts)
            tracks[car_id] = pd.concat([tracks[car_id], new_row])
        return x, tracks
    
    def stitch_objects_gnn(self, THRESHOLD_1, THRESHOLD_2):
         # define the x,y range to keep track of cars in FOV (meter)
    
        xmin, xmax = min(self.df["x"].values)-10,max(self.df["x"].values)+10
    
        ns = int(np.amin(np.array(self.df[['Frame #']]))) # start frame
        nf = int(np.amax(np.array(self.df[['Frame #']]))) # end frame
        tracks = dict() # a dictionary to store all current objects in view. key:ID, value:dataframe
        pts = ['bbr_x','bbr_y','fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
        # pts_img = ["fbrx","fbry","fblx","fbly",	"bbrx",	"bbry",	"bblx",	"bbly",	"ftrx",	"ftry",	"ftlx",	"ftly",	"btrx",	"btry",	"btlx",	"btly"]
        newdf = pd.DataFrame()
        
        for k in range(ns,nf):
            print("\rFrame {}/{}".format(k,nf),end = "\r",flush = True)
            
            frame = self.df.loc[(self.df['Frame #'] == k)] # TODO: use groupby frame to save time
            y = np.array(frame[pts])
            notnan = ~np.isnan(y).any(axis=1)
            y = y[notnan] # remove rows with missing values (dim = mx8)
            frame = frame.iloc[notnan,:]
            frame = frame.reset_index(drop=True)
            
            m_box = len(frame)
            n_car = len(tracks)
            # invalid_tracks = set()
            
            if (n_car > 0): # delete track that are out of view
                for car_id in list(tracks.keys()):
                    # delete track if total matched frames < 
                    last_frame = tracks[car_id].iloc[-1]
                    last_frame_x = np.array(last_frame[pts])[[0,2,4,6]]
                    x1,x2 = min(last_frame_x),max(last_frame_x)
                    frames = tracks[car_id]["Frame #"].values
                    matched_bool = ~np.isnan(frames)
                    frames_matched = tracks[car_id].loc[matched_bool]
    
                    if (x1<xmin) or (x2>xmax):
                        if len(frames_matched) > 0: # TODO: this threshold could be a ratio
                            newid = frames_matched["ID"].iloc[0] 
                            frames_matched["ID"] = newid #unify ID
                            newdf = pd.concat([newdf,frames_matched])
                        del tracks[car_id]
                        n_car -= 1
            
            if (m_box == 0) and (n_car == 0): # simply advance to the next frame
                continue
                
            elif (m_box == 0) and (n_car > 0): # if no measurements in current frame, simply predict
                x, tracks = self.predict_tracks_df(tracks)
                
            elif (m_box > 0) and (n_car == 0): # create new tracks (initialize)
                for i, row in frame.iterrows():
                    row = frame.loc[i:i,:]
                    tracks[row['ID'].iloc[0]] = row
            
            else: 
                x, tracks = self.predict_tracks_df(tracks)
                n_car = len(tracks)
                curr_id = list(tracks.keys()) # should be n id's 
    
                # score = np.ones([m_box,n_car])*(99)
                score_dist = np.zeros([m_box,n_car])
                score_iou =np.zeros([m_box,n_car])
                
                invalid_meas = set()
                # invalid_tracks = set()
                for m in range(m_box):
                    for n in range(n_car):
                        score_dist[m,n] = self.dist_score(x[n],y[m],'maha')
                        score_iou[m,n], areaa, areab = self.iou(x[n],y[m],DIRECTION=False,AREA=True)
                        # if areaa < 0.5:
                        #     invalid_tracks.add(n)
                    if areab < 1:
                        invalid_meas.add(m)
     
                # if (1120<k<1130):  
                #     vis.plot_track(np.array(np.vstack(x), dtype=float), np.array(y,dtype=float), curr_id, frame["ID"].values, xmin,xmax, k)
                # if 333 in curr_id:
                #     print("")
                gate = np.logical_or(score_dist<THRESHOLD_1, score_iou>0)
                matched_length = []
                for carid in curr_id:
                    track = tracks[carid]
                    matched_length.append(track["Frame #"].count())
                pq = np.argsort(matched_length)[::-1] # priority queue
                matched_m = set() 
                for n in pq:
                    if not any(gate[:,n]): # no matched meas for this track
                        continue
                    # find the best match meas for this track
                    tracks[curr_id[n]] = tracks[curr_id[n]].reset_index(drop=True)
                    idx_in_gate = np.where(gate[:,n])[0]
                    best_idx = np.argmin(score_dist[idx_in_gate,n])
                    m = idx_in_gate[best_idx]
                    avg_meas = frame.loc[m:m] 
                    tracks[curr_id[n]].drop(tracks[curr_id[n]].tail(1).index,inplace=True) # drop the last row (prediction)
                    tracks[curr_id[n]] = pd.concat([tracks[curr_id[n]], avg_meas],ignore_index=True)  
                    gate[m,:] = False # elimite m from future selection
                    matched_m.add(m)
    
                        
                m_unassociated = set(np.arange(m_box))-matched_m
                for m in m_unassociated:
                    # !TODO: make sure that y[m] at not in the gate of each other
                    if (m not in invalid_meas):
                        new_id = frame['ID'].iloc[m]
                        new_meas = frame.loc[m:m]
                        tracks[new_id] = new_meas
     
        self.df = newdf
        return newdf
        
    def stitch_objects_bm(self, THRESHOLD_1, THRESHOLD_2):
        '''
        bipartite matching based on Maha distance cost
        '''
        xmin, xmax = min(self.df["x"].values)-10,max(self.df["x"].values)+10

        ns = int(np.amin(np.array(self.df[['Frame #']]))) # start frame
        nf = int(np.amax(np.array(self.df[['Frame #']]))) # end frame
        tracks = dict() # a dictionary to store all current objects in view. key:ID, value:dataframe
        pts = ['bbr_x','bbr_y','fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
        # pts_img = ["fbrx","fbry","fblx","fbly",	"bbrx",	"bbry",	"bblx",	"bbly",	"ftrx",	"ftry",	"ftlx",	"ftly",	"btrx",	"btry",	"btlx",	"btly"]
        newdf = pd.DataFrame()
        
        for k in range(ns,nf):
            print("\rFrame {}/{}".format(k,nf),end = "\r",flush = True)
            
            frame = self.df.loc[(self.df['Frame #'] == k)] # TODO: use groupby frame to save time
            y = np.array(frame[pts])
            notnan = ~np.isnan(y).any(axis=1)
            y = y[notnan] # remove rows with missing values (dim = mx8)
            frame = frame.iloc[notnan,:]
            frame = frame.reset_index(drop=True)
            
            m_box = len(frame)
            n_car = len(tracks)
            invalid_tracks = set()
            
            if (n_car > 0): # delete track that are out of view
                for car_id in list(tracks.keys()):
                    # delete track if total matched frames < 
                    last_frame = tracks[car_id].iloc[-1]
                    last_frame_x = np.array(last_frame[pts])[[0,2,4,6]]
                    x1,x2 = min(last_frame_x),max(last_frame_x)
                    frames = tracks[car_id]["Frame #"].values
                    matched_bool = ~np.isnan(frames)
                    frames_matched = tracks[car_id].loc[matched_bool]
    
                    if (x1<xmin) or (x2>xmax) or (car_id in invalid_tracks):
                        if len(frames_matched) > 0: # TODO: this threshold could be a ratio
                            newid = frames_matched["ID"].iloc[0] 
                            frames_matched["ID"] = newid #unify ID
                            newdf = pd.concat([newdf,frames_matched])
                        del tracks[car_id]
                        n_car -= 1
            
            if (m_box == 0) and (n_car == 0): # simply advance to the next frame
                continue
                
            elif (m_box == 0) and (n_car > 0): # if no measurements in current frame, simply predict
                x, tracks = self.predict_tracks_df(tracks)
                
            elif (m_box > 0) and (n_car == 0): # create new tracks (initialize)
                for i, row in frame.iterrows():
                    row = frame.loc[i:i,:]
                    tracks[row['ID'].iloc[0]] = row
            
            else: 
                x, tracks = self.predict_tracks_df(tracks)
                n_car = len(tracks)
                curr_id = list(tracks.keys()) # should be n id's 
    
                # score = np.ones([m_box,n_car])*(99)
                score_dist = np.zeros([m_box,n_car])
                score_iou =np.zeros([m_box,n_car])
                
                invalid_meas = set()
                invalid_tracks = set()
                for m in range(m_box):
                    for n in range(n_car):
                        score_dist[m,n] = self.dist_score(x[n],y[m],'maha')
                        score_iou[m,n], areaa, areab = self.iou(x[n],y[m],DIRECTION=False,AREA=True)
                        if areaa < 0.5:
                            invalid_tracks.add(n)
                        if areab < 1:
                            invalid_meas.add(m)
     
                # if (1715<k<1760):  
                #     vis.plot_track(np.array(np.vstack(x), dtype=float), np.array(y,dtype=float), curr_id, frame["ID"].values, xmin,xmax, k)
    
                # bipartite matching
                a,b = scipy.optimize.linear_sum_assignment(score_dist)
                
                gate = np.logical_or(score_dist<THRESHOLD_1, score_iou>0)
                matched_m = set()
                for i in range(len(a)):
                    if gate[a[i]][b[i]]:
                        n,m = b[i], a[i]
                        tracks[curr_id[n]] = tracks[curr_id[n]].reset_index(drop=True)
                        avg_meas = frame.loc[m:m]
                        
                        tracks[curr_id[n]].drop(tracks[curr_id[n]].tail(1).index,inplace=True) # drop the last row (prediction)
                        tracks[curr_id[n]] = pd.concat([tracks[curr_id[n]], avg_meas],ignore_index=True)  
                        
                        matched_m.add(m)
                # m_unassociated = np.where(np.sum(gate, axis=1)==0)[0]
                m_unassociated = set(np.arange(m_box))-matched_m
                for m in m_unassociated:
                    # !TODO: make sure that y[m] at not in the gate of each other
                    if (m not in invalid_meas) and (all(gate[m,:])==False) :
                        new_id = frame['ID'].iloc[m]
                        new_meas = frame.loc[m:m]
                        tracks[new_id] = new_meas
        self.df = newdf
        return
    
    def stitch_objects_jpda(self, THRESHOLD_1, THRESHOLD_2):
        '''
        10/20/2021
        use JPDA, weighted average of all meas that fall into a gate (defined by IOU and mahalanobis distance)
        create new ID for meas out side of the gate
        '''
        
        # define the x,y range to keep track of cars in FOV (meter)
        xmin, xmax = min(self.df["x"].values)-10,max(self.df["x"].values)+10
        
        ns = int(np.amin(np.array(self.df[['Frame #']]))) # start frame
        nf = int(np.amax(np.array(self.df[['Frame #']]))) # end frame
        tracks = dict() # a dictionary to store all current objects in view. key:ID, value:dataframe
        pts = ['bbr_x','bbr_y','fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
        pts_img = ["fbrx","fbry","fblx","fbly",	"bbrx",	"bbry",	"bblx",	"bbly",	"ftrx",	"ftry",	"ftlx",	"ftly",	"btrx",	"btry",	"btlx",	"btly"]
        newdf = pd.DataFrame()
        
        for k in range(ns,nf):
            print("\rFrame {}/{}".format(k,nf),end = "\r",flush = True)
            
            frame = self.df.loc[(self.df['Frame #'] == k)] # TODO: use groupby frame to save time
            y = np.array(frame[pts])
            notnan = ~np.isnan(y).any(axis=1)
            y = y[notnan] # remove rows with missing values (dim = mx8)
            frame = frame.iloc[notnan,:]
            frame = frame.reset_index(drop=True)
            
            m_box = len(frame)
            n_car = len(tracks)
            
            if (n_car > 0): # delete track that are out of view
                for car_id in list(tracks.keys()):
                    # delete track if total matched frames < 
                    last_frame = tracks[car_id].iloc[-1]
                    last_frame_x = np.array(last_frame[pts])[[0,2,4,6]]
                    x1,x2 = min(last_frame_x),max(last_frame_x)
                    frames = tracks[car_id]["Frame #"].values
                    matched_bool = ~np.isnan(frames)
                    frames_matched = tracks[car_id].loc[matched_bool]
    
                    if (x1<xmin) or (x2>xmax):
                        if len(frames_matched) > 0: # TODO: this threshold could be a ratio
                            newid = frames_matched["ID"].iloc[0] 
                            frames_matched["ID"] = newid #unify ID
                            newdf = pd.concat([newdf,frames_matched])
                        del tracks[car_id]
                        n_car -= 1
            
            if (m_box == 0) and (n_car == 0): # simply advance to the next frame
                continue
                
            elif (m_box == 0) and (n_car > 0): # if no measurements in current frame, simply predict
                x, tracks = self.predict_tracks_df(tracks)
                
            elif (m_box > 0) and (n_car == 0): # create new tracks (initialize)
                for i, row in frame.iterrows():
                    row = frame.loc[i:i,:]
                    tracks[row['ID'].iloc[0]] = row
            
            else: 
                x, tracks = self.predict_tracks_df(tracks)
                n_car = len(tracks)
                curr_id = list(tracks.keys()) # should be n id's 
    
                # score = np.ones([m_box,n_car])*(99)
                score_dist = np.zeros([m_box,n_car])
                score_iou =np.zeros([m_box,n_car])
                
                invalid_meas = []
                for m in range(m_box):
                    for n in range(n_car):
                        score_dist[m,n] = self.dist_score(x[n],y[m],'maha')
                        score_iou[m,n], areaa, areab = self.iou(x[n],y[m],DIRECTION=False,AREA=True)
    
                    if areab < 2: # invalid measurement
                        score_dist[m,:] = 99
                        score_iou[m,:] = -1
                        invalid_meas.append(m)
     
                if (1260<k<1300):  
                    vis.plot_track(np.array(np.vstack(x), dtype=float), np.array(y,dtype=float), curr_id, frame["ID"].values, xmin,xmax, k)
                # if k == 409:
                #     print("")
                # matching
                gate = np.logical_or(score_dist<THRESHOLD_1, score_iou>0)
                for n in range(n_car):
                    if any(gate[:,n]):
                        # calculate weighted average
                        tracks[curr_id[n]] = tracks[curr_id[n]].reset_index(drop=True)
                        frames_in_gate = frame.iloc[gate[:,n]]
                        if len(frames_in_gate) == 1:
                            avg_meas = frames_in_gate
                        else:
                            w = 1/score_dist[gate[:,n],n]
                            w = w / w.sum(axis=0)
                            frame_vals = np.array(frames_in_gate[pts_img+pts])
                            avg_meas_vals = np.reshape(np.dot(w,frame_vals),(1,-1))
                            avg_meas = pd.DataFrame(data=avg_meas_vals,  columns=pts_img + pts) 
                            avg_meas["Frame #"] = k
                        tracks[curr_id[n]].drop(tracks[curr_id[n]].tail(1).index,inplace=True) # drop the last row (prediction)
                        tracks[curr_id[n]] = pd.concat([tracks[curr_id[n]], avg_meas],ignore_index=True)  
                        
                m_unassociated = np.where(np.sum(gate, axis=1)==0)[0]
    
                for m in m_unassociated:
                    # !TODO: make sure that y[m] at not in the gate of each other
                    if m not in invalid_meas:
                        new_id = frame['ID'].iloc[m]
                        new_meas = frame.loc[m:m]
                        tracks[new_id] = new_meas  
        self.df = newdf
        return

    def pts_to_line(self,x,y,a,b,c):
        '''
        distance from (x,y) to a line defined by ax+by+c=0
        x,y are vectors of the same length
        a,b,c, are scalars
        return the average distances
        '''
        d = np.abs(a*x + b*y + c)/np.sqrt(a**2+b**2)
        return np.nanmean(d)

    def stitch_objects_tsm(self, THRESHOLD_1, THRESHOLD_2):
        '''
        try to match tracks that are "close" to each other in time-space dimension
        offline matching, take advantage of a linear model
        '''
        groups = self.df.groupby("ID")
        groupList = list(groups.groups)
        pts = ['bbr_x','bbr_y','fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
       
        n_car = len(groupList)
        DX = np.ones((n_car, n_car)) * 99 
        DY = np.ones((n_car, n_car)) * 99
        
        for i,c1 in enumerate(groupList):
            # get the fitted line for c1
            track1 = groups.get_group(c1)
            t1 = track1["Frame #"].values
            track1 = np.array(track1[pts])
            x1 = (track1[:,0] + track1[:,2])/2
            y1 = np.nanmean((track1[:,1] + track1[:,7])/2)
            cx1 = np.nanmean(x1)
     
            fit = np.polyfit(t1,x1,1)
            a,b,c = -fit[0], 1, -fit[1] # x1 = a*t1 + c
            
            for j in range(n_car):
                if i == j:
                    continue
                track2 = groups.get_group(groupList[j])
                t2 = track2["Frame #"].values
                track2 = np.array(track2[pts])
                x2 = (track2[:,0] + track2[:,2])/2
                y2 = np.nanmean((track2[:,1] + track2[:,7])/2)
                cx2 = np.nanmean(x2)
                DX[i,j] = self.pts_to_line(t2,x2,a,b,c)
                DY[i,j] = np.abs(y1-y2)/np.abs(cx1-cx2) # tan\theta
                
        matched = np.logical_and(DX<THRESHOLD_1, DY<THRESHOLD_2)
    
        for i in range(n_car-1,-1,-1):
            idx = np.where(np.logical_and(matched[:,i], matched[i,:]))[0]
            if len(idx)>0:
                newid = groupList[idx[0]] # assigned to the first ID appeared in scene
                parent = {groupList[i]: newid for i in list(idx)+[i]}
                self.df['ID'] = self.df['ID'].apply(lambda x: parent[x] if x in parent else x)
                
        # post process
        return
    
    def stitch_objects_tsmn(self, THRESHOLD_1=0.3, THRESHOLD_2=0.03):
        '''
        try to match tracks that are "close" to each other in time-space dimension
        for each track define a "cone" based on the tolerable acceleration
        match tracks that fall into the cone of other tracks in both x and y
        
        THRESHOLD_1: allowable acceleration/deceleration (m/s2)
        THRESHOLD_2: allowable steering angle (rad)
        '''
        groups = self.df.groupby("ID")
        groupList = list(groups.groups)
        pts = ['bbr_x','bbr_y','fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
       
        n_car = len(groupList)
        CX = np.zeros((n_car, n_car)) # cone of X
        CY = np.zeros((n_car, n_car)) # 
        
        for i,c1 in enumerate(groupList):
            # get the fitted line for c1
            track1 = groups.get_group(c1)
            t1 = track1["Frame #"].values
            ct1 = np.nanmean(t1)
            track1 = np.array(track1[pts])
            x1 = (track1[:,0] + track1[:,2])/2
            y1 = (track1[:,1] + track1[:,7])/2
            cx1, cy1 = np.nanmean(x1), np.nanmean(y1)
     
            # fit time vs. x
            fit = np.polyfit(t1,x1,1)
            vx = fit[0]
            bp = cx1-(vx+THRESHOLD_1)*ct1 # recalculate y-intercept
            bpp = cx1-(vx-THRESHOLD_1)*ct1
            
            # fit time vs. y
            fit = np.polyfit(t1,y1,1)
            vy = fit[0]
            cp = cy1-(vy+THRESHOLD_2)*ct1 # recalculate y-intercept
            cpp = cy1-(vy-THRESHOLD_2)*ct1
            
            for j in range(n_car):
                if i == j:
                    continue
                track2 = groups.get_group(groupList[j])
                t2 = track2["Frame #"].values
                track2 = np.array(track2[pts])
                x2 = (track2[:,0] + track2[:,2])/2
                y2 = (track2[:,1] + track2[:,7])/2

                if (all(x2 < t2*(vx+THRESHOLD_1)+bp) and all(x2 > t2*(vx-THRESHOLD_1)+bpp)) or (all(x2 > t2*(vx+THRESHOLD_1)+bp) and all(x2 < t2*(vx-THRESHOLD_1)+bpp)):
                    CX[i,j] = 1
                if (all(y2 < t2*(vy+THRESHOLD_2)+cp) and all(y2 > t2*(vy-THRESHOLD_2)+cpp)) or (all(y2 > t2*(vy+THRESHOLD_2)+cp) and all(y2 < t2*(vy-THRESHOLD_2)+cpp)):
                    CY[i,j] = 1
                # CY[i,j] = max(np.abs(y1[[0,-1, 0, -1]]-y2[[0,-1, -1, 0]])/np.abs(x1[[0,-1, 0 ,-1]]-x2[[0,-1,-1,0]]) ) # tan\theta
            
        CX = CX == 1
        CY = CY == 1
        for i in range(len(CX)):
            CX[i,:] = np.logical_and(CX[i,:], CX[:,i])
            CY[i,:] = np.logical_and(CY[i,:], CY[:,i])
            
        matched = np.logical_and(CX,CY)
    
        for i in range(n_car-1,-1,-1):
            idx = np.where(matched[:,i])[0]
            if len(idx)>0:
                newid = groupList[idx[0]] # assigned to the first ID appeared in scene
                parent = {groupList[i]: newid for i in list(idx)+[i]}
                self.df['ID'] = self.df['ID'].apply(lambda x: parent[x] if x in parent else x)
                
        # post process
        # df = applyParallel(df.groupby("Frame #"), del_repeat_meas_per_frame).reset_index(drop=True)
        return
    
    def iou_ts(self,a,b):
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
    
        p = Polygon([(a[0,2*i],a[0,2*i+1]) for i in range(4)])
        q = Polygon([(b[0,2*i],b[0,2*i+1]) for i in range(4)])

        intersection_area = p.intersection(q).area
        union_area = min(p.area, q.area)
        iou = float(intersection_area/union_area)
                
        return iou
        
    def remove_invalid(self):
        '''
        valid: length covers more than 50% of the FOV
        invalid: length covers less than 10% of FOV, or
                crashes with any valid tracks
        undetermined: tracks that are short but not overlaps with any valid tracks
        '''
        
        xmin, xmax = min(self.df["x"].values),max(self.df["x"].values)
        groups = self.df.groupby("ID")
        groupList = list(groups.groups)
        pts = ['bbr_x','bbr_y','fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
        
        valid = {}
        invalid = set()
        for carid, group in groups:
            if (max(group.x.values)-min(group.x.values)>0.5*(xmax-xmin)): # long tracks
                frames = group["Frame #"].values
                first = group.head(1)
                last = group.tail(1)
                x0, x1 = max(first.bbr_x.values[0],first.fbr_x.values[0]),min(first.bbr_x.values[0],first.fbr_x.values[0])
                x2, x3 = min(last.bbr_x.values[0],last.fbr_x.values[0]),max(last.bbr_x.values[0],last.fbr_x.values[0])
                y0, y1 = max(first.bbr_y.values[0],first.bbl_y.values[0]),min(first.bbr_y.values[0],first.bbl_y.values[0])
                y2, y3 = min(last.bbr_y.values[0],last.bbl_y.values[0]),max(last.bbr_y.values[0],last.bbl_y.values[0])
                t0,t1 = min(frames), max(frames)
                valid[carid] = [np.array([t0,x0,t0,x1,t1,x2,t1,x3]),np.array([t0,y0,t0,y1,t1,y2,t1,y3])]
            elif (max(group.x.values)-min(group.x.values)<0.1*(xmax-xmin)): # short tracks
                invalid.add(carid)
                
        # print(valid.keys())
        for carid, group in groups:
            if (carid not in valid.keys()) and (carid not in invalid):
                frames = group["Frame #"].values
                first = group.head(1)
                last = group.tail(1)
                x0, x1 = max(first.bbr_x.values[0],first.fbr_x.values[0]),min(first.bbr_x.values[0],first.fbr_x.values[0])
                x2, x3 = min(last.bbr_x.values[0],last.fbr_x.values[0]),max(last.bbr_x.values[0],last.fbr_x.values[0])
                y0, y1 = max(first.bbr_y.values[0],first.bbl_y.values[0]),min(first.bbr_y.values[0],first.bbl_y.values[0])
                y2, y3 = min(last.bbr_y.values[0],last.bbl_y.values[0]),max(last.bbr_y.values[0],last.bbl_y.values[0])
                t0,t1 = min(frames), max(frames)
                
                bx = np.array([t0,x0,t0,x1,t1,x2,t1,x3])
                by = np.array([t0,y0,t0,y1,t1,y2,t1,y3])
                for valid_id in valid:
                    ax,ay = valid[valid_id]
                    ioux = self.iou_ts(ax,bx)
                    iouy = self.iou_ts(ay,by)
                    if ioux > 0 and iouy > 0: # trajectory overlaps with a valid track
                        invalid.add(carid)
        
        print("Undetermined IDs: ",set(groupList)-valid.keys()-invalid)
        self.df = self.df.groupby("ID").filter(lambda x: (x['ID'].iloc[0] not in invalid)).reset_index(drop=True)
        return
    
    def visualize_BA(self, lanes=[1,2,3,4,7,8,9,10]):
        
        for lane_idx in lanes:
            fig, axs = plt.subplots(1,2, figsize=(15,5), facecolor='w', edgecolor='k')
            axs = axs.ravel()
            vis.plot_time_space(self.original, lanes=[lane_idx], time="frame", space="x", ax=axs[0])
            vis.plot_time_space(self.df, lanes=[lane_idx], time="frame", space="x", ax=axs[1])
            fig.tight_layout()
    
if __name__ == "__main__":

    data_path = r"E:\I24-postprocess\MC_tracking" 
    file_path = data_path+r"\MC_reinterpolated.csv"
    
    params = {"method": "gnn",
              "start": 900,
              "end": 1500,
              "lanes": [],
              "plot_start": 0, # for plotting tracks in online methods
              "plot_end": 10,
              "preprocess": False
              }
    
    da = Data_Association(file_path, params)
    # da.visualize(lanes = [1,3,4])
    da.stitch_objects_tsmn(0.3,0.1)
    da.remove_invalid()
    da.visualize_BA()
    
    