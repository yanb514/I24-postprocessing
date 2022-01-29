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
# from shapely.geometry import Polygon
import pandas as pd
import utils_vis as vis
# import scipy
import matplotlib.pyplot as plt
import utils_evaluation as ev
# import torch
import itertools
import multiprocessing

# import cProfile
# import pstats
# import io
# from line_profiler import LineProfiler

# import warnings
# warnings.filterwarnings("error")

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
        # if len(params["lanes"]) > 0:
        #     self.df = self.df[self.df["lane"].isin(params["lanes"])]
        self.original = self.df.copy()
        self.data = {}
    
    def dist_score(self, B, B_data, DIST_MEAS='maha', DIRECTION=True):
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
            alpha = (1/1)**2
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
        #     # direction = track_df["direction"].iloc[0]
            # direction = np.sign(track_df["fbr_x"].iloc[-1]-track_df["bbr_x"].iloc[-1])
            track = np.array(track_df[pts])
            direction = np.sign(track[-1][2]-track[-1][0])
            
        #     if len(track)>1:   # average speed
    
        #         frames = np.arange(0,len(track))
        #         fit = np.polyfit(frames,track,1)
        #         est_speed = np.mean(fit[0,[0,2,4,6]])
        #         x_pred = np.polyval(fit, len(track))
        #         if abs(est_speed)<mpf/2 or (np.sign(est_speed)!=direction) or (abs(x_pred[0]-x_pred[2])<1): # too slow
        #             x_pred = track[-1,:] + direction* np.array([mpf,0,mpf,0,mpf,0,mpf,0])
    
        #     else:
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
                # score_dist[score_dist>THRESHOLD_1]=np.inf
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
        print("\n")
        print("Before DA: {} unique IDs".format(self.df.groupby("ID").ngroups)) 
        print("After DA: {} unique IDs".format(newdf.groupby("ID").ngroups))
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
        track_len = []
        for i,c1 in enumerate(groupList):
            print("\rTracklet {}/{}".format(i,n_car),end = "\r",flush = True)
            # get the fitted line for c1
            track1 = groups.get_group(c1)
            track_len.append(len(track1))
            if len(track1)<2:
                fit = None
            else:
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
                if (i == j) or (fit is None):
                    continue
                track2 = groups.get_group(groupList[j])
                if len(track2)<2:
                    continue
                else:
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
        for i in range(len(CX)): # make CX CY symmetrical
            CX[i,:] = np.logical_and(CX[i,:], CX[:,i])
            CY[i,:] = np.logical_and(CY[i,:], CY[:,i])
            
        match = np.logical_and(CX,CY) # symmetrical
    
        # assign matches: note this this is NP-complete (see cliques subgraph problem)
        # 1: not consider conflicts
        # for i in range(n_car-1,-1,-1):
        #     idx = np.where(match[:,i])[0]
        #     if len(idx)>0:
        #         newid = groupList[idx[0]] # assigned to the first ID appeared in scene
        #         parent = {groupList[i]: newid for i in list(idx)+[i]}
        #         self.df['ID'] = self.df['ID'].apply(lambda x: parent[x] if x in parent else x)
        
        # 2: sort by length of tracks descending
        sort_len = sorted(range(len(track_len)), key=lambda k: track_len[k])[::-1]
        matched = set()
        for i in sort_len:
            idx = np.where(match[:,i])[0]
            if len(idx)>0:
                parent = {groupList[ii]: groupList[i] for ii in idx if (ii not in matched) and (i not in matched)}
                self.df['ID'] = self.df['ID'].apply(lambda x: parent[x] if x in parent else x)
            matched = matched.union(set(idx))
        print("Before DA: {} unique IDs".format(len(groupList))) 
        print("After DA: {} unique IDs".format(self.df.groupby("ID").ngroups))
      
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
     
    def log_likelihood(self, input, target, var):
        '''
        https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
        input: N x 1
        target: N x 1
        var: variance
        '''
        N = len(target)
        var = np.maximum(1e-6*np.ones(var.shape),var)
        return 1/(2*N)*np.sum(np.log(var)+(input-target)**2/var)
        
        
   
    def stitch_objects_tsmn_ll(self, THRESHOLD_C=50, VARX=0.03, VARY=0.03, time_out = 500):
        '''
        similar to stitch_objects_tsmn
        instead of binary fit, calculate a loss (log likelihood)
        '''
 
        groups = {k: v for k, v in self.df.groupby("ID")}
        groupList = list(groups.keys())
        # groupList = list(groups.groups)
        pts = ['bbr_x','bbr_y','fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
       
        n_car = len(groupList)
        CX = np.ones((n_car, n_car)) * 999 # cone of X

        loss = torch.nn.GaussianNLLLoss()
        empty_id = set()
        
        
        for i,c1 in enumerate(groupList):
            print("\rTracklet {}/{}".format(i,n_car),end = "\r",flush = True)
            # get the fitted line for c1
            # track1 = groups.get_group(c1)
            track1 = groups[c1]

            t1 = track1["Frame #"].values
            ct1 = np.nanmean(t1)
            track1 = np.array(track1[pts])
            x1 = (track1[:,0] + track1[:,2])/2
            y1 = (track1[:,1] + track1[:,7])/2
            notnan = ~np.isnan(x1)
            t1 = t1[notnan]
            x1 = x1[notnan]
            y1 = y1[notnan]
            
            if len(t1)<1 or (c1 in empty_id):
                empty_id.add(c1)
                continue
            
            elif len(t1)<2:
                v = np.sign(track1[0,2]-track1[0,0]) # assume 1/-1 m/frame = 30m/s
                b = x1-v*ct1 # recalculate y-intercept
                fitx = np.array([v,b[0]])
                fity = np.array([0,y1[0]])
            else:
                X = np.vstack([t1,np.ones(len(t1))]).T # N x 2
                fitx = np.linalg.lstsq(X, x1, rcond=None)[0]
                fity = np.linalg.lstsq(X, y1, rcond=None)[0]
                
                    
            for j in range(i+1,n_car):
                # track2 = groups.get_group(groupList[j])
                track2 = groups[groupList[j]]
                
                t2 = track2["Frame #"].values
                track2 = np.array(track2[pts])
                x2 = (track2[:,0] + track2[:,2])/2
                y2 = (track2[:,1] + track2[:,7])/2

                notnan = ~np.isnan(x2)
                if sum(notnan)==0 or (groupList[j] in empty_id): # if all data is nan (missing)
                    empty_id.add(groupList[j])
                    continue
                t2 = t2[notnan]
                x2 = x2[notnan]
                y2 = y2[notnan]
                ct2 = np.mean(t2)
            
                if len(t2)<2:
                    v = np.sign(track2[0,2]-track2[0,0])
                    b = x2-v*ct2 # recalculate y-intercept
                    fit2x = np.array([v,b[0]])
                    fit2y = np.array([0,y2[0]])
                else:
                    # OLS faster
                    X = np.vstack([t2,np.ones(len(t2))]).T
                    fit2x = np.linalg.lstsq(X, x2, rcond=None)[0]
                    fit2y = np.linalg.lstsq(X, y2, rcond=None)[0] 
                
                nll = 999
                if all(t2-t1[-1]>=0): # t1 comes first
                    if t2[0] - t1[-1] > time_out:
                        # print("time out {} and {}".format(c1, groupList[j]))
                        continue
                    # 1. project t1 forward to t2's time
                    # targetx = np.polyval(fitx, t2)
                    # targety = np.polyval(fity, t2)
                    targetx = np.matmul(X, fitx)
                    targety = np.matmul(X, fity)
                    pt1 = t1[-1]
                    varx = (t2-pt1) * VARX 
                    vary = (t2-pt1) * VARY
                    input = torch.transpose(torch.tensor([x2,y2]),0,1)
                    target = torch.transpose(torch.tensor([targetx, targety]),0,1)
                    var = torch.transpose(torch.tensor([varx,vary]),0,1)
                    nll = min(nll, loss(input,target,var))
                    # print("{} and {}: {}".format(c1, groupList[j],nll))
                    # 2. project t2 backward to t1's time
                    targetx = np.polyval(fit2x, t1)
                    targety = np.polyval(fit2y, t1)
                    pt2 = t2[0]
                    varx = (pt2-t1) * VARX 
                    vary = (pt2-t1) * VARY
                    input = torch.transpose(torch.tensor([x1,y1]),0,1)
                    target = torch.transpose(torch.tensor([targetx, targety]),0,1)
                    var = torch.transpose(torch.tensor([varx,vary]),0,1)
                    nll = min(nll, loss(input,target,var))
                    # print("{} and {}: {}".format(c1, groupList[j],nll))
                    
                elif all(t1-t2[-1]>=0): # t2 comes first:
                    if t1[0] - t2[-1] > time_out:
                        continue
                    # 3. project t1 backward to t2's time
                    targetx = np.polyval(fitx, t2)
                    targety = np.polyval(fity, t2)
                    pt1 = t1[0]
                    varx = (pt1-t2) * VARX 
                    vary = (pt1-t2) * VARY
                    input = torch.transpose(torch.tensor([x2,y2]),0,1)
                    target = torch.transpose(torch.tensor([targetx, targety]),0,1)
                    var = torch.transpose(torch.tensor([varx,vary]),0,1)
                    nll = min(nll, loss(input,target,var))
                    # print("{} and {}: {}".format(c1, groupList[j],nll))
                    
                    # 4. project t2 forward to t1's time
                    targetx = np.polyval(fit2x, t1)
                    targety = np.polyval(fit2y, t1)
                    pt2 = t2[-1]
                    varx = (t1-pt2) * VARX 
                    vary = (t1-pt2) * VARY
                    input = torch.transpose(torch.tensor([x1,y1]),0,1)
                    target = torch.transpose(torch.tensor([targetx, targety]),0,1)
                    var = torch.transpose(torch.tensor([varx,vary]),0,1)
                    nll = min(nll, loss(input,target,var))
                    # print("{} and {}: {}".format(c1, groupList[j],nll))
                    
                # else:
                    # print("time overlap {} and {}".format(c1, groupList[j]))
                CX[i,j] = nll
                
        # for debugging only
        self.CX = CX
        self.groupList = groupList
        self.empty_id = empty_id
        
        BX = CX < THRESHOLD_C
        
        for i in range(len(CX)): # make CX CY symmetrical
            BX[i,:] = np.logical_or(BX[i,:], BX[:,i])
            
        # 4. start by sorting CX
        a,b = np.unravel_index(np.argsort(CX, axis=None), CX.shape)

        path = {idx: {idx} for idx in range(n_car)} # disjoint set
        for i in range(len(a)):
            if CX[a[i],b[i]] > THRESHOLD_C:
                break
            else: 
                path_a, path_b = list(path[a[i]]),list(path[b[i]])
                if np.all(BX[np.ix_(path_a, path_b)]): # if no conflict with any path
                    path[a[i]] = path[a[i]].union(path[b[i]])
                    for aa in path[a[i]]:
                        path[aa] = path[a[i]].copy()
                    
        # delete IDs that are empty
        self.df = self.df.groupby("ID").filter(lambda x: (x["ID"].iloc[0] not in empty_id))
        self.path = path.copy()
        # modify ID
        while path:
            key = list(path.keys())[0]
            reid = {groupList[old]: groupList[key] for old in path[key]} # change id=groupList[v] to groupList[key]
            self.df = self.df.replace({'ID': reid})
            for v in list(path[key]) + [key]:
                try:
                    path.pop(v)
                except KeyError:
                    pass

            
        self.df = self.df.sort_values(by=['Frame #','ID']).reset_index(drop=True)         
        print("\n")
        print("Before DA: {} unique IDs".format(len(groupList))) 
        print("After DA: {} unique IDs".format(self.df.groupby("ID").ngroups))
        print("True: {} unique IDs".format(len([id for id in groupList if id<1000])))
      
        return 
    
    def stitch_objects_tsmn_ll_mp(self, THRESHOLD_C=50, VARX=0.03, VARY=0.03, time_out = 500):
        '''
        multiple pool for stitch_objects_tsmn_ll
        '''
 
        groups = {k: v for k, v in self.df.groupby("ID")}
        groupList = list(groups.keys())
        # groupList = list(groups.groups)
        pts = ['bbr_x','bbr_y','fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
       
        n_car = len(groupList)
        CX = np.ones((n_car, n_car)) * 999 # cone of X

        loss = torch.nn.GaussianNLLLoss()
        empty_id = set()
        
        combinations = itertools.combinations(groupList, 2)
        for i,c1 in enumerate(groupList):
            print("\rTracklet {}/{}".format(i,n_car),end = "\r",flush = True)
            # get the fitted line for c1
            # track1 = groups.get_group(c1)
            track1 = groups[c1]

            t1 = track1["Frame #"].values
            ct1 = np.nanmean(t1)
            track1 = np.array(track1[pts])
            x1 = (track1[:,0] + track1[:,2])/2
            y1 = (track1[:,1] + track1[:,7])/2
            notnan = ~np.isnan(x1)
            t1 = t1[notnan]
            x1 = x1[notnan]
            y1 = y1[notnan]
            
            if len(t1)<1 or (c1 in empty_id):
                empty_id.add(c1)
                continue
            
            elif len(t1)<2:
                v = np.sign(track1[0,2]-track1[0,0]) # assume 1/-1 m/frame = 30m/s
                b = x1-v*ct1 # recalculate y-intercept
                fitx = np.array([v,b[0]])
                fity = np.array([0,y1[0]])
            else:
                X = np.vstack([t1,np.ones(len(t1))]).T # N x 2
                fitx = np.linalg.lstsq(X, x1, rcond=None)[0]
                fity = np.linalg.lstsq(X, y1, rcond=None)[0]
                
                    
            for j in range(i+1,n_car):
                # track2 = groups.get_group(groupList[j])
                track2 = groups[groupList[j]]
                
                t2 = track2["Frame #"].values
                track2 = np.array(track2[pts])
                x2 = (track2[:,0] + track2[:,2])/2
                y2 = (track2[:,1] + track2[:,7])/2

                notnan = ~np.isnan(x2)
                if sum(notnan)==0 or (groupList[j] in empty_id): # if all data is nan (missing)
                    empty_id.add(groupList[j])
                    continue
                t2 = t2[notnan]
                x2 = x2[notnan]
                y2 = y2[notnan]
                ct2 = np.mean(t2)
            
                if len(t2)<2:
                    v = np.sign(track2[0,2]-track2[0,0])
                    b = x2-v*ct2 # recalculate y-intercept
                    fit2x = np.array([v,b[0]])
                    fit2y = np.array([0,y2[0]])
                else:
                    # OLS faster
                    X = np.vstack([t2,np.ones(len(t2))]).T
                    fit2x = np.linalg.lstsq(X, x2, rcond=None)[0]
                    fit2y = np.linalg.lstsq(X, y2, rcond=None)[0] 
                
                nll = 999
                if all(t2-t1[-1]>=0): # t1 comes first
                    if t2[0] - t1[-1] > time_out:
                        # print("time out {} and {}".format(c1, groupList[j]))
                        continue
                    # 1. project t1 forward to t2's time
                    # targetx = np.polyval(fitx, t2)
                    # targety = np.polyval(fity, t2)
                    targetx = np.matmul(X, fitx)
                    targety = np.matmul(X, fity)
                    pt1 = t1[-1]
                    varx = (t2-pt1) * VARX 
                    vary = (t2-pt1) * VARY
                    input = torch.transpose(torch.tensor([x2,y2]),0,1)
                    target = torch.transpose(torch.tensor([targetx, targety]),0,1)
                    var = torch.transpose(torch.tensor([varx,vary]),0,1)
                    nll = min(nll, loss(input,target,var))
                    # print("{} and {}: {}".format(c1, groupList[j],nll))
                    # 2. project t2 backward to t1's time
                    targetx = np.polyval(fit2x, t1)
                    targety = np.polyval(fit2y, t1)
                    pt2 = t2[0]
                    varx = (pt2-t1) * VARX 
                    vary = (pt2-t1) * VARY
                    input = torch.transpose(torch.tensor([x1,y1]),0,1)
                    target = torch.transpose(torch.tensor([targetx, targety]),0,1)
                    var = torch.transpose(torch.tensor([varx,vary]),0,1)
                    nll = min(nll, loss(input,target,var))
                    # print("{} and {}: {}".format(c1, groupList[j],nll))
                    
                elif all(t1-t2[-1]>=0): # t2 comes first:
                    if t1[0] - t2[-1] > time_out:
                        continue
                    # 3. project t1 backward to t2's time
                    targetx = np.polyval(fitx, t2)
                    targety = np.polyval(fity, t2)
                    pt1 = t1[0]
                    varx = (pt1-t2) * VARX 
                    vary = (pt1-t2) * VARY
                    input = torch.transpose(torch.tensor([x2,y2]),0,1)
                    target = torch.transpose(torch.tensor([targetx, targety]),0,1)
                    var = torch.transpose(torch.tensor([varx,vary]),0,1)
                    nll = min(nll, loss(input,target,var))
                    # print("{} and {}: {}".format(c1, groupList[j],nll))
                    
                    # 4. project t2 forward to t1's time
                    targetx = np.polyval(fit2x, t1)
                    targety = np.polyval(fit2y, t1)
                    pt2 = t2[-1]
                    varx = (t1-pt2) * VARX 
                    vary = (t1-pt2) * VARY
                    input = torch.transpose(torch.tensor([x1,y1]),0,1)
                    target = torch.transpose(torch.tensor([targetx, targety]),0,1)
                    var = torch.transpose(torch.tensor([varx,vary]),0,1)
                    nll = min(nll, loss(input,target,var))
                    # print("{} and {}: {}".format(c1, groupList[j],nll))
                    
                # else:
                    # print("time overlap {} and {}".format(c1, groupList[j]))
                CX[i,j] = nll
                
        # for debugging only
        self.CX = CX
        self.groupList = groupList
        self.empty_id = empty_id
        
        BX = CX < THRESHOLD_C
        
        for i in range(len(CX)): # make CX CY symmetrical
            BX[i,:] = np.logical_or(BX[i,:], BX[:,i])
            
        # 4. start by sorting CX
        a,b = np.unravel_index(np.argsort(CX, axis=None), CX.shape)

        path = {idx: {idx} for idx in range(n_car)} # disjoint set
        for i in range(len(a)):
            if CX[a[i],b[i]] > THRESHOLD_C:
                break
            else: 
                path_a, path_b = list(path[a[i]]),list(path[b[i]])
                if np.all(BX[np.ix_(path_a, path_b)]): # if no conflict with any path
                    path[a[i]] = path[a[i]].union(path[b[i]])
                    for aa in path[a[i]]:
                        path[aa] = path[a[i]].copy()
                    
        # delete IDs that are empty
        self.df = self.df.groupby("ID").filter(lambda x: (x["ID"].iloc[0] not in empty_id))
        self.path = path.copy()
        # modify ID
        while path:
            key = list(path.keys())[0]
            reid = {groupList[old]: groupList[key] for old in path[key]} # change id=groupList[v] to groupList[key]
            self.df = self.df.replace({'ID': reid})
            for v in list(path[key]) + [key]:
                try:
                    path.pop(v)
                except KeyError:
                    pass

            
        self.df = self.df.sort_values(by=['Frame #','ID']).reset_index(drop=True)         
        print("\n")
        print("Before DA: {} unique IDs".format(len(groupList))) 
        print("After DA: {} unique IDs".format(self.df.groupby("ID").ngroups))
        print("True: {} unique IDs".format(len([id for id in groupList if id<1000])))
      
        return 
    
    def average_meas(self, x):
        if len(x)>2:
            print("Found {} measurements for ID {} Frame {}".format(len(x),x["ID"].iloc[0],x["Frame #"].iloc[0]))
        mean = x.head(1)
        pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
        Y = np.array(x[pts])
        Y = np.nanmean(Y, axis=0)
        mean.loc[:,pts] = Y
        return mean
    
    def del_repeat_meas_per_frame(self, framesnap):
        framesnap = framesnap.groupby('ID').apply(self.average_meas)
        return framesnap
      
    def associate(self):
        method = self.params["method"]
        thresh1, thresh2 = self.params["threshold"]
            
        if "tsmn" in method:
            self.stitch_objects_tsmn_ll(thresh1, thresh2) # 0.3, 0.04 for tsmn
        elif "gnn" in method:
            self.stitch_objects_gnn(thresh1, thresh2)
        elif "bm" in method:
            self.stitch_objects_bm(thresh1, thresh2)
        elif "jpda" in method:
            self.stitch_objects_jpda(thresh1, thresh2)
        else:
            print("DA method not defined")
        return
    

    def evaluate(self, synth=True):
        '''
        synth = True: for synthetic experiments only
            compute fragments and ID switches
        get_invalid
        mark_outliers
        '''
        if synth:
            idpath = {}
            FRAG = {}
            IDS = {}
            visited = set()
            for idx, idx_v in self.path.items():
                trueID = self.groupList[idx]
                if trueID in self.empty_id or trueID in visited:
                    continue
                assID = [self.groupList[id] for id in idx_v]
                idpath[trueID] = assID
                if trueID >= 1000 and all([l>=1000 for l in assID]) and (trueID//1000 in idpath.keys()):
                    FRAG[trueID] = assID
                visited.add(trueID)
                visited = visited.union(set(assID))
                if trueID>=1000: trueID = trueID//1000
                if any([l//1000 != trueID and l!= trueID for l in assID]):
                    IDS[trueID] = assID
            self.idpath = idpath
            self.FRAG = FRAG
            self.IDS = IDS
            print("Fragments: {}".format(FRAG))
            print("ID switches: {}".format(IDS))
            return     
            
        print("{} tracklets are stitched".format(self.original.groupby("ID").ngroups-self.df.groupby("ID").ngroups))
        valid0,collision0 = ev.get_invalid(self.original, ratio=0.4)
        valid,collision = ev.get_invalid(self.df, ratio=0.4) # remove obvious invalid tracks from df
        print("{} more valid tracklets are created".format(len(valid)-len(valid0)))
        self.data["valid"] = valid
        self.data["collision"] = collision
        print("Valid tracklets: {}/{}".format(len(valid), self.df.groupby("ID").ngroups))
        print("Collision with valid: {}".format(collision))
        
        # check lane-change vehicles
        groups = self.df.groupby("ID")
        multiple_lane = set()
        for carid, group in groups:
            if group.lane.nunique()>1:
                # frames = group.groupby("Frame #")
                # for frame_id, frame in frames:
                if np.abs(np.max(group[["bbr_y","bbl_y"]].values)-np.min(group[["bbr_y","bbl_y"]].values)) > 12/3.281:
                    if carid in valid:
                        multiple_lane.add(carid)
                        # break
                    
        self.data["lane_change"] = multiple_lane
        print("Possible lane-change tracks:", multiple_lane)
        
        self.df = self.df.groupby("ID").apply(ev.mark_outliers_car).reset_index(drop=True)
        outlier_ratio = {carid: np.count_nonzero(car["Generation method"].values=="outlier")/car.bbr_x.count() for carid, car in self.df.groupby("ID")}
        self.data["outlier_ratio"] = outlier_ratio
        outlier_high = {key: value for key, value in outlier_ratio.items() if (value > self.params["outlier_thresh"]) and (key in valid)}  
        print("Outlier ratio above {}: {}".format(self.params["outlier_thresh"],outlier_high))
        vis.plot_histogram(np.fromiter(outlier_ratio.values(), dtype=float), bins=40,
                           labels="", 
                           xlabel = "Outlier ratio", 
                           ylabel = "Probability", 
                           title = "Outlier ratio distribution")
        return
    
    def postprocess(self, REMOVE_INVALID=True, REMOVE_OUTLIER=True, SELECT_ONE_MEAS=True, CONNECT_TRACKS=True,  SAVE=""):
        # count lane spaned for each ID
        
        if REMOVE_INVALID:
            print("Remove invalid tracks...")
            self.df = self.df.groupby("ID").filter(lambda x: (x['ID'].iloc[0] in self.data["valid"])).reset_index(drop=True)
        if REMOVE_OUTLIER:
            print("Remove outliers...")
            if "outlier_ratio" not in self.data.keys():
                print("mark outliers")
                self.df["Generation method"] = ""
                self.df = self.df.groupby("ID").apply(ev.mark_outliers_car).reset_index(drop=True)
            pts = ['bbr_x','bbr_y','fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
            self.df.loc[self.df["Generation method"]=="outlier", pts]=np.nan
        if SELECT_ONE_MEAS:
            print("Select one meas...")
            self.df = utils.applyParallel(self.df.groupby("Frame #"), self.del_repeat_meas_per_frame).reset_index(drop=True) # keep only one meas per frame per track
        if CONNECT_TRACKS:
            print("Connect invalid tracks...")
            self.df = self.df.groupby("ID").apply(utils.connect_track).reset_index(drop=True)     # connect tracks such that frame # are continuous
        
        # mark missing as missing in Generation method
        self.df.loc[self.df.bbr_x.isnull(), 'Generation method'] = 'missing'
        
        if len(SAVE)>0:
            self.df.to_csv(SAVE, index = False)
            
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

    raw_path = r"E:\I24-postprocess\benchmark\TM_1000_pollute.csv"
    
    params = {"method": "tsmn",
              "threshold": (0,0), # 0.3, 0.04 for tsmn
              "start": 1000, # starting frame
              "end": 2050, # ending frame
              "plot_start": 0, # for plotting tracks in online methods
              "plot_end": 10,
              "preprocess": False,
              "outlier_thresh": 0.25
              }
    
    da = Data_Association(raw_path, params)

    # da.df = da.df[da.df["ID"].isin([66,66008,66009])] # for debugging purposes
    da.stitch_objects_tsmn_ll(THRESHOLD_C=30, VARX = 0.05, VARY=0.02, time_out = 2000)
    
    # %% profile
    # pr = cProfile.Profile()
    # pr.enable()
    # da.stitch_objects_tsmn_ll(THRESHOLD_C=30, VARX = 0.05, VARY=0.02, time_out = 100)
    # pr.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    # ps.print_stats()
    
    # with open('test_OLS.txt', 'w+') as f:
    #     f.write(s.getvalue())
        
    
    # %%
    # da.stitch_objects_tsmn_ll(THRESHOLD_C=30, VARX = 0.05, VARY=0.02, time_out = 100)
    # da.df.to_csv(r"E:\I24-postprocess\benchmark\TM_300_pollute_DA.csv", index=False)
    da.evaluate(synth=True) # set to False if absence of GT
    # da.df = utils.read_data(r"E:\I24-postprocess\benchmark\TM_200_pollute_DA.csv")
    #%%
    da.postprocess(REMOVE_INVALID=False, 
                    REMOVE_OUTLIER=False,
                    SELECT_ONE_MEAS=False, 
                    CONNECT_TRACKS=True, 
                    # SAVE = data_path+r"\DA\MC_tsmn.csv"
                    # SAVE = ""
                    SAVE =  r"E:\I24-postprocess\benchmark\TM_1000_pollute_DA.csv"
                    )
    # da.visualize_BA(lanes=[1,2,3,4])
    
    