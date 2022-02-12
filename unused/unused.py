import numpy as np
import utils
import itertools
from shapely.geometry import Polygon
import pandas as pd
# import collections
import utils_vis as vis
import scipy
# import matplotlib.pyplot as plt
import torch

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
        # S = np.diag(np.tile([(1/4)**2,(1/0.3)**2],4)) # covariance matrix of x,y distances
        # d = np.sqrt(np.dot(np.dot(diff.T, S),diff))/4
        alpha = (1/3.5)**2
        beta = (1/0.27)**2
        d2 = 0
        for i in range(4):
            d2 += np.sqrt(alpha*diff[i]**2+beta*diff[2*i+1]**2)
        return d2/4
    
    # euclidean distance
    elif DIST_MEAS == 'ed':
        # S = np.diag(np.tile([1,1],4)) # covariance matrix of x,y distances
        # d = np.sqrt(np.dot(np.dot(diff.T, S),diff))/4
        d2 = 0
        for i in range(4):
            d2 += np.sqrt(diff[i]**2+diff[2*i+1]**2)
        return d2/4
    else:
        return


def iou(a,b,DIRECTION=True,AREA=False):
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

def predict_tracks_df(tracks):
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

    
def stitch_objects(df, THRESHOLD_1 = 2.5, THRESHOLD_2 = 2.5, mc=True):
    '''
    10/20/2021
    make sure each meas is either associated to an existing track
    or create a new track
    '''
    
    # define the x,y range to keep track of cars in FOV (meter)
    if mc==True:
        xmin, xmax = min(df["x"].values)-10,max(df["x"].values)+10
    else:
        camera_id_list = df['camera'].unique()
        xmin, xmax, ymin, ymax = utils.get_camera_range(camera_id_list)
        xrange = xmax-xmin
        alpha = 0.4
        xmin, xmax = xmin - alpha*xrange, xmax + alpha*xrange # extended camera range for prediction
    
    ns = int(np.amin(np.array(df[['Frame #']]))) # start frame
    nf = int(np.amax(np.array(df[['Frame #']]))) # end frame
    tracks = dict() # a dictionary to store all current objects in view. key:ID, value:dataframe
    pts = ['bbr_x','bbr_y','fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
    newdf = pd.DataFrame()
    
    for k in range(ns,nf):
        print("\rFrame {}/{}".format(k,nf),end = "\r",flush = True)
        
        frame = df.loc[(df['Frame #'] == k)] # TODO: use groupby frame to save time
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
            x, tracks = predict_tracks_df(tracks)
            
        elif (m_box > 0) and (n_car == 0): # create new tracks (initialize)
            for i, row in frame.iterrows():
                row = frame.loc[i:i,:]
                tracks[row['ID'].iloc[0]] = row
        
        else: # try biparte matching
            x, tracks = predict_tracks_df(tracks)
            n_car = len(tracks)
            curr_id = list(tracks.keys()) # should be n id's 

            # score = np.ones([m_box,n_car])*(99)
            score_dist = np.zeros([m_box,n_car])
            score_iou =np.zeros([m_box,n_car])
            
            for m in range(m_box):
                for n in range(n_car):
                    score_dist[m,n] = dist_score(x[n],y[m],'xyw')
                    score_iou[m,n], _ ,_ = iou(x[n],y[m],DIRECTION=True)
 
            # if  (808<k<810):  
            #     vis.plot_track(np.array(np.vstack(x), dtype=float), np.array(y,dtype=float), curr_id, frame["ID"].values, xmin,xmax, k)
                    
            # distance score
            bool_arr1 = score_dist == score_dist.min(axis=1)[:,None] # every row has true:every measurement gets assigned
            valid = score_dist<THRESHOLD_1
            bool_arr = np.logical_and(bool_arr1,valid)
            multiple_meas = np.sum(bool_arr,axis=0) # multiple meas match to the same ID
            c=np.where(multiple_meas>1)[0] # index of curr_id that's got multiple meas candidates
            if len(c)>0: # exists multiple_meas match, select the nearest one
                bool_arr0 = score_dist == score_dist.min(axis=0) # every col has a true: find the nearest measurement for each prediction
                bool_arr = np.logical_and(bool_arr, bool_arr0) # only choose mutual matching
              
            
    
            pairs = np.transpose(np.where(bool_arr)) # pair if score is under threshold

            if len(pairs) > 0:
                for m,n in pairs:
                    new_id = curr_id[n]
                    tracks[new_id] = tracks[new_id].reset_index(drop=True)
                    new_meas = frame.loc[m:m]   
                    tracks[new_id].drop(tracks[new_id].tail(1).index,inplace=True) # drop the last row (predictino)
                    tracks[new_id] = pd.concat([tracks[new_id], new_meas])  
                    x[n] = np.array(new_meas[pts])
                    
            # measurements that have no cars associated, create new
            if len(pairs) < m_box:
                m_unassociated = list(set(np.arange(m_box)) - set(pairs[:,0]))    
                # if two meas are too close, choose the larger one
                to_del = set()
                for i,m1 in enumerate(m_unassociated):
                    for m2 in m_unassociated[i+1:]:
                        score,a1,a2 = iou(y[m1],y[m2],DIRECTION=False)
                        if score > 0:
                            if a1>a2: # delete a2
                                to_del.add(m2)
                            else:
                                to_del.add(m1)
                                
                            
                    
                for m in m_unassociated:
                    # only create new if meas is far from current meas/pred
                    score_m_dist = [dist_score(y[m], xn,'xyw') for xn in x]
                    score_m_iou = [iou(y[m], xn)[0] for xn in x]
                    if all(np.array(score_m_iou)<=0) and all(np.array(score_m_dist)>THRESHOLD_2) and (m not in to_del): # if ym does not overlap with any existing tracks
                        new_id = frame['ID'].iloc[m]
                        new_meas = frame.loc[m:m]
                        tracks[new_id] = new_meas

    print("Remove wrong direction", len(newdf))
    newdf = utils.remove_wrong_direction_df(newdf)
    print('Connect tracks', len(newdf)) # Frames of a track (ID) might be disconnected after DA
    newdf = newdf.groupby("ID").apply(utils.connect_track).reset_index(drop=True)    
    return newdf  

def count_overlaps(df):
    '''
    similar to remove_overlap
    '''
    # df = df_original.copy() # TODO: make this a method
    
    groups = df.groupby('ID')
    gl = list(groups.groups)
    
    count = 0 # number of pairs that overlaps
    combs = 0
    SCORE_THRESHOLD = 0 # IOU score
    overlaps = set()    
    comb = itertools.combinations(gl, 2)
    for c1,c2 in comb:
        combs+=1
        car1 = groups.get_group(c1)
        car2 = groups.get_group(c2)
        if ((car1['direction'].iloc[0])==(car2['direction'].iloc[0])):
            score = IOU_score(car1,car2)
            if score > SCORE_THRESHOLD:
                count+=1
                overlaps.add((c1,c2))
        else:
            continue
                
    # print('{} of {} pairs overlap'.format(count,combs))
    return overlaps

def IOU_score(car1, car2):
    '''
    calculate the intersection of union of trajectories associated with car1 and car2 based on their overlapped measurements
    https://stackoverflow.com/questions/57885406/get-the-coordinates-of-two-polygons-intersection-area-in-python
    '''
    end = min(car1['Frame #'].iloc[-1],car2['Frame #'].iloc[-1])
    start = max(car1['Frame #'].iloc[0],car2['Frame #'].iloc[0])
    
    if end <= start: # if no overlaps in time
        return -1
    car1 = car1.loc[(car1['Frame #'] >= start) & (car1['Frame #'] <= end)]
    car2 = car2.loc[(car2['Frame #'] >= start) & (car2['Frame #'] <= end)]
    pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
    Y1 = np.array(car1[pts]) # N x 8
    Y2 = np.array(car2[pts])
    IOU = 0
    N = 0
    for j in range(min(len(Y1), len(Y2))):
        D1 = Y1[j,:]
        # try:
        D2 = Y2[j,:]

        if ~np.isnan(np.sum([D1,D2])): # if no Nan in any measurements
            p = Polygon([(D1[2*i],D1[2*i+1]) for i in range(int(len(D1)/2))])
            q = Polygon([(D2[2*i],D2[2*i+1]) for i in range(int(len(D2)/2))])
            if (p.intersects(q)):
                N += 1
                intersection_area = p.intersection(q).area
                union_area = p.union(q).area
        #          print(intersection_area, union_area)
                IOU += float(intersection_area/union_area)
            else:
                IOU += 0
    if N == 0:
        return -1
    return IOU / N
 
def stitch_objects_jpda(df, THRESHOLD_1 = 2.5, mc=True):
    '''
    10/20/2021
    use JPDA, weighted average of all meas that fall into a gate (defined by IOU and mahalanobis distance)
    create new ID for meas out side of the gate
    '''
    
    # define the x,y range to keep track of cars in FOV (meter)
    if mc==True:
        xmin, xmax = min(df["x"].values)-10,max(df["x"].values)+10
    else:
        camera_id_list = df['camera'].unique()
        xmin, xmax, ymin, ymax = utils.get_camera_range(camera_id_list)
        xrange = xmax-xmin
        alpha = 0.4
        xmin, xmax = xmin - alpha*xrange, xmax + alpha*xrange # extended camera range for prediction
    
    ns = int(np.amin(np.array(df[['Frame #']]))) # start frame
    nf = int(np.amax(np.array(df[['Frame #']]))) # end frame
    tracks = dict() # a dictionary to store all current objects in view. key:ID, value:dataframe
    pts = ['bbr_x','bbr_y','fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
    pts_img = ["fbrx","fbry","fblx","fbly",	"bbrx",	"bbry",	"bblx",	"bbly",	"ftrx",	"ftry",	"ftlx",	"ftly",	"btrx",	"btry",	"btlx",	"btly"]
    newdf = pd.DataFrame()
    
    for k in range(ns,nf):
        print("\rFrame {}/{}".format(k,nf),end = "\r",flush = True)
        
        frame = df.loc[(df['Frame #'] == k)] # TODO: use groupby frame to save time
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
            x, tracks = predict_tracks_df(tracks)
            
        elif (m_box > 0) and (n_car == 0): # create new tracks (initialize)
            for i, row in frame.iterrows():
                row = frame.loc[i:i,:]
                tracks[row['ID'].iloc[0]] = row
        
        else: 
            x, tracks = predict_tracks_df(tracks)
            n_car = len(tracks)
            curr_id = list(tracks.keys()) # should be n id's 

            # score = np.ones([m_box,n_car])*(99)
            score_dist = np.zeros([m_box,n_car])
            score_iou =np.zeros([m_box,n_car])
            
            invalid_meas = []
            for m in range(m_box):
                for n in range(n_car):
                    score_dist[m,n] = dist_score(x[n],y[m],'maha')
                    score_iou[m,n], areaa, areab = iou(x[n],y[m],DIRECTION=False,AREA=True)

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

    # print("Remove wrong direction", len(newdf))
    # newdf = utils.remove_wrong_direction_df(newdf)
    # print('Connect tracks', len(newdf)) # Frames of a track (ID) might be disconnected after DA
    # newdf = newdf.groupby("ID").apply(utils.connect_track).reset_index(drop=True)    
    return newdf

def stitch_objects_bm(df, THRESHOLD_1 = 2.5, mc=True):
    '''
    bipartite matching based on Maha distance cost
    '''
    
    # define the x,y range to keep track of cars in FOV (meter)
    if mc==True:
        xmin, xmax = min(df["x"].values)-10,max(df["x"].values)+10
    else:
        camera_id_list = df['camera'].unique()
        xmin, xmax, ymin, ymax = utils.get_camera_range(camera_id_list)
        xrange = xmax-xmin
        alpha = 0.4
        xmin, xmax = xmin - alpha*xrange, xmax + alpha*xrange # extended camera range for prediction
    
    ns = int(np.amin(np.array(df[['Frame #']]))) # start frame
    nf = int(np.amax(np.array(df[['Frame #']]))) # end frame
    tracks = dict() # a dictionary to store all current objects in view. key:ID, value:dataframe
    pts = ['bbr_x','bbr_y','fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
    # pts_img = ["fbrx","fbry","fblx","fbly",	"bbrx",	"bbry",	"bblx",	"bbly",	"ftrx",	"ftry",	"ftlx",	"ftly",	"btrx",	"btry",	"btlx",	"btly"]
    newdf = pd.DataFrame()
    
    for k in range(ns,nf):
        print("\rFrame {}/{}".format(k,nf),end = "\r",flush = True)
        
        frame = df.loc[(df['Frame #'] == k)] # TODO: use groupby frame to save time
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
            x, tracks = predict_tracks_df(tracks)
            
        elif (m_box > 0) and (n_car == 0): # create new tracks (initialize)
            for i, row in frame.iterrows():
                row = frame.loc[i:i,:]
                tracks[row['ID'].iloc[0]] = row
        
        else: 
            x, tracks = predict_tracks_df(tracks)
            n_car = len(tracks)
            curr_id = list(tracks.keys()) # should be n id's 

            # score = np.ones([m_box,n_car])*(99)
            score_dist = np.zeros([m_box,n_car])
            score_iou =np.zeros([m_box,n_car])
            
            invalid_meas = set()
            invalid_tracks = set()
            for m in range(m_box):
                for n in range(n_car):
                    score_dist[m,n] = dist_score(x[n],y[m],'maha')
                    score_iou[m,n], areaa, areab = iou(x[n],y[m],DIRECTION=False,AREA=True)
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

    # print("Remove wrong direction", len(newdf))
    # newdf = utils.remove_wrong_direction_df(newdf)
    # print('Connect tracks', len(newdf)) # Frames of a track (ID) might be disconnected after DA
    # newdf = newdf.groupby("ID").apply(utils.connect_track).reset_index(drop=True)    
    return newdf

def stitch_objects_gnn(df, THRESHOLD_1 = 2.5, mc=True):
    '''
    find the best meas for each track
    prioritize on tracks that have higher # meas matched
    '''
    
    # define the x,y range to keep track of cars in FOV (meter)
    if mc==True:
        xmin, xmax = min(df["x"].values)-10,max(df["x"].values)+10
    else:
        camera_id_list = df['camera'].unique()
        xmin, xmax, ymin, ymax = utils.get_camera_range(camera_id_list)
        xrange = xmax-xmin
        alpha = 0.4
        xmin, xmax = xmin - alpha*xrange, xmax + alpha*xrange # extended camera range for prediction
    
    ns = int(np.amin(np.array(df[['Frame #']]))) # start frame
    nf = int(np.amax(np.array(df[['Frame #']]))) # end frame
    tracks = dict() # a dictionary to store all current objects in view. key:ID, value:dataframe
    pts = ['bbr_x','bbr_y','fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
    # pts_img = ["fbrx","fbry","fblx","fbly",	"bbrx",	"bbry",	"bblx",	"bbly",	"ftrx",	"ftry",	"ftlx",	"ftly",	"btrx",	"btry",	"btlx",	"btly"]
    newdf = pd.DataFrame()
    
    for k in range(ns,nf):
        print("\rFrame {}/{}".format(k,nf),end = "\r",flush = True)
        
        frame = df.loc[(df['Frame #'] == k)] # TODO: use groupby frame to save time
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
            x, tracks = predict_tracks_df(tracks)
            
        elif (m_box > 0) and (n_car == 0): # create new tracks (initialize)
            for i, row in frame.iterrows():
                row = frame.loc[i:i,:]
                tracks[row['ID'].iloc[0]] = row
        
        else: 
            x, tracks = predict_tracks_df(tracks)
            n_car = len(tracks)
            curr_id = list(tracks.keys()) # should be n id's 

            # score = np.ones([m_box,n_car])*(99)
            score_dist = np.zeros([m_box,n_car])
            score_iou =np.zeros([m_box,n_car])
            
            invalid_meas = set()
            # invalid_tracks = set()
            for m in range(m_box):
                for n in range(n_car):
                    score_dist[m,n] = dist_score(x[n],y[m],'maha')
                    score_iou[m,n], areaa, areab = iou(x[n],y[m],DIRECTION=False,AREA=True)
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

    # print("Remove wrong direction", len(newdf))
    # newdf = utils.remove_wrong_direction_df(newdf)
    # print('Connect tracks', len(newdf)) # Frames of a track (ID) might be disconnected after DA
    # newdf = newdf.groupby("ID").apply(utils.connect_track).reset_index(drop=True)    
    return newdf

def pts_to_line(x,y,a,b,c):
    '''
    distance from (x,y) to a line defined by ax+by+c=0
    x,y are vectors of the same length
    a,b,c, are scalars
    return the average distances
    '''
    d = np.abs(a*x + b*y + c)/np.sqrt(a**2+b**2)
    return np.nanmean(d)
    
    
def time_space_matching(df, THRESHOLD_X = 9, THRESHOLD_Y = 0.5):
    '''
    try to match tracks that are "close" to each other in time-space dimension
    offline matching, take advantage of a linear model
    '''
    groups = df.groupby("ID")
    groupList = list(groups.groups)
    pts = ['bbr_x','bbr_y','fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
   
    n_car = len(groupList)
    DX = np.ones((n_car, n_car)) * 99 # x-distance triangle matrix
    DY = np.ones((n_car, n_car)) * 99 # lower triangle matrix
    
    for i,c1 in enumerate(groupList):
        # get the fitted line for c1
        track1 = groups.get_group(c1)
        t1 = track1["Frame #"].values
        track1 = np.array(track1[pts])
        x1 = (track1[:,0] + track1[:,2])/2
        y1 = np.nanmean((track1[:,1] + track1[:,7])/2)
        cx1 = np.nanmean(x1)
 
        fit = np.polyfit(t1,x1,1)
        a,b,c = -fit[0], 1, -fit[1]
        
        for j in range(n_car):
            if i == j:
                continue
            track2 = groups.get_group(groupList[j])
            t2 = track2["Frame #"].values
            track2 = np.array(track2[pts])
            x2 = (track2[:,0] + track2[:,2])/2
            y2 = np.nanmean((track2[:,1] + track2[:,7])/2)
            cx2 = np.nanmean(x2)
            DX[i,j] = pts_to_line(t2,x2,a,b,c)
            DY[i,j] = np.abs(y1-y2)/np.abs(cx1-cx2) # tan\theta
            
    matched = np.logical_and(DX<THRESHOLD_X, DY<THRESHOLD_Y)

    for i in range(n_car-1,-1,-1):
        idx = np.where(np.logical_and(matched[:,i], matched[i,:]))[0]
        if len(idx)>0:
            newid = groupList[idx[0]] # assigned to the first ID appeared in scene
            parent = {groupList[i]: newid for i in list(idx)+[i]}
            df['ID'] = df['ID'].apply(lambda x: parent[x] if x in parent else x)
            
    # post process
    
    return df, groupList,DX,DY
    
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

def stitch_objects_tsmn_ll(o, THRESHOLD_C=50, VARX=0.03, VARY=0.03, time_out = 500):
        '''
        similar to stitch_objects_tsmn
        instead of binary fit, calculate a loss (log likelihood)
        max missing frames T to go above threshold: 0.5 log(VARX T) + 0.5 log(VARY T) = THRESHOLD_C
            -> T = np.exp(THRESHOLD_C) / np.sqrt(VARX*VARY)
        '''
        df = o.df
        groups = {k: v for k, v in df.groupby("ID")}
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

                CX[i,j] = nll
                
        # for debugging only
        o.CX = CX
        o.groupList = groupList
        o.empty_id = empty_id
        
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
        
        o.path = path          
        # delete IDs that are empty
        df = df.groupby("ID").filter(lambda x: (x["ID"].iloc[0] not in empty_id))
        path = path.copy()
        # modify ID
        while path:
            key = list(path.keys())[0]
            reid = {groupList[old]: groupList[key] for old in path[key]} # change id=groupList[v] to groupList[key]
            df = df.replace({'ID': reid})
            for v in list(path[key]) + [key]:
                try:
                    path.pop(v)
                except KeyError:
                    pass

        df = df.sort_values(by=['Frame #','ID']).reset_index(drop=True)         
        o.df = df
        print("\n")
        print("Before DA: {} unique IDs".format(len(groupList))) 
        print("After DA: {} unique IDs".format(df.groupby("ID").ngroups))
        print("True: {} unique IDs".format(len([id for id in groupList if id<1000])))
      
        return o
      
def stitch_objects_tsmn_online(o, THRESHOLD_MIN, THRESHOLD_MAX=3, VARX=0.03, VARY=0.03, time_out = 500):
        '''
        pitfall: strictly lowest-cost match, pairs occur later in time cannot be matched because they're not ready. Pairs occurs first may not be matched beacuase lower in priority
        THRESHOLD_MIN: below which pairs have to be matched
        THRESHOLD_MAX: aobve which pairs should never be matched
        online version of stitch_objects_tsmn_ll
        track: dict with key: id, t, x, y
            {"id": 20,
             "t": [frame1, frame2,...],
             "x":[x1,x2,...], 
             "y":[y1,y2...],
             "fitx": [vx, bx], least square fit
             "fity": [vy, by]}
        tracks come incrementally as soon as they end
        '''
        # define cost
        def _getCost(track1, track2):
            '''
            track1 always ends before track2 ends
            999: mark as conflict
            -1: invalid
            '''
            if track1['id']==track2['id']:
                return -1
            # if (track1['id'] in empty_id) or (track1['id'] in empty_id):
            #     return -1
            if track2["t"][0] < track1['t'][-1]: # if track2 starts before track1 ends
                return 999
            if track2['t'][0] - track1['t'][-1] > time_out: # if track2 starts TIMEOUT after track1 ends
                return -1
            
            xx = np.vstack([track2['t'],np.ones(len(track2['t']))]).T # N x 2
            targetx = np.matmul(xx, track1['fitx'])
            targety = np.matmul(xx, track1['fity'])
            pt1 = track1['t'][-1]
            varx = (track2['t']-pt1) * VARX 
            vary = (track2['t']-pt1) * VARY
            input = torch.transpose(torch.tensor([track2['x'],track2['y']]),0,1)
            target = torch.transpose(torch.tensor([targetx, targety]),0,1)
            var = torch.transpose(torch.tensor([varx,vary]),0,1)
            nll = loss(input,target,var)
            return nll.item()
        
        def _addEdge(graph,u,v):
            # add undirected edge
            graph[u].add(v)
            graph[v].add(u)
        
        def _first(s):
            '''Return the first element from an ordered collection
               or an arbitrary element from an unordered collection.
               Raise StopIteration if the collection is empty.
            '''
            return next(iter(s.values()))
        
        df = o.df
        # sort tracks by start/end time - not for real deployment
        
        groups = {k: v for k, v in df.groupby("ID")}
        ids = list(groups.keys())
        ordered_tracks = deque() # list of dictionaries
        all_tracks = {}
        S = []
        E = []
        for id, car in groups.items():
            t = car["Frame #"].values
            x = (car.bbr_x.values + car.bbl_x.values)/2
            y = (car.bbr_y.values + car.bbl_y.values)/2
            notnan = ~np.isnan(x)
            t,x,y = t[notnan], x[notnan],y[notnan]
            if len(t)>1: # ignore empty or only has 1 frame
                S.append([t[0], id])
                E.append([t[-1], id])
                track = {"id":id, "t": t, "x": x, "y": y} 
                # ordered_tracks.append(track)
                all_tracks[id] = track

            
        heapq.heapify(S) # min heap (frame, id)
        heapq.heapify(E)
        EE = E.copy()
        while EE:
            e, id = heapq.heappop(EE)
            ordered_tracks.append(all_tracks[id])
            
        # Initialize
        X = defaultdict(set) # exclusion graph
        curr_tracks = deque() # tracks in view. list of tracks. should be sorted by end_time
        path = {} # oldid: newid. to store matching assignment
        C = [] # min heap. {cost: (id1, id2)} cost to match start of id1 to end of id2
        past_tracks = set() # set of ids indicate end of track ready to be matched
        processed = set() # set of ids whose tails are matched
        matched = 0 # count matched pairs
        
        running_tracks = OrderedDict() # tracks that start but not end at e 
        
        for track in ordered_tracks:
            print("\n")
            
            curr_id = track['id'] # last_track = track['id']
            path[curr_id] = curr_id
            print('at end of: ',curr_id)
            
            right = track['t'][-1] # right pointer: current time
            
            # get tracks that started but not end - used to define the window left pointer
            while S and S[0][0] < right: # append all the tracks that already starts
                started_time, started_id = heapq.heappop(S)
                running_tracks[started_id] = started_time
            print('running tracks: ', running_tracks.keys())
            # compute track statistics
            t,x,y = track['t'],track['x'],track['y']
            ct = np.nanmean(t)

            #     empty_id.add(track['id'])
            #     continue
            if len(t)<2:
                v = np.sign(x[-1]-x[0]) # assume 1/-1 m/frame = 30m/s
                b = x-v*ct # recalculate y-intercept
                fitx = np.array([v,b[0]])
                fity = np.array([0,y[0]])
            else:
                xx = np.vstack([t,np.ones(len(t))]).T # N x 2
                fitx = np.linalg.lstsq(xx,x, rcond=None)[0]
                fity = np.linalg.lstsq(xx,y, rcond=None)[0]
            track['t'] = t
            track['x'] = x
            track['y'] = y
            track['fitx'] = fitx
            track['fity'] = fity
            
            try: left = max(0,_first(running_tracks) - time_out)
            except: left = 0
            print("pointers :", left, right)
            # remove out of sight tracks
            
            while curr_tracks and curr_tracks[0]['t'][-1] < left:           
                past_tracks.add(curr_tracks.popleft()['id'])
            
            print("curr tracks: ",[i['id'] for i in curr_tracks])
            print("past tracks: ", past_tracks)
            # compute score from every track in curr to track, update Cost
            for curr_track in curr_tracks:
                if curr_track['id']==track['id']: continue
                cost = _getCost(curr_track, track)
                if cost > THRESHOLD_MAX:
                    _addEdge(X, curr_track['id'], track['id'])
                elif cost > 0:
                    heapq.heappush(C, (cost, (curr_track['id'], track['id'])))

            try: print("best cost:", C[0])
            except: print(" **no cost")
            # start matching
            while True:
                if len(C) == 0:
                    break
                cost = C[0][0]
                id1, id2 = C[0][1] # min cost pair
                if id2 in X[id1]: # oonflicts
                    heapq.heappop(C) # remove from current pool
                elif id1 in past_tracks: # if the former track is ready to be matched
                # elif cost < THRESHOLD_MIN:
                    # if path[id2]._find() == path[id1]._find(): 
                    if path[id2] == path[id1]: # already matched
                        heapq.heappop(C) # remove from current pool
                    # elif id2 not in X[id1]: # match if no exlusion
                    else:
                        print("match", id1, id2)
                        # path[id1]._union(path[id2]) # update id2's root to id1's
                        path[id2] = path[id1]
                        matched += 1
                        # update X: make id1's neighbors(conflicts) also id2's neighbors
                        conflicts1 = X[id1].copy()
                        conflicts2 = X[id2].copy()
                        for conf in conflicts1: _addEdge(X, id2, conf)  
                        for conf in conflicts2: _addEdge(X, id1, conf)      
                        # update cost: 
                        heapq.heappop(C)
                    # else: # id1 is matched to all possible tails
                    #     processed.add(id1)
                    #     break
                else: # keep waiting for new tracks
                    break # break while loop, but stay in for loop
                    
            curr_tracks.append(track)        
            running_tracks.pop(curr_id) # remove tracks that ended
            print("matched:", matched)
            
        # delete IDs that are empty
        print("\n")
        print("{} Ready: ".format(len(past_tracks)))
        # print("{} Processsed: ".format(len(processed)))
        print("{} pairs matched".format(matched))
        # print("Deleting {} empty tracks".format(len(empty_id)))
        # df = df.groupby("ID").filter(lambda x: (x["ID"].iloc[0] not in empty_id))
        
        # for debugging only
        o.path = path
        o.C = C
        o.X = X
        o.groupList = ids
        o.past_tracks = past_tracks
        
        # replace IDs
        newids = [v for _,v in path.items()]
        m = dict(zip(path.keys(), newids))
        df = df.replace({'ID': m})
        df = df.sort_values(by=['Frame #','ID']).reset_index(drop=True)         
        
        print("Before DA: {} unique IDs".format(len(ids))) 
        print("After DA: {} unique IDs".format(df.groupby("ID").ngroups))
        print("True: {} unique IDs".format(len([id for id in ids if id<1000])))
      
        o.df = df
        return o
 