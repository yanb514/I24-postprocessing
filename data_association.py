import numpy as np
import utils
import itertools
from shapely.geometry import Polygon
import pandas as pd
import collections
import utils_vis as vis
import scipy
import matplotlib.pyplot as plt

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
            if abs(est_speed)<mpf/2 or (np.sign(est_speed)!=direction): # too slow
                x_pred = track[-1,:] + direction* np.array([mpf,0,mpf,0,mpf,0,mpf,0])
            else:
                x_pred = np.polyval(fit, len(track)) # have "backward" moving cars

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
 
            if  (808<k<810):  
                vis.plot_track(np.array(np.vstack(x), dtype=float), np.array(y,dtype=float), curr_id, frame["ID"].values, xmin,xmax, k)
                    
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
 
def stitch_objects_playground(df, THRESHOLD_1 = 2.5, mc=True):
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
 
            if (1715<k<1760):  
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
