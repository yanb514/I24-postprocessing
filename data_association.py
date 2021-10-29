import numpy as np
import utils
import itertools
from shapely.geometry import Polygon
import pandas as pd
import collections
import utils_vis as vis
import scipy

def dist_score(B, B_data, DIST_MEAS='xy'):
    '''
    compute euclidean distance between two boxes B and B_data
    B: predicted bbox location ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
    B_data: measurement
    '''
    B = np.reshape(B,(-1,8))
    B_data = np.reshape(B_data,(-1,8))
    # check sign
    # if np.sign(B[0,[2]]-B[0,[0]])!=np.sign(B_data[0,[2]]-B_data[0,[0]]): # if not the same direction
    #     return 99
    
    if (np.sign(B[0,[2]]-B[0,[0]])!=np.sign(B_data[0,[2]]-B_data[0,[0]])) : # if not the same x direction
        return 99
    
    diff = B-B_data
    diff = diff[0]
    mae_x = np.mean(np.abs(diff[[0,2,4,6]])) 
    mae_y = np.mean(np.abs(diff[[1,3,5,7]])) 
    
    if DIST_MEAS == 'xy':
        # return np.linalg.norm(B-B_data,2) # RMSE
        return (mae_x + mae_y)/2

    # weighted x,y displacement, penalize y more heavily
    elif DIST_MEAS == 'xyw':
        alpha = 0.2
        # return alpha*np.linalg.norm(B[[0,2,4,6]]-B_data[[0,2,4,6]],2) + (1-alpha)*np.linalg.norm(B[[1,3,5,7]]-B_data[[1,3,5,7]],2)
        return alpha*mae_x + (1-alpha)*mae_y
    else:
        return
    
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
        # except:
            # print(Y2.shape)
            # print(j)
            # print(car1)
            # print(car2)
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

def iou(a,b):
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
            return 0
        
        if (np.sign(a[0,[2]]-a[0,[0]])!=np.sign(b[0,[2]]-b[0,[0]])):# if not the same x / y direction
            return -1
        
        ax = np.sort(a[0,[0,2,4,6]])
        ay = np.sort(a[0,[1,3,5,7]])
        bx = np.sort(b[0,[0,2,4,6]])
        by = np.sort(b[0,[1,3,5,7]])

        area_a = (ax[3]-ax[0]) * (ay[3]-ay[0])
        area_b = (bx[3]-bx[0]) * (by[3]-by[0])
        
        minx = max(ax[0], bx[0]) # left
        maxx = min(ax[2], bx[2])
        miny = max(ay[1], by[1])
        maxy = min(ay[3], by[3])
        
        intersection = max(0, maxx-minx) * max(0,maxy-miny)
        # union = area_a + area_b - intersection + 1e-06
        union = min(area_a,area_b)
        iou = intersection/union
        
        return iou
    
    

            
            

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
        #     temp = track_df[~track_df["bbr_x"].isna()]
        #     time_range = (len(temp)-1)/30
        #     vx_bbr = (max(temp.bbr_x.values)-min(temp.bbr_x.values))/time_range
        #     vx_fbr = (max(temp.fbr_x.values)-min(temp.fbr_x.values))/time_range
        #     vx = (vx_bbr+vx_fbr)/2
        #     print(vx)
        #     delta = np.array([vx,0,vx,0,vx,0,vx,0])/30
        #     x_pred = track[-1,:] + direction*delta 
            
            frames = np.arange(0,len(track))
            fit = np.polyfit(frames,track,1)
            est_speed = np.mean(fit[0,[0,2,4,6]])
            if abs(est_speed)<mpf/2 or (np.sign(est_speed)!=direction): # too slow
                x_pred = track[-1,:] + direction* np.array([mpf,0,mpf,0,mpf,0,mpf,0])
            else:
                x_pred = np.polyval(fit, len(track)) # have "backward" moving cars
            
        # if track_df["Frame #"].count()>1: # predict based on places that have measurements
        # if len(track)>1:
        #     frames = np.arange(0,len(track))
        #     fit = np.polyfit(frames,track,1)
        #     x_curr = np.polyval(fit, len(track)-1)
        #     x_pred = x_curr + direction* np.array([mpf,0,mpf,0,mpf,0,mpf,0])   

        else:
            x_pred = track[-1,:] + direction* np.array([mpf,0,mpf,0,mpf,0,mpf,0])
        x_pred = np.reshape(x_pred,(1,-1))
        x.append(x_pred) # prediction next frame, dim=nx8
        new_row = pd.DataFrame(x_pred, columns=pts)
        tracks[car_id] = pd.concat([tracks[car_id], new_row])
    return x, tracks

    
def stitch_objects(df, THRESHOLD_1 = 2.5, THRESHOLD_2 = 2.5):
    '''
    10/20/2021
    make sure each meas is either associated to an existing track
    or create a new track
    '''
    
    # define the x,y range to keep track of cars in FOV (meter)
    camera_id_list = df['camera'].unique()
    xmin, xmax, ymin, ymax = utils.get_camera_range(camera_id_list)
    xrange = xmax-xmin
    alpha = 0.4
    xmin, xmax = xmin - alpha*xrange, xmax + alpha*xrange # extended camera range for prediction
    ns = np.amin(np.array(df[['Frame #']])) # start frame
    nf = np.amax(np.array(df[['Frame #']])) # end frame
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
                    score_iou[m,n] = iou(x[n],y[m])
 
            if  (150<k<160):  
                vis.plot_track(np.array(np.vstack(x), dtype=float), np.array(y,dtype=float), curr_id, frame["ID"].values, ["p1c2"], k)
            
            # lienar sum assignment for biparte matching
            # pairs = []
            # matched_meas = set()
            # a, b = scipy.optimize.linear_sum_assignment(score_dist)
            # for i in range(len(a)):
            #     m,n = a[i], b[i]
            #     if score_dist[m,n] < SCORE_THRESHOLD:
            #         pairs.append([m,n])
            #         matched_meas.add(m)
                    
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
                for m in m_unassociated:
                    # only create new if m does not result in close gap
                    # iou_m = []
                    # for xn in x:
                    #     iou_m.append(iou(y[m], xn))
                    score_m_dist = [dist_score(y[m], xn,'xyw') for xn in x]
                    score_m_iou = [iou(y[m], xn) for xn in x]
                    if all(np.array(score_m_iou)<=0) and all(np.array(score_m_dist)>THRESHOLD_2): # if ym does not overlap with any existing tracks
                        new_id = frame['ID'].iloc[m]
                        new_meas = frame.loc[m:m]
                        tracks[new_id] = new_meas

    print("Remove wrong direction", len(newdf))
    newdf = utils.remove_wrong_direction_df(newdf)
    print('Connect tracks', len(newdf)) # Frames of a track (ID) might be disconnected after DA
    newdf = newdf.groupby("ID").apply(utils.connect_track).reset_index(drop=True)    
    return newdf  


 

def associate_cross_camera(df_original):
    '''
    this function is essentially the same as associate_overlaps
    '''
    df = df_original.copy() # TODO: make this a method
    
    camera_list = ['p1c1','p1c2','p1c3','p1c4','p1c5','p1c6']
    # camera_list = ['p1c5','p1c6'] # for debugging
    groups = df.groupby('ID')
    gl = list(groups.groups)
    
    # initialize tree
    parent = {}
    for g in gl:
        parent[g] = g
            
    df = df.groupby(['ID']).filter(lambda x: len(x['camera'].unique()) != len(camera_list)) # filter
    SCORE_THRESHOLD = 0 # IOU score
    
    for i in range(len(camera_list)-1):
        camera1, camera2 = camera_list[i:i+2]
        print('Associating ', camera1, camera2)
        df2 = df[(df['camera']==camera1) | (df['camera']==camera2)]
        df2 = df2.groupby(['ID']).filter(lambda x: len(x['camera'].unique()) < 2) # filter
        
        groups2 = df2.groupby('ID')
        gl2 = list(groups2.groups)
        
        # initialize tree
        # parent = {}
        # for g in gl:
            # parent[g] = g
            
        comb = itertools.combinations(gl2, 2)
        
        for c1,c2 in comb:
            car1 = groups2.get_group(c1)
            car2 = groups2.get_group(c2)
            if ((car1['Object class'].iloc[0]) == (car2['Object class'].iloc[0])) & ((car1['camera'].iloc[0])!=(car2['camera'].iloc[0])):
                score = IOU_score(car1,car2)
                if score > SCORE_THRESHOLD:
                    # associate!
                    # parent[c2] = c1
                    parent = union(parent, c1, c2)
            else:
                continue
                
        # path compression (part of union find): compress multiple ID's to the same object            
    parent = compress(parent, gl)
        # change ID to first appeared ones
        # df2['ID'] = df2['ID'].apply(lambda x: parent[x] if x in parent else x)
        
    return parent

# def check_overlaps(pair, groups, parent):
    # c1,c2 = pair
    # car1 = groups.get_group(c1)
    # car2 = groups.get_group(c2)
    # if ((car1['direction'].iloc[0])==(car2['direction'].iloc[0])):
        # score = IOU_score(car1,car2)
        # if score > SCORE_THRESHOLD:
            # associate!
            # parent[c2] = c1
    # else:
        # continue
    # return parent
            
def associate_overlaps(df_original):
    '''
    get all the ID pairs that associated to the same car based on overlaps
    '''
    df = df_original.copy() # TODO: make this a method
    
    groups = df.groupby('ID')
    gl = list(groups.groups)
    
    # initialize tree
    parent = {}
    for g in gl:
        parent[g] = g
            
    SCORE_THRESHOLD = 0 # IOU score
                
    comb = itertools.combinations(gl, 2)
    for c1,c2 in comb:
        car1 = groups.get_group(c1)
        car2 = groups.get_group(c2)
        if ((car1['direction'].iloc[0])==(car2['direction'].iloc[0])):
            score = IOU_score(car1,car2)
            if score > SCORE_THRESHOLD:
                # associate!
                if len(car1)>len(car2): # change car2's id to car1's
                    parent = union(parent, c2, c1)
                else:
                    parent = union(parent, c1, c2)
        else:
            continue
    # with multiprocessing.Pool() as pool:
        # parent = pool.map(partial(check_overlaps,groups=groups,parent=parent), range(comb))
                
    # path compression (part of union find): compress multiple ID's to the same object            
    parent = compress(parent, gl)
        
    return parent

def remove_overlaps(df):
    '''
    based on the occasions where multiple boxes and IDs are associated with the same object at the same time
    remove the shorter track
    '''
    # df = df_original.copy() # TODO: make this a method
    
    groups = df.groupby('ID')
    gl = list(groups.groups)
    
    id_rem = {} # the ID's to be removed
    
    SCORE_THRESHOLD = 0 # IOU score
                
    comb = itertools.combinations(gl, 2)
    for c1,c2 in comb:
        car1 = groups.get_group(c1)
        car2 = groups.get_group(c2)
        if ((car1['direction'].iloc[0])==(car2['direction'].iloc[0])):
            score = IOU_score(car1,car2)
            if score > SCORE_THRESHOLD:
                first1 = car1['Frame #'][car1['bbr_x'].notna().idxmax()]
                first2 = car2['Frame #'][car2['bbr_x'].notna().idxmax()]
                last1 = car1['Frame #'][car1['bbr_x'].notna()[::-1].idxmax()]
                last2 = car2['Frame #'][car2['bbr_x'].notna()[::-1].idxmax()]
                # end = min(car1['Frame #'].iloc[-1],car2['Frame #'].iloc[-1])
                # start = max(car1['Frame #'].iloc[0],car2['Frame #'].iloc[0])
                start = max(first1, first2)
                end = min(last1, last2)
                # associate!
                if len(car1)>len(car2): # change removes the overlaps from car 2
                    id_rem[c2] = (start,end)

                else:
                    id_rem[c1] = (start,end)
        else:
            continue
                
    # remove ID that are not in the id_rem set    
    # df = df.groupby("ID").filter(lambda x: (x['ID'].iloc[0] not in id_rem))
    print('id_rem',len(id_rem))
    df = df.groupby("ID").apply(remove_overlaps_per_id, args = id_rem).reset_index(drop=True)
    return df

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

def remove_overlaps_per_id(car, args):
    id_rem = args
    if car['ID'].iloc[0] in id_rem:
        start,end = id_rem[car['ID'].iloc[0]] # remove all frames between start and end, including
        car = car[(car['Frame #']<start) | (car['Frame #']>end)] # what to keep
        # car = car[car['Frame #']>end]
        if len(car)==0:
            return None
        return car
    else:
        return car
# path compression
def find(parent, i):
    # if parent[parent[i]] == i:
        # parent[i] = i
    if parent[i] == i:
        return i
    # if (parent[i] != i):
        # print(i, parent[i])
        # parent[i] = find(parent, parent[i])
    # return parent[i]
    return find(parent, parent[i])

def union(parent, x,y):
    xset = find(parent,x)
    yset = find(parent,y)
    parent[xset] = yset
    return parent
    
def compress(parent, groupList):    
    for i in groupList:
        find(parent, i)
    return parent 
    
def assign_unique_id(df1, df2):
    '''
    modify df2 such that no IDs in df2 is a duplicate of that in df1
    '''
    set1 = dict(zip(list(df1['ID'].unique()),list(df1['ID'].unique()))) # initially is carid:map to the same carid
    g2 = df2.groupby('ID')
    max_id = max(max(df1['ID'].values),max(df2['ID'].values))
    for carid, group in g2:
        if carid in set1:
            set1[carid] = max_id + 1
            max_id += 1
    df2['ID'] = df2['ID'].apply(lambda x: set1[x] if x in set1 else x)
    return df2
