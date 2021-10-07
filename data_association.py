import numpy as np
import utils
import itertools
from shapely.geometry import Polygon

def dist_score(B, B_data, DIST_MEAS='xy'):
    '''
    compute euclidean distance between two boxes B and B_data
    B: predicted bbox location ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
    B_data: measurement
    '''

    # average displacement RMSE of all points
    if DIST_MEAS == 'xy':
        return np.linalg.norm(B-B_data,2)

    # weighted x,y displacement, penalize y more heavily
    elif DIST_MEAS == 'xyw':
        alpha = 0.2
        return alpha*np.linalg.norm(B[[0,2,4,6]]-B_data[[0,2,4,6]],2) + (1-alpha)*np.linalg.norm(B[[1,3,5,7]]-B_data[[1,3,5,7]],2)
    
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
    
def predict_tracks(tracks):
    '''
    tracks: [dictionary]. Key: car_id, value: mx8 matrix with footprint positions
    if a track has only 1 frame, make the second frame nans
    otherwise do constant-velocity one-step-forward prediction
    '''
    x = []
    for car_id, track in tracks.items():
        if len(track)>1:  
            delta = (track[-1,:] - track[0,:])/(len(track)-1)
            x_pred = track[-1,:] + delta
            tracks[car_id] = np.vstack([track, x_pred])
            x.append(x_pred) # prediction next frame, dim=nx8
        else:
#              x_pred = np.nan*np.empty((1,8)) # nan as place holder, to be interpolated
            # TODO: assume traveling 30m/s based on direction (y axis)
            x_pred = track[-1,:] # keep the last measurement
            tracks[car_id] = np.vstack([track, x_pred])
            x.append(track[-1,:]) # take the last row
#              raise Exception('must have at least 2 frames to predict')
    return x, tracks
            
            
def stitch_objects(df):
    '''
    10/5/2021 modify this function to do one-pass data association
    nearest neighbor DA.
    for every predicted measurement, gate candidate measurements (all bbox within score threshold)
    choose the average of all candidate measurements

    '''
    SCORE_THRESHOLD = 6 # TODO: to be tested, pair if under score_threshold
    
    # define the x,y range to keep track of cars in FOV (meter)
    camera_id_list = df['camera'].unique()
    xmin, xmax, ymin, ymax = utils.get_camera_range(camera_id_list)
    ns = np.amin(np.array(df[['Frame #']])) # start frame
    nf = np.amax(np.array(df[['Frame #']])) # end frame
    tracks = dict() # a dictionary to store all current objects in view
    parent = {} # a dictionary to store all associated tracks
    
    # initialize parent{} with each car itself
    groups = df.groupby('ID')
    gl = list(groups.groups)
    for g in gl:
        parent[g] = g
                
    for k in range(ns,nf):
        # if (k%100==0):
            # print("Frame : %4d" % (k), flush=True)
        # get all measurements from current frame
        frame = df.loc[(df['Frame #'] == k)] # TODO: use groupby frame to save time
        y = np.array(frame[['bbr_x','bbr_y','fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']])
        notnan = ~np.isnan(y).any(axis=1)
        y = y[notnan] # remove rows with missing values (dim = mx8)
        frame = frame.iloc[notnan,:]
        
        m_box = len(frame)
        n_car = len(tracks)
        
        if (n_car > 0): # delete track that are out of view
            for car_id in list(tracks.keys()):
                last_frame_x = tracks[car_id][-1,[0,2,4,6]]
                x1 = min(last_frame_x)
                x2 = max(last_frame_x)
                if (x1<xmin) or (x2>xmax):
#                     print('--------------- deleting {}'.format(car_id), flush=True)
                    del tracks[car_id]
                    n_car -= 1
        
        if (m_box == 0) and (n_car == 0): # simply advance to the next frame
#             print('[1] frame ',k,', no measurement and no tracks')
            continue
            
        elif (m_box == 0) and (n_car > 0): # if no measurements in current frame
#             print('[2] frame ',k,', no measurement, simply predict')
            # make predictions to all existing tracks
            x, tracks = predict_tracks(tracks)
            
        elif (m_box > 0) and (n_car == 0): # create new tracks (initialize)
#             print('[3] frame ',k,', no tracks, initialize with first measurements')
            for index, row in frame.iterrows():
                new_id = row['ID']
                ym = np.array(row[['bbr_x','bbr_y','fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']])
                tracks[new_id] = np.reshape(ym, (1,-1))
        
        else: # if measurement boxes exist in current frame k and tracks is not empty
            # make prediction for each track for frame k
            x, tracks = predict_tracks(tracks)
            n_car = len(tracks)
            curr_id = list(tracks.keys()) # should be n id's 
            
            # calculate score matrix: for car out of scene, score = 99 for place holder
            score = np.ones([m_box,n_car])*99
            for m in range(m_box):
                for n in range(n_car):
                    score[m,n] = dist_score(x[n],y[m],'xyw')
                
            # identify associated (m,n) pairs
#             print('m:',m_box,'total car:',curr_id, 'car in view:',len(tracks))
            bool_arr = score == score.min(axis=1)[:,None]
            score =     bool_arr*score+np.invert(bool_arr)*99 # get the min of each row
            pairs = np.transpose(np.where(score<SCORE_THRESHOLD)) # pair if score is under threshold
#             print(pairs)
            
            # associate based on pairs!
            if len(pairs) > 0:
#                 print('[4a] frame ',k, len(pairs),' pairs are associated')
                for m,n in pairs:
                    new_id = curr_id[n]
                    old_id = frame['ID'].iloc[m]
                    tracks[new_id][-1,:] = y[m] # change the last row from x_pred to ym                  
                    # parent[old_id] = new_id
                    parent = union(parent,old_id, new_id)
                    
            # measurements that have no cars associated, create new
            if len(pairs) < m_box:
    #              print('pairs:',len(pairs),'measuremnts:',m_box)
                m_unassociated = list(set(np.arange(m_box)) - set(pairs[:,0]))
    #              print('[4b] frame ',k, len(m_unassociated),' measurements are not associated, create new')
                for m in m_unassociated:
                    new_id = frame['ID'].iloc[m]
                    tracks[new_id] = np.reshape(y[m], (1,-1))
    parent = compress(parent,gl)
    df['ID'] = df['ID'].apply(lambda x: parent[x] if x in parent else x)
    # TODO: Bayesian approach. take the average of multiple measurements of the same ID at the same frame
    print('Select averaged measurments', len(df))
    df = utils.applyParallel(df.groupby("Frame #"), utils.del_repeat_meas_per_frame).reset_index(drop=True)
    print(len(df))
    return df
    
    
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
                
    print('{} of {} pairs overlap'.format(count,combs))
    return    overlaps

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
    set1 = dict(zip(list(df1['ID'].unique()),list(df1['ID'].unique()))) # initially is carid:map to the same carid
    g2 = df2.groupby('ID')
    max_id = max(max(df1['ID'].values),max(df2['ID'].values))
    for carid, group in g2:
        if carid in set1:
            set1[carid] = max_id + 1
            max_id += 1
    df2['ID'] = df2['ID'].apply(lambda x: set1[x] if x in set1 else x)
    return df2
