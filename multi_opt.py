#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 10:33:54 2022

@author: yanbing_wang
"""
import json
import numpy as np
from bson.objectid import ObjectId
import os
import matplotlib.pyplot as plt
from i24_database_api import DBClient
import cvxpy as cp
# from cvxopt import matrix, solvers, sparse,spdiag,spmatrix
import time
from scipy.sparse import identity, coo_matrix, hstack, csr_matrix,lil_matrix
from utils.utils_opt import resample
import pandas as pd
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

from itertools import combinations
from collections import defaultdict, OrderedDict
import queue
from datetime import datetime
from multiprocessing.pool import ThreadPool
from utils.utils_opt import combine_fragments
import i24_logger.log_writer as log_writer


def plot_track(tracks):
    fig,ax = plt.subplots(1,2,figsize=(9,4))
    
    for track in tracks:
        try:
            length = np.nanmedian(track["length"])
            width = np.nanmedian(track["width"])
        except TypeError:
            length = track["length"]
            width = track["width"]
        x = np.array(track["x_position"])
        t = track["timestamp"]
        y = np.array(track["y_position"])

        ax[0].fill_between(t, x, x +track["direction"]*length, alpha=0.5, label=track["_id"], interpolate=True)
        ax[1].fill_between(t, y + 0.5*width, y- 0.5*width, alpha=0.5,interpolate=True)
    ax[0].legend(loc="lower right")
    plt.show()
    return


def _blocdiag_scipy(X, n):
    """
    makes diagonal blocs of X, for indices in [sub1,sub2]
    n indicates the total number of blocks (horizontally)
    """
    # if not isinstance(X, spmatrix):
    #     X = sparse(X)
    a,b = X.shape
    if n==b:
        return X
    else:
        mat = lil_matrix((n-b+1, n))
        for i in range(n-b+1):
            mat[i, i:i+b] = X
        return mat
    
def animate_tracks(snapshots, offset = 0, save=False, name="before"):
    '''
    resample tracks, make to the dataframe, join the dataframe to get the overlapped time
    make an animation
    '''

    # animate it!
    fig, ax = plt.subplots(1,1, figsize=(25,5))

    ax.set(ylim=[0,120])
    ax.set(xlim=[0,2000])
    ax.set_aspect('equal', 'box')
    
    start_time = min(snapshots.keys())
    frames = sorted([key for key in snapshots.keys() if key > start_time+offset]) # sort time index
    dt = frames[1]-frames[0]
    print("dt: ",dt)
    

    def animate(i):
        # plot the ith row in df3
        # remove all car_boxes 
        if i > start_time + offset:
        
            time_text = datetime.utcfromtimestamp(int(i)).strftime('%m/%d/%Y, %H:%M:%S')
            plt.suptitle(time_text, fontsize = 20)
            
            for pc in ax._children:
                pc.remove()
            
            snapshot = snapshots[i]
            # pos = [centerx, centery, l, w, dir]
            boxes = [patches.Rectangle(xy=(pos[0]-pos[4]*0.5*pos[2], pos[1]-0.5*pos[3]), width=pos[2], height=pos[3]) for _, pos in snapshot.items()]
            pc = PatchCollection(boxes, alpha=1,
                             edgecolor="blue")
            ax.add_collection(pc)
            return ax,
        
    # Init only required for blitting to give a clean slate.
    def init():
        print("init")
        return ax,

    anim = animation.FuncAnimation(fig, func=animate,
                                        init_func= init,
                                        frames=frames,
                                        repeat=False,
                                        interval=dt*1000*0.5, # in ms
                                        blit=False,
                                        cache_frame_data = False,
                                        save_count = 1)
    if save:
        
        file_name = name+ ".mp4"
        anim.save(file_name, writer='ffmpeg', fps=25)
        # self.anim.save('{}.gif'.format(file_name), writer='imagemagick', fps=self.framerate)
        print("saved.")
        
    # fig.tight_layout()
    # plt.show()
        
    return anim

    
def solve_single(track, lam2_x, lam2_y, lam3_x, lam3_y, lam1_x, lam1_y):
    '''
    solve for smoothing (opt2_l1) in cvxpy
    '''
    track = resample(track)
    N = len(track["x_position"])
    idx = [i.item() for i in np.argwhere(~np.isnan(track["x_position"])).flatten()]
    zx = np.array(track["x_position"])[idx] # measurement
    zy = np.array(track["y_position"])[idx] # measurement
    M = len(zx)
    
    # define some matrices
    t1 = time.time()
    D1 = _blocdiag_scipy(coo_matrix([-1,1]), N) * (1/dt)
    D2 = _blocdiag_scipy(coo_matrix([1,-2,1]), N) * (1/dt)
    D3 = _blocdiag_scipy(coo_matrix([-1,3,-3,1]), N) * (1/dt)
    
    I = identity(N).toarray()
    H = I[idx,:]
    t2 = time.time()
    print("build mats: ", t2-t1)
    
    
    # Construct a CVXPY problem
    x = cp.Variable(N)
    y = cp.Variable(N)
    ex = cp.Variable(M)
    ey = cp.Variable(M)
    
    constraints = [
        -D1@x <= 0,
        D2@x <= 10,
        -D2@x <= 10,
        D3@x <= 10,
        -D3@x <= 10,
    ]
    t1 = time.time()
    
    # solve for x
    cx_pre, cx = 999, 998
    max_iter = 10
    iter = 0
    while cx - cx_pre < 0 and iter <= max_iter:
        print("iter, ", iter, cx)
        obj1 = 1/M * cp.sum_squares(zx- H@x - ex) + lam2_x/(N-2) * cp.sum_squares(D2 @ x) + lam3_x/(N-3) * cp.sum_squares(D3 @ x) + lam1_x/M * cp.norm(ex, 1)
        prob1 = cp.Problem(cp.Minimize(obj1), constraints)
        prob1.solve(solver="CVXOPT", warm_start=True)
        cx_pre = cx
        cx = sum(abs(H@x.value-zx))/M
        iter += 1
        lam1_x += 1e-3

     
    # solve for y
    obj2 = 1/M * cp.sum_squares(zy- H@y - ey) + lam2_y/(N-2) * cp.sum_squares(D2 @ y) + lam3_y/(N-3) * cp.sum_squares(D3 @ y) + lam1_y/M * cp.norm(ey, 1)
    prob2 = cp.Problem(cp.Minimize(obj2))
    prob2.solve(solver='CVXOPT', warm_start=True)
    
    t2 = time.time()
    print("Time: ", t2-t1)
    print("Status: ", prob1.status, prob2.status)
    print("The optimal value is", prob1.value, prob2.value)
    # print("A solution x is")
    # print(x.value)
    track["x_position"] = x.value
    track["y_position"] = y.value
    return track
    

def find_overlap_idx(x, y):
    '''
    x,y are timestamp arrays
    y ends before x
    find the intervals for x and y overlap, i.e.,
    x[s1: e1] overlaps with y[s2, e2]
    '''
    s1,s2=0,0
    # find starting pointers
    while s1 < len(x) and s2 < len(y):
        if abs(x[s1] - y[s2]) < 1e-3:
            break
        elif x[s1] < y[s2]:
            s1 += 1
        else:
            s2 += 1
    # find ending poitners
    e1, e2 = len(x)-1, len(y)-1
    while e1 >0 and e2 >0:
        if abs(x[e1] - y[e2]) < 1e-3:
            break
        if x[e1] < y[e2]:
            e2 -= 1
        else:
            e1 -= 1
            
    return s1, e1, s2, e2
  
    
def plot_configuration(tracks):
    '''
    tracks are after resampling
    make tracks into a matrix X = [xi,
                                   xj,
                                   yi,
                                   yj]
    AX + b < 0 is the conflict constraint (at least 1 should be satisifed)
    compute AX + b
    '''
    zxi = np.array(tracks[0]["x_position"])
    zxj = np.array(tracks[1]["x_position"])
    zyi = np.array(tracks[0]["y_position"])
    zyj = np.array(tracks[1]["y_position"])

    ti = tracks[0]["timestamp"]
    tj = tracks[1]["timestamp"]
    li = np.nanmedian(tracks[0]["length"])
    lj = np.nanmedian(tracks[1]["length"])
    wi = np.nanmedian(tracks[0]["width"])
    wj = np.nanmedian(tracks[1]["width"])

    s1, e1, s2, e2 = find_overlap_idx(ti, tj)
    K = e1-s1+1
    Oi = hstack([csr_matrix((K, s1)), identity(K), csr_matrix((K, len(ti)-e1-1))])
    Oj = hstack([csr_matrix((K, s2)), identity(K), csr_matrix((K, len(tj)-e2-1))])
    
    X = np.vstack([Oi@ zxi, Oj@zxj, Oi@zyi, Oj@zyj])
    A = np.array([[-1, 1, 0, 0],
                  [1, -1, 0, 0],
                  [0, 0, 1, -1],
                  [0, 0, -1, +1]])
    padx, pady = 5, 1
    b = np.array([lj + padx, li + padx, 0.5*(wi+wj) + pady, 0.5*(wi+wj) + pady])
    
    conf = A @ X + np.tile(b, (K,1)).T # 4 x K
    # confb = conf < 0 # binary matrix
    
    # plot
    fig,ax = plt.subplots(1,2,figsize=(9,4))
    pos0 = ax[0].imshow(conf < 0, cmap="RdYlGn", aspect="auto", interpolation='none')
    ax[0].set_xlabel("timestamp")
    ax[0].set_title("configuration before conflict resolution")
    ax[0].set_yticklabels(["", "i is in front of j", "", "i is behind j","", "i is right of j", "","i is left of j",""])
    
    
    # fillnan with interpolation
    fcn = lambda z: z.nonzero()[0]
    nans = np.isnan(conf[1,:])
    for i in range(conf.shape[0]):
        conf[i,nans]= np.interp(fcn(nans), fcn(~nans), conf[i,~nans])
    
    # resolve conflict by pushing to the "best" direction
    # take the min of each column
    mincol = np.argmin(conf, axis=0)
    confb = conf < 0 # binary matrix

    
    # resolve conflict
    conf_time = np.where(~np.any(confb, axis=0))[0] # select time where all are false (conflicts occur)
    confb[mincol[conf_time], conf_time] = True # flip those to True -> indicating the flipped bisection is the direction to pull apart
    
    # plot the rest
    pos1 = ax[1].imshow(confb, cmap="RdYlGn", aspect="auto", interpolation='none')
    fig.colorbar(pos0, ax=ax[0])
    
    fig.colorbar(pos1, ax=ax[1])
    ax[1].set_yticklabels(["", "i is in front of j", "", "i is behind j","", "i is right of j", "","i is left of j",""])
    
    ax[1].set_xlabel("timestamp")
    ax[1].set_title("configuration after conflict resolution")
    
    return conf, confb
    
  
def configuration(p1, p2):
    '''
    p1, p2: [centerx, cnetery, l, w]
    return configuration
    [1 is behind 2, 1 is in front of 2, 1 is above 2, 1 is below 2]
    TODO: ADD DIRECTION
    '''
    xpad, ypad = 5,1

    return [p1[0]-0.5*p1[2]-xpad>p2[0]+0.5*p2[2], # x1>>x2, 1 is infront of 2 
            p1[0]+0.5*p1[2]+xpad<p2[0]-0.5*p2[2], # x1<<x2, 1 is behind 2
            p1[1]+0.5*p1[3]+ypad<p2[1]-0.5*p2[3], # y1<<y2, 1 is right of 2
            p1[1]-0.5*p1[3]-ypad>p2[1]+0.5*p2[3]] # y1>>y2, 1 is left of 2
    
def separatum(snapshots):
    '''
    https://mikekling.com/comparing-algorithms-for-dispersing-overlapping-rectangles/
    snpashot = {
        i:[centerx, centery, l, w],
        j:[...],
        ...
        }
    beta is non-overlap configuration for all overalpped timestamps
    beta = { # this schema is easier to build constraints
        (i,j): { "overlap_t": [t1,t2,...],
                "configuration": [[T,T,F,F], [F,F,T,T],...]
                }
        }
    is possible that beta may not contain all the combinations of all ids in snapshots, because overlap is only evaluated if two cars appear in the same time
    '''

    beta = defaultdict(dict) # 
    moved = False
    min_moved_t = 1e10
    for t, snapshot in snapshots.items():
        overlap = True
        while overlap:
            overlap = False
            vec = {}
            for key,val in snapshot.items():
                vec[key] = np.array([0,0])
                snapshot[key] = np.array(val)
                
            for i,j in combinations(snapshot.keys(), 2):
                    
                conf = configuration(snapshot[i], snapshot[j])
                beta[t][(i,j)] = conf # at this point should have no conflicts
                
                if not any(conf): # if all are true, meaning has overlap
                    overlap = True # reset
                    # print(i,j)
                    moved = True
                    min_moved_t = min(min_moved_t, t)
                    # TODO: this vector could be based on iou shape
                    vector = snapshot[i][:2] - snapshot[j][:2]
                    l2 = np.linalg.norm(vector,2)
                    vec[i] = vec[i] + 0.5/l2 * vector
                    vec[j] = vec[j] - 0.5/l2 * vector
                # else:
                    
                
            # move them! 
            for i, v in vec.items():
                snapshot[i][:2] += v
                
    # write beta in a different schema
    beta_transform = defaultdict(dict)
    for t,beta_t in beta.items():
        for pair, conf in beta_t.items():
            try:
                beta_transform[pair]["overlap_t"].append(t)
            except:
                beta_transform[pair]["overlap_t"] = [t]
            try:
                beta_transform[pair]["configuration"].append(conf)
            except:
                beta_transform[pair]["configuration"] = [conf]

    return snapshots, beta_transform, moved, min_moved_t
            

def time_transform(tracks, dt=0.04):
    '''
    outer join on time intervals of all tracks
    tracks are NOT resampled
    return snapshots:{
        t1: {
            traj1: [centerx, centery, l, w],
            traj2: [...].
            },
        t2: {
            traj1: [],
            traj2: [],
            traj3: [],
            ...
            }
        ...
        }
        
    '''
    time_series_field = ["timestamp", "x_position", "y_position"]
    snapshots = defaultdict(dict)
    for traj in tracks:
        _id = traj["_id"]
        dir = traj["direction"]
        try:
            l,w = np.nanmedian(traj["length"]), np.nanmedian(traj["width"])
        except:
            l,w = traj["length"], traj["width"]
            
        data = {key:traj[key] for key in time_series_field}
        df = pd.DataFrame(data, columns=data.keys()) 
        index = pd.to_timedelta(df["timestamp"], unit='s')
        df = df.set_index(index)
        df = df.drop(columns = "timestamp")
        
        # resample to 5hz
        df=df.groupby(df.index.floor(str(dt)+"S")).mean().resample(str(dt)+"S").asfreq()
        df.index = df.index.values.astype('datetime64[ns]').astype('int64')*1e-9
        df = df.interpolate(method='linear')
        
        # add to result dictionary
        for t in df.index:
            # [centerx, centery, l ,w]
            snapshots[t][_id] = np.array([df["x_position"][t] + dir*0.5*l,df["y_position"][t],l,w, dir])
    
    return snapshots



def tracks_2_snapshots(tracks):
    '''
    outer join on time intervals of all tracks
    tracks ARE resampled already
    track:
        {
            "_id": _id
            "t":[],
            "x":[],
            "y":[],
            "w":float,
            "l":float,
            "dir":1/-1
            }
    return snapshots:{
        t1: {
            traj1: [centerx, centery, l, w],
            traj2: [...].
            },
        t2: {
            traj1: [],
            traj2: [],
            traj3: [],
            ...
            }
        ...
        }
        
    '''
    snapshots = defaultdict(dict)
    for traj in tracks:
        _id = traj["_id"]
        dir = traj["dir"]
        try:
            l,w = np.nanmedian(traj["l"]), np.nanmedian(traj["w"])
        except:
            l,w = traj["l"], traj["w"]
        
        # add to result dictionary
        for i,t in enumerate(traj["t"]):
            # [centerx, centery, l ,w]
            snapshots[t][_id] = np.array([traj["x"][i] + dir*0.5*l,traj["y"][i],l,w, dir])
    
    return snapshots
    


def snapshots_2_tracks(snapshots):
    '''
    snapshots are time-transformed representation of tracks
    time-indexed, resampled, outer-join
    [centerx, centery, l,w,dir]
    TODO: treat l,w as time-series in time-transform()
    tracks = [
         {
            _id: id
            t: [],
            x: [],
            y: [],
            l; [],
            w: [],
            dir: 1
            },
        { ... }
        ]
    '''
    # back to traj based documents
    lru = OrderedDict()
    tracks = []
    timestamps = sorted(snapshots.keys())
    dt = timestamps[1] - timestamps[0] # assume uniform sample
    for t in timestamps:
        for _id, pos in snapshots[t].items():
            x,y,l,w,dir = pos
            # create new
            if _id not in lru:
                lru[_id] = defaultdict(list)
                lru[_id]["dir"] = dir
            
            lru[_id]["t"].append(t)
            lru[_id]["x"].append(x)
            lru[_id]["y"].append(y)
            lru[_id]["l"].append(l)
            lru[_id]["w"].append(w)
            lru.move_to_end(_id)
            
        # check if any is timed out -> output to tracks
        while lru[next(iter(lru))]["t"][-1] < t - dt:
            ID, traj = lru.popitem(last=False) #FIFO
            traj["_id"] = ID
            # idList.append(ID)
            tracks.append(traj)
    
    # pop the rest in lru
    while lru:
        ID, traj = lru.popitem(last=False) #FIFO
        traj["_id"] = ID
        # idList.append(ID)
        tracks.append(traj)
    
    return tracks
 
    

def _build_obj_fcn_constr(tracks, beta, lam2_x, lam2_y, lam3_x, lam3_y, lam1_x, lam1_y):
    '''
    # TODO: assume no missing data in tracks because it was linearly interpolated before
    beta = { # this schema is easier to build constraints
        (i,j): { "overlap_t": [t1,t2,...],
                "configuration": [[T,T,F,F], [F,F,T,T],...]
                }
        
        }
    config:[p1[0]+0.5*p1[2]+xpad<p2[0]-0.5*p2[2], # x1>>x2
            p1[0]-0.5*p1[2]-xpad>p2[0]+0.5*p2[2], # x2>>x1
            p1[1]-0.5*p1[3]-ypad>p2[1]+0.5*p2[3], # y1>>y2
            p1[1]+0.5*p1[3]+ypad<p2[1]-0.5*p2[3]] # y2>>y1
    '''
    # get decision variables
    X = []
    Y = []
    EX = []
    EY = []
    OBJX = []
    OBJY = []

    for i,track in enumerate(tracks):
        n = len(track["t"])
        # dt = track["t"][1]-track["t"][0]

        zx = track["x"]
        zy = track["y"]
        x = cp.Variable(n)
        y = cp.Variable(n)
        ex = cp.Variable(n)
        ey = cp.Variable(n)

        # D1 = _blocdiag_scipy(coo_matrix([-1,1]), n) * (1/dt)
        D2 = _blocdiag_scipy(coo_matrix([1,-2,1]), n) * (1/dt)
        D3 = _blocdiag_scipy(coo_matrix([-1,3,-3,1]), n) * (1/dt)
        objx = 1/n * cp.sum_squares(zx-x-ex) + lam2_x/(n-2) * cp.sum_squares(D2@x) + lam3_x/(n-3) * cp.sum_squares(D3@x) + lam1_x/n * cp.norm(ex,1)
        objy = 1/n * cp.sum_squares(zy-y-ey) + lam2_y/(n-2) * cp.sum_squares(D2@y) + lam3_y/(n-3) * cp.sum_squares(D3@y) + lam1_y/n * cp.norm(ey,1)
    
        # get your s*** together
        X.append(x)
        Y.append(y)
        EX.append(ex)
        EY.append(ey)
        OBJX.append(objx)
        OBJY.append(objy)
        
    
    
    constraints = []
    big_M = 1e6
    padx, pady = 4,1
    # num_cars = len(tracks)
    
    for i,j in combinations(range(len(tracks)), 2):

        tracki, trackj = tracks[i], tracks[j]
        try:
            confb = np.array(beta[(tracki["_id"], trackj["_id"])]["configuration"]).T # 4xK
        except KeyError: # track i and j do not have time overlap
            continue
        si, ei, sj, ej = find_overlap_idx(tracki["t"], trackj["t"])
        xi, xj, yi, yj = X[i], X[j], Y[i], Y[j]
        wi, wj, li, lj = np.nanmedian(tracki["w"]), np.nanmedian(trackj["w"]), np.nanmedian(tracki["l"]), np.nanmedian(trackj["l"])
        RHS = big_M*(1-confb*1)
        constraints.extend([
            -xi[si:ei+1] + xj[sj:ej+1] + lj+padx <= RHS[0,:],
            xi[si:ei+1] - xj[sj:ej+1] + li+padx <= RHS[1,:],
            yi[si:ei+1] - yj[sj:ej+1] + 0.5*(wi+wj)+pady <= RHS[2,:],
            -yi[si:ei+1] + yj[sj:ej+1] + 0.5*(wi+wj)+pady <= RHS[3,:],
        ])
        
           
    # all decision variables, all obj functions and all constraints
    return X,Y,EX,EY,OBJX,OBJY,constraints


  

def _build_obj_fcn(tracks, lam2_x, lam2_y, lam3_x, lam3_y, lam1_x, lam1_y):
    '''
    No constraints! just obj fcn!
    '''
    # get decision variables
    X = []
    Y = []
    EX = []
    EY = []
    OBJX = []
    OBJY = []

    for i,track in enumerate(tracks):
        n = len(track["t"])
        # dt = track["t"][1]-track["t"][0]

        zx = track["x"]
        zy = track["y"]
        x = cp.Variable(n)
        y = cp.Variable(n)
        ex = cp.Variable(n)
        ey = cp.Variable(n)

        # D1 = _blocdiag_scipy(coo_matrix([-1,1]), n) * (1/dt)
        D2 = _blocdiag_scipy(coo_matrix([1,-2,1]), n) * (1/dt)
        D3 = _blocdiag_scipy(coo_matrix([-1,3,-3,1]), n) * (1/dt)
        objx = 1/n * cp.sum_squares(zx-x-ex) + lam2_x/(n-2) * cp.sum_squares(D2@x) + lam3_x/(n-3) * cp.sum_squares(D3@x) + lam1_x/n * cp.norm(ex,1)
        objy = 1/n * cp.sum_squares(zy-y-ey) + lam2_y/(n-2) * cp.sum_squares(D2@y) + lam3_y/(n-3) * cp.sum_squares(D3@y) + lam1_y/n * cp.norm(ey,1)
    
        # get your s*** together
        X.append(x)
        Y.append(y)
        EX.append(ex)
        EY.append(ey)
        OBJX.append(objx)
        OBJY.append(objy)
        
           
    # all decision variables, all obj functions and all constraints
    return X,Y,EX,EY,OBJX,OBJY



def solve_collision_avoidance3(tracks, lam2_x, lam2_y, lam3_x, lam3_y, lam1_x, lam1_y):
    
    # resolve conflict first -> may need to iterate
    snapshots = time_transform(tracks, dt=0.04)
    snapshots, beta, _ = separatum(snapshots)
    tracks_s = snapshots_2_tracks(snapshots) # tracks_s are de-conflicted, but not smoothed
    
    # get decision vars and objective functions
    X,Y,EX,EY,OBJX,OBJY,constraints = _build_obj_fcn(tracks_s, beta, lam2_x, lam2_y, lam3_x, lam3_y, lam1_x, lam1_y)
    
    # combine to a problem
    obj = cp.sum(OBJX) + cp.sum(OBJY) # may need to scale y
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(verbose=True) #solver="ECOS_BB" runs forever, solver="SCIP" takes forever, qcp=True if problem is DQCP
     
    print("Status: ", prob.status)
    print("The optimal value is", prob.value)
    
    # modify tracks
    for i, track in enumerate(tracks):
        track["timestamp"] = tracks_s[i]["t"]
        track["x_position"] = X[i].value
        track["y_position"] = Y[i].value
        track["length"] = np.median(tracks_s[i]["l"])
        track["width"] = np.median(tracks_s[i]["w"])
    
    return tracks

dt = 0.04
def preprocess_reconcile(direction, stitched_queue, parameters, db_param):
    '''
    stitched_queue is ordered by last_timestamp
    1. combine and resample stitched fragments from queue
    2. write to rec (combined, resampled but not smoothed)
    3. transform trajs to time-indexed and write to temp_raw
    
    TODO: make combine_fragments more efficient
    tracks are NOT resampled
    temp schema:
    {
         {
            "_id":
            "timetamp": t1
            "traj1_id": [centerx, centery, l, w],
            "traj2_id": [...],
            ...
            },
        {
           "_id":
           "timetamp": t2
           "traj1_id": [centerx, centery, l, w],
           "traj2_id": [...],
           ...
           },
    }   
    '''
    logger = log_writer.logger
    logger.set_name("preproc_reconcile_"+direction)
    setattr(logger, "_default_logger_extra",  {})
    
    temp = DBClient(**db_param, database_name = "temp", collection_name = parameters["raw_collection"]) # write results to temp
    rec = DBClient(**db_param, database_name = parameters["reconciled_database"], collection_name = parameters["reconciled_collection"])
    
    temp.collection.create_index("timestamp")
    time_series_field = ["timestamp", "x_position", "y_position"]
    lru = OrderedDict()
    stale = defaultdict(int) # key: timestamp, val: number of timestamps that it hasn't been updated
    stale_thresh = 50 # if a timestamp is not updated after processing [stale_thresh] number of trajs, then update to database
    last_poped_t = 0
    pool = ThreadPool(processes=100)
    
    
    while True:
        try:
            traj = stitched_queue.get(timeout = 10)
            traj = combine_fragments(traj)
        except queue.Empty:
            logger.info("queue empty.")
            break
    
        # increment stale
        for k in stale:  
            stale[k] += 1
        
        _id = traj["_id"]
        dir = traj["direction"]
        try:
            l,w = np.nanmedian(traj["length"]), np.nanmedian(traj["width"])
        except:
            l,w = traj["length"], traj["width"]
            
        data = {key:traj[key] for key in time_series_field}
        df = pd.DataFrame(data, columns=data.keys()) 
        index = pd.to_timedelta(df["timestamp"], unit='s')
        df = df.set_index(index)
        df = df.drop(columns = "timestamp")
        
        # resample to 1/dt hz
        df=df.groupby(df.index.floor(str(dt)+"S")).mean().resample(str(dt)+"S").asfreq()
        df.index = df.index.values.astype('datetime64[ns]').astype('int64')*1e-9
        df = df.interpolate(method='linear')
        
        # assemble in traj
        # do not extrapolate for more than 1 sec
        first_valid_time = pd.Series.first_valid_index(df['x_position'])
        last_valid_time = pd.Series.last_valid_index(df['x_position'])
        first_time = max(min(traj['timestamp']), first_valid_time-1)
        last_time = min(max(traj['timestamp']), last_valid_time+1)
        df=df[first_time:last_time]
        
        # traj['x_position'] = list(df['x_position'].values)
        # traj['y_position'] = list(df['y_position'].values)
        # traj['timestamp'] = list(df.index.values)
        traj["first_timestamp"] = df.index.values[0]
        traj["last_timestamp"] = df.index.values[-1]
        traj["starting_x"] = df['x_position'].values[0]
        traj["ending_x"] = df['x_position'].values[-1]
        
        # insert to rec collection, for postprocess use
        # insert without timestamp, x, y
        traj.pop('timestamp', None)
        traj.pop('x_position', None)
        traj.pop('y_position', None)
        pool.apply_async(thread_insert_one, (rec.collection, traj,))

        
        # add to result dictionary
        for t in df.index:
            
            # [centerx, centery, l ,w]
            try:
                lru[t][str(_id)] = [df["x_position"][t] + dir*0.5*l,df["y_position"][t],l,w, dir]
            except: # t does not exists in lru yet
                if t <= last_poped_t:
                    # meaning t was poped pre-maturely
                    print("t was poped prematurely from LRU in transform_queue")
                    
                lru[t] = {str(_id): [df["x_position"][t] + dir*0.5*l,df["y_position"][t],l,w, dir]}
            lru.move_to_end(t, last=True)
            stale[t] = 0 # reset staleness
            
        # update db from lru
        while stale[next(iter(lru))] > stale_thresh:
            t, d = lru.popitem(last=False) # pop first
            last_poped_t = t
            stale.pop(t)
            # change d to value.objectid: array, so that it does not reset the value field, but only update it
            d={"value."+key: val for key,val in d.items()}
            pool.apply_async(thread_update_one, (temp.collection, {"timestamp": round(t,2)},{"$set": d},))
            
            
    # write the rest of lru to database
    logger.info("Flush out the rest in LRU")
    while len(lru) > 0:
        t, d = lru.popitem(last=False) # pop first
        d={"value."+key: val for key,val in d.items()}
        pool.apply_async(thread_update_one, (temp.collection, {"timestamp": round(t,2)},{"$set": d},))
        
        
    pool.close()
    pool.join()
    logger.info("Final timestamps in temp: {}".format(temp.collection.count_documents({})))   
    del temp
    del rec
    return 



def solve_collision_avoidance_rolling(direction, parameters, db_param):
    '''
    a rolling window approach to solve reconciliation problem
    for each window, iterate between separation and smoothing until no move conflicts
    and move on to the next window
    TODO: query direction
    rec_transformed.collection.find_one({}, sort=[("timestamp",1)])["timestamp"]
    '''
    logger = log_writer.logger
    logger.set_name("reconcile_rolling_"+direction)
    setattr(logger, "_default_logger_extra",  {})
    
    # listen to the transformed collection in temp db
    raw_transformed = DBClient(**db_param, database_name = "temp", collection_name = parameters["raw_collection"])
    rec_transformed = DBClient(**db_param, database_name = "temp", collection_name = parameters["reconciled_collection"])
    rec_transformed.create_index("timestamp")
    pool = ThreadPool(processes=200) # for insert
    
    PH = parameters["ph"]
    IH = parameters["ih"]
    continuity_steps = 3
    reconciliation_args = parameters["reconciliation_args"]
    
    constr_rhs = {} # continuity constraints are initialized to none (first window)
    maxiter = 1
    timeout = 5
    
    # wait for the first timestamp in raw_transformed
    while raw_transformed.collection.count_documents({}) == 0:
        time.sleep(2)
        logger.info("Waiting for the start of the first window")
    start = raw_transformed.get_min("timestamp")
    end = start + PH
    last = False # indicate if it is the last rolling window for stop condition
    
    while not last: # for each distinct rolling window
        
        # wait for end to show up in database
        t1 = time.time()
        while raw_transformed.get_max("timestamp") < end:
            time.sleep(2)
            logger.info("Waiting for temp_raw to accumulate enough time-based data")
            t2 = time.time()
            if t2-t1 > timeout:
                # set end pointer as the last timestamp in this database
                end = raw_transformed.get_max("timestamp") + 0.1
                last = True
                logger.info("Enter the last rolling window")
                break
            
            
        query = raw_transformed.get_range("timestamp", start, end)
        snapshots = {time_doc["timestamp"]: time_doc["value"] for time_doc in query}
        snapshots, _, hasconflicts, min_moved_t = separatum(snapshots)
        logger.info("Conflict before smoothing: {}".format(hasconflicts))
        
        tracks_s = snapshots_2_tracks(snapshots) # tracks_s are de-conflicted, but not smoothed
        tracks_s = [track for track in tracks_s if len(track["t"]) > 3]# possible that tracks in tracks_s have len < 3
        
        hasconflicts = True # set to true first to enter the following while loop
        iter = 0
        
        while hasconflicts: # within each window
            # query from [curr: curr+PH)
            start_text = datetime.utcfromtimestamp(int(start)).strftime('%H:%M:%S')
            end_text = datetime.utcfromtimestamp(int(end)).strftime('%H:%M:%S')
            max_text = datetime.utcfromtimestamp(int(raw_transformed.get_max("timestamp"))).strftime('%H:%M:%S') 
            logger.info("start: {}, end: {}, max: {}".format(start_text, end_text, max_text))
            
            # get decision vars and objective functions
            # X,Y,EX,EY,OBJX,OBJY,constraints = _build_obj_fcn_constr(tracks_s, beta, **reconciliation_args)
            X,Y,EX,EY,OBJX,OBJY = _build_obj_fcn(tracks_s, **reconciliation_args)
            
            # add continuity constraints: the first <continuity_steps> is set as equality constraints. RHS values should be fixed in the inner while loop
            constraints = []
            # constr_rhs = {} # TODO: delete me debug only
            if constr_rhs:
                for i, track in enumerate(tracks_s):
                    try:
                        ns = len(constr_rhs[track["_id"]][0])
                        constraints.extend([
                            X[i][:ns] == constr_rhs[track["_id"]][0],
                            Y[i][:ns] == constr_rhs[track["_id"]][1]
                            ])
                    except KeyError: # not long enough
                        pass
            
            if len(constraints) > 0:
                logger.info("{} tracks have {} continuity constraints".format(len(tracks_s), len(constraints)/2))
            
            # combine to a problem
            obj = cp.sum(OBJX) + cp.sum(OBJY) # may need to scale y
            prob = cp.Problem(cp.Minimize(obj), constraints)
            prob.solve(verbose=False) #solver="ECOS_BB" runs forever, solver="SCIP" takes forever, qcp=True if problem is DQCP
            logger.info("Status: {}".format(prob.status))

            # modify tracks
            # check if smoothing re-introduced conflicts
            for i, track in enumerate(tracks_s):
                for j,t in enumerate(track["t"]):
                    tracks_s[i]["x"][j] = X[i].value[j]
                    tracks_s[i]["y"][j] = Y[i].value[j]
                    
            snapshots_post = tracks_2_snapshots(tracks_s)
            snapshots_post, _, hasconflicts,min_moved_t = separatum(snapshots_post)
            logger.info("Conflict after smoothing: {}".format(hasconflicts))

            if min_moved_t < start + continuity_steps*dt:
                print("min_moved_T within continuity step")
                
            # hasconflicts = False
            if hasconflicts and iter < maxiter:
                tracks_s = snapshots_2_tracks(snapshots_post) # re-separate tracks to iterate next loop
                iter += 1
                logger.info("iter {}: re-solve for this window".format(iter))

            # only write result and move to the next window if no more conflicts in this window
            else:
                if hasconflicts:
                    logger.info("*** STILL has conflicts!!!")
                constr_rhs = {} # reset constraints
                hasconflicts = False
                # update to the next window
                start += IH
                end = start+PH
    
                # write result of [curr: curr+IH] to database temp->reconciled_collection
                upper = start + continuity_steps * dt
                for i, track in enumerate(tracks_s):
                    val_id = "value." + str(track["_id"])
                    xx,yy = [],[]
                    for j,t in enumerate(track["t"]):
                        # update constr_rhs
                        if t >= start and t < upper: # TODO check open and close range
                            xx.append(track["x"][j])
                            yy.append(track["y"][j])
                        
                        if not last and t > start + continuity_steps: # curr is already updated to the end of this rolling window
                            break # should break the inner for loop
                         
                        filter = {"timestamp": round(t,2)}
                        update = {"$set": {val_id: [X[i].value[j],Y[i].value[j],
                                                    np.median(tracks_s[i]["l"]),
                                                    np.median(tracks_s[i]["w"]),
                                                    tracks_s[i]["dir"]]
                                  }}
                        pool.apply_async(thread_update_one, (rec_transformed.collection, filter, update, ))# none-blocking!
                    
                    # update constr_rhs here
                    ns = len(xx)
                    if ns > 0:
                        constr_rhs[track["_id"]] = [xx,yy]

    logger.info("Exit rolling solver.") 
    pool.close()
    pool.join()
    del raw_transformed
    del rec_transformed
    return


def thread_update_one(collection, filter, update, upsert=True):
    '''
    TODO: put in the database_api later
    '''
    collection.update_one(filter, update, upsert)
    return

def thread_insert_one(collection, doc):
    '''
    TODO: put in the database_api later
    '''
    collection.insert_one(doc)
    return
 
    
 
def postprocess_reconcile(direction, parameters, db_param):
    '''
    direction: "eb" or "wb"
    read from the temporary database with snapshots (time-based documents), transform to trajectory-based documents
    update t,x,y to the reconciled trajectory database, which was originally created during combine_trajectories right after stitch
    if temp is delete after processing, this process can be restarted without data loss
    '''
    logger = log_writer.logger
    logger.set_name("postprocess_reconcile_"+direction)
    setattr(logger, "_default_logger_extra",  {})
    
    temp = DBClient(**db_param, database_name = "temp", collection_name = parameters["reconciled_collection"])
    rec = DBClient(**db_param, database_name = "reconciled", collection_name = parameters["reconciled_collection"])
    pool = ThreadPool(processes = 100)
    timeout = 5 # in sec. if no new documents in 30 from temp, timeout this process
    
    # wait for the first document to show up
    while temp.count() == 0:
        time.sleep(2)
    start = temp.get_min("timestamp")
    
    # make sure that end > start
    while temp.get_max("timestamp") <= start:
        time.sleep(2)
    end = temp.get_max("timestamp") # end may increase as more data comes in
    
    # back to traj based documents
    lru = OrderedDict() # key:str(id), val={"t":[], "x":[], "y":[]}
    last = False # indicate the last batch operation
    
    while not last:
        logger.info("Query temp_rec from {:.2f} - {:.2f}".format(start, end))
        query = temp.get_range("timestamp", start, end) # [start, end)
        for snapshot in query:
            curr_time = snapshot["timestamp"]
            for _id, pos in snapshot["value"].items():
                x,y,l,w,dir = pos
                try:
                    lru[ObjectId(_id)]["t"].append(curr_time)
                    lru[ObjectId(_id)]["x"].append(x-0.5*dir*l) # make centerx to backcenter x
                    lru[ObjectId(_id)]["y"].append(y)
                    # lru[ObjectId(_id)]["l"] = l
                    # lru[ObjectId(_id)]["w"] = w
                    
                except KeyError: #initialize
                    lru[ObjectId(_id)] = {
                        "t":[curr_time],
                        "x":[x],
                        "y":[y],
                        "l":l,
                        "w":w,
                        # "dir":dir
                        }
                lru.move_to_end(ObjectId(_id))
                
             
            temp.collection.delete_one({"_id": snapshot["_id"]})
        
            # if a traj is ready, update to rec_collection
            while lru[next(iter(lru))]["t"][-1] < curr_time - 0.5:
                _id, lru_traj = lru.popitem(last=False) #FIFO
                # do not "$set", otherwise if timeout is premature, it will reset the previous records
                pool.apply_async(thread_update_one, (rec.collection,{"_id":_id}, 
                                                     {"$push": {
                                                         "timestamp": {"$each": lru_traj["t"]},
                                                         "x_position": {"$each": lru_traj["x"]},
                                                         "y_position": {"$each": lru_traj["y"]}
                                                         },
                                                      "$set": {
                                                          "length":lru_traj["l"],
                                                          "width":lru_traj["w"]
                                                          }
                                                      }))

        # see if query (window) can be expanded
        start = end
        newend = end
        t1 = time.time()
        while temp.count() > 0 and not newend > temp.get_max("timestamp"): # should keep expending
            time.sleep(2)
            end = temp.get_max("timestamp") # expand window
            t2 = time.time()
            if t2-t1 > timeout:
                last = True
                logger.info("timeout for postprocess_reconcile ")
                break


    # pop the rest in lru
    logger.info("Update the rest in LRU, size: {}".format(len(lru)))
    while lru:
        _id, lru_traj = lru.popitem(last=False) #FIFO
        pool.apply_async(thread_update_one, (rec.collection,{"_id":_id}, 
                                             {"$push": {
                                                 "timestamp": {"$each": lru_traj["t"]},
                                                 "x_position": {"$each": lru_traj["x"]},
                                                 "y_position": {"$each": lru_traj["y"]}
                                                 },
                                              "$set": {
                                                  "length":lru_traj["l"],
                                                  "width":lru_traj["w"]
                                                  }
                                              }))
    
    pool.close()
    pool.join()
    del temp
    del rec
    return
    
        


if __name__ == '__main__':
    
    # initialize parameters
    import multiprocessing as mp
    mp_manager = mp.Manager()
    
    with open("config/parameters.json") as f:
        parameters = json.load(f)
        
    with open(os.path.join(os.environ["USER_CONFIG_DIRECTORY"], "db_param.json")) as f:
        db_param = json.load(f)
        
    
    # for key in ["lam3_x","lam3_y", "lam2_x", "lam2_y", "lam1_x", "lam1_y"]:
    #     reconciliation_args[key] = parameters[key]
    reconciliation_args = {
        "lam2_x": 1e-2,
        "lam2_y": 1e-1,
        "lam3_x": 1e-3,
        "lam3_y": 1e-1,
        "lam1_x": 1e-3,
        "lam1_y": 1e-3
    }
    raw_collection = "sanctimonious_beluga--RAW_GT1"
    rec_collection = "sanctimonious_beluga--RAW_GT1__test"
    raw = DBClient(**db_param, database_name = "trajectories", collection_name = raw_collection)
    rec = DBClient(**db_param, database_name = "reconciled", collection_name = rec_collection)
    
    ids = [
            ObjectId('62fd2a29b463d2b0792821c1'), # semi
            ObjectId('62fd2a2bb463d2b0792821c6'),# small top in front of semi
            ObjectId('62fd2a2eb463d2b0792821c9'),# small on top and behind semi
            ObjectId('62fd2a2ab463d2b0792821c3'), # small in front of c6
            ObjectId('62fd2a24b463d2b0792821b7'), # these two are double detection
            # ObjectId('62fd2a14b463d2b079282195') # 
           ] # raw

    
    # test_dbr = DBClient(**db_param, database_name = "reconciled", collection_name = "sanctimonious_beluga--RAW_GT1__administers")
    # ids = [ObjectId('62fd2a7dd913d95fd3282359'), ObjectId('62fd2a7dd913d95fd328235d')] # rec
    
    # docs = []
    # stitched_queue = mp_manager.Queue()
    
    # # for doc in raw.collection.find({"_id": {"$in": ids}}).sort("last_timestamp1",1):
    # start = raw.get_min("first_timestamp") + 35
    # end = start + 25
    # for doc in raw.collection.find({"direction": 1}).sort("last_timestamp",1):
    #     # doc = resample(doc)
    #     # docs.append(doc)
    #     if doc["first_timestamp"] > start and doc["first_timestamp"] < end:
    #         stitched_queue.put([doc])
            
    # print("total number of trajs: ", stitched_queue.qsize())
    # #       datetime.utcfromtimestamp(int(start)).strftime('%H:%M:%S'), 
    # #       datetime.utcfromtimestamp(int(end)).strftime('%H:%M:%S'))

    
    #%% prepare databases
    
    parameters["raw_collection"] = raw_collection
    parameters["reconciled_collection"] = rec_collection
    parameters["reconciliation_args"] = reconciliation_args
    raw_transformed = DBClient(**db_param, database_name = "temp", collection_name = parameters["raw_collection"])
    rec_transformed = DBClient(**db_param, database_name = "temp", collection_name = parameters["reconciled_collection"])
    
    # # print(raw_transformed.get_min("timestamp"), raw_transformed.get_max("timestamp"))
    
    # raw_transformed.collection.drop()
    # rec_transformed.collection.drop()
    # rec.collection.drop()
    
    #%% start a single process
    
    # preprocess_reconcile(stitched_queue, parameters, db_param) # 
    # solve_collision_avoidance_rolling(parameters, db_param)
    # postprocess_reconcile(parameters, db_param)
    
    
    #%% animate the rolling solver solution
    
    # query = rec_transformed.collection.find({}).sort("timestamp", 1)
    # snapshots = {time_doc["timestamp"]: time_doc["value"] for time_doc in query}
    # ani = animate_tracks(snapshots, offset=0, save=False, name="before_maxiter3")
    
    
    
    