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
from cvxopt import matrix, solvers, sparse,spdiag,spmatrix
import time
from scipy.sparse import identity, coo_matrix, block_diag,vstack, hstack, csr_matrix,lil_matrix
from utils.utils_opt import combine_fragments, resample, _blocdiag
import pandas as pd
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

from itertools import combinations
from collections import defaultdict, OrderedDict

dt = 1/30

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
        # print(t,x)
        # ax[0].scatter(t, x, label="somthing")
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
    
def animate_tracks(snapshots, save=False, name="before"):
    '''
    resample tracks, make to the dataframe, join the dataframe to get the overlapped time
    make an animation
    '''

    # animate it!
    fig, ax = plt.subplots(1,1, figsize=(25,5))

    ax.set(ylim=[0,120])
    ax.set(xlim=[0,2000])
    ax.set_aspect('equal', 'box')

    def animate(i):
        # plot the ith row in df3
        # remove all car_boxes 

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

    frames = sorted(snapshots.keys()) # sort time index
    dt = frames[1]-frames[0]
    anim = animation.FuncAnimation(fig, func=animate,
                                        init_func= init,
                                        frames=frames,
                                        repeat=False,
                                        interval=dt*1000*0.8, # in ms
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

    
def solve_collision_avoidance(tracks, lam2_x, lam2_y, lam3_x, lam3_y, lam1_x, lam1_y):
    '''
    minimize opt2_l1
    s.t. |Oixi - Ojxj| >= cx, OR
         |Oiyi - Ojyj| >= cy
    convert the constraints to
    s.t. cx - |Oixi - Ojxj| <= M*b
         cy - |Oiyi - Ojyj| <= M*(1-b), 
    where b is a binary variable
    '''
    objx_arr = []
    objy_arr = []
    x_arr = []
    y_arr = []
    t_arr = []
    zx_arr = []
    zy_arr = []

    
    for track in tracks:
        track = resample(track)
        datax = track["x_position"]
        datay = track["y_position"]
        
        N = len(datax)
        idx = [i.item() for i in np.argwhere(~np.isnan(datax)).flatten()]
        zx = np.array(datax)[idx] # measurement
        zy = np.array(datay)[idx] # measurement
        M = len(zx)
        
        D1 = _blocdiag_scipy(coo_matrix([-1,1]), N) * (1/dt)
        D2 = _blocdiag_scipy(coo_matrix([1,-2,1]), N) * (1/dt)
        D3 = _blocdiag_scipy(coo_matrix([-1,3,-3,1]), N) * (1/dt)
        I = identity(N).toarray()
        H = I[idx,:]
        
        x = cp.Variable(N)
        ex = cp.Variable(M)
        y = cp.Variable(N)
        ey = cp.Variable(M)
        
        obj1 = 1/M * cp.sum_squares(zx- H@x - ex) + lam2_x/(N-2) * cp.sum_squares(D2 @ x) + lam3_x/(N-3) * cp.sum_squares(D3 @ x) + lam1_x/M * cp.norm(ex, 1)
        obj2 = 1/M * cp.sum_squares(zy- H@y - ey) + lam2_y/(N-2) * cp.sum_squares(D2 @ y) + lam3_y/(N-3) * cp.sum_squares(D3 @ y) + lam1_y/M * cp.norm(ey, 1)
    
        t_arr.append(track["timestamp"])
        zx_arr.append(zx)
        zy_arr.append(zy)
        x_arr.append(x)
        y_arr.append(y)
        objx_arr.append(obj1)
        objy_arr.append(obj2)

    # interaction term
    s1, e1, s2, e2 = find_overlap_idx(t_arr[0], t_arr[1])
    O1 = hstack([csr_matrix((e1-s1+1, s1)), identity(e1-s1+1), csr_matrix((e1-s1+1, len(t_arr[0])-e1-1))])
    O2 = hstack([csr_matrix((e2-s2+1, s2)), identity(e2-s2+1), csr_matrix((e2-s2+1, len(t_arr[1])-e2-1))])
    dist_square = cp.square(O1 @ x_arr[0] - O2 @ x_arr[1]) + cp.square(O1 @ y_arr[0] - O2 @ y_arr[1])+1
    # dist_cost = 1/(dist_norm/(e1-s1+1)) # 1/MAE
    
    y_scale = 1
    constraints = [
        # (x_arr[0][s1+t]- x_arr[1][s2+t])**2 +  (y_arr[0][s1+t]- y_arr[1][s2+t])**2 >= 10**2 for t in range(e1-s1)
        # O1 @ x_arr[0] >= 100
        # cp.norm(O1 @ x_arr[0] - O2 @ x_arr[1]) >= 100
        # dist_square >= 100 # not DCP
        O2 @ y_arr[1]-4 - O1 @ y_arr[0]-5 >= 1 # lateral gap
        ] # should have (e1-s1) number of constraints
    
    # for const in constraints:
    #     print(const.is_qdcp())
    # solve jointly 
    obj = objx_arr[0] + objx_arr[1] + y_scale * (objy_arr[0] + objy_arr[1]) - cp.sum(cp.log(dist_square))
    prob1 = cp.Problem(cp.Minimize(obj))
    prob1.solve(warm_start=True)
     
    print("Status: ", prob1.status)
    print("The optimal value is", prob1.value)
    # print("A solution x is")
    # print(x.value)

    tracks[0]["timestamp"] = t_arr[0]
    tracks[0]["x_position"] = x_arr[0].value
    tracks[0]["y_position"] = y_arr[0].value
    
    tracks[1]["timestamp"] = t_arr[1]
    tracks[1]["x_position"] = x_arr[1].value
    tracks[1]["y_position"] = y_arr[1].value
    
    print( "dist y: ", np.min(O2 @ y_arr[1].value - O1 @ y_arr[0].value))
    return tracks
    
    
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
    
    
def solve_collision_avoidance2(tracks, lam2_x, lam2_y, lam3_x, lam3_y, lam1_x, lam1_y):
    '''
    minimize opt2_l1
    s.t.  AX(t) + b <= beta(t), forall t
          ||beta(t)||1 >= 1 # disjunctive constraints at each t
          beta(n,t) \in {0,1}
    A: (4x4), X: (4xK), beta:(4xK), xi,yi:(1xNi), xj,yj:(1xNj), Oi:(KxNi), Oj:(KxNj)
    
    where beta is a binary variable
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
    
    idxi = np.argwhere(~np.isnan(zxi)).flatten()
    idxj = np.argwhere(~np.isnan(zxj)).flatten()
    
    # zx = zxi[idx] # measurement
    # zy = np.array(datay)[idx] # measurement
    Mi = len(idxi)
    Mj = len(idxj)
    
    Ni = len(zxi)
    Nj = len(zxj)
    
    s1, e1, s2, e2 = find_overlap_idx(ti, tj)
    K = e1-s1+1
    xi, xj, yi, yj = cp.Variable(Ni), cp.Variable(Nj), cp.Variable(Ni), cp.Variable(Nj) # decision variables composed of xi,xj,yi,yj
    Oi = hstack([csr_matrix((K, s1)), identity(K), csr_matrix((K, len(ti)-e1-1))])
    Oj = hstack([csr_matrix((K, s2)), identity(K), csr_matrix((K, len(tj)-e2-1))])
    
    # decision variables
    X = cp.vstack([Oi@xi, Oj@xj, Oi@yi, Oj@yj]) # regroup selected decision variables for the time-overlapped part
    DX = cp.vstack([Oi@xi- Oj@xj, Oi@yi-Oj@yj]) # 
    E = [cp.Variable(Mi), cp.Variable(Mj), cp.Variable(Mi), cp.Variable(Mj)] # all the outliers of xi,xj,yi,y
    
    # get objective functions for tracki
    D1 = _blocdiag_scipy(coo_matrix([-1,1]), Ni) * (1/dt)
    D2 = _blocdiag_scipy(coo_matrix([1,-2,1]), Ni) * (1/dt)
    D3 = _blocdiag_scipy(coo_matrix([-1,3,-3,1]), Ni) * (1/dt)
    I = identity(Ni).toarray()
    Hi = I[idxi,:]
    vi = tracks[0]["direction"]-D1@xi 
    ai = D2@xi
    ji = D3@xi
    
    # E[0] is the first row of E
    objxi = 1/Mi * cp.sum_squares(zxi[idxi]- Hi@xi - E[0]) + lam2_x/(Ni-2) * cp.sum_squares(D2 @ xi) + lam3_x/(Ni-3) * cp.sum_squares(D3 @ xi) + lam1_x/Mi * cp.norm(E[0], 1)
    objyi = 1/Mi * cp.sum_squares(zyi[idxi]- Hi@yi - E[2]) + lam2_y/(Ni-2) * cp.sum_squares(D2 @ yi) + lam3_y/(Ni-3) * cp.sum_squares(D3 @ yi) + lam1_y/Mi * cp.norm(E[2], 1)
    
    # get objective functions for trackj
    D1 = _blocdiag_scipy(coo_matrix([-1,1]), Nj) * (1/dt)
    D2 = _blocdiag_scipy(coo_matrix([1,-2,1]), Nj) * (1/dt)
    D3 = _blocdiag_scipy(coo_matrix([-1,3,-3,1]), Nj) * (1/dt)
    I = identity(Nj).toarray()
    Hj = I[idxj,:]
    vj = tracks[1]["direction"]-D1@xj
    aj = D2@xj
    jj = D3@xj
    
    objxj = 1/Mj * cp.sum_squares(zxj[idxj]- Hj@xj - E[1]) + lam2_x/(Nj-2) * cp.sum_squares(D2 @ xj) + lam3_x/(Nj-3) * cp.sum_squares(D3 @ xj) + lam1_x/Mj * cp.norm(E[1], 1)
    objyj = 1/Mj * cp.sum_squares(zyj[idxj]- Hj@yj - E[3]) + lam2_y/(Nj-2) * cp.sum_squares(D2 @ yj) + lam3_y/(Nj-3) * cp.sum_squares(D3 @ yj) + lam1_y/Mj * cp.norm(E[3], 1)
    
    # collision penalty
    # dist = cp.sum(cp.log(cp.norm(DX,2, axis=0))) # sum of quasiconcave is not quasiconcave
    # dist = cp.log(cp.min(cp.norm(DX,2, axis=0))) # nope
    # constraints

    big_M = 1e6
    Y = np.vstack([Oi@ zxi, Oj@zxj, Oi@zyi, Oj@zyj]) # all the data of the overlapped time
    A = np.array([[-1, 1, 0, 0],
                  [1, -1, 0, 0],
                  [0, 0, 1, -1],
                  [0, 0, -1, +1]])
    padx, pady = 5, 1
    b = np.array([lj + padx, li + padx, 0.5*(wi+wj) + pady, 0.5*(wi+wj) + pady])
    
    conf = A @ Y + np.tile(b, (K,1)).T # 4 x K
    
    #TODO: fillnan with interpolation
    fcn = lambda z: z.nonzero()[0]
    nans = np.isnan(conf[1,:])
    for i in range(conf.shape[0]):
        conf[i,nans]= np.interp(fcn(nans), fcn(~nans), conf[i,~nans])
    
    # resolve conflict by pushing to the "best" direction
    # take the min of each column
    mincol = np.argmin(conf, axis=0)
    confb = conf < 0 # binary matrix
    conf_time = np.where(~np.any(confb, axis=0))[0] # select time where all are false (conflicts occur)
    confb[mincol[conf_time], conf_time] = True # flip those to True -> indicating the flipped bisection is the direction to pull apart 4xK
    
    # X = np.vstack([Oi@xi, Oj@xj, Oi@yi, Oj@yj]) # all the decision variables at the overlapped time
    # LHS = A @ X + np.tile(b, (K,1)).T
    RHS = big_M*(1-confb*1)
    constraints = [
         # LHS[i,j] <= RHS[i,j] for i in range(4) for j in range(K) # takes long
         # LHS[i,j] <= 0 for i in mincol[conf_time] for j in conf_time# takes even longer to even compile the problem
         -Oi@xi + Oj@xj + lj+padx <= RHS[0,:],
         Oi@xi - Oj@xj + li+padx <= RHS[1,:],
         Oi@yi - Oj@yj + 0.5*(wi+wj)+pady <= RHS[2,:],
         -Oi@yi + Oj@yj + 0.5*(wi+wj)+pady <= RHS[3,:],
         vi <= 0,
         vj <= 0,
         # ai <= 10, # acceleration and jerk constraints fail QSQP, and CVXOPT solver is slower
         # -ai <= 10,
         # aj <= 10,
         # -aj <= 10
        ]
    
    # combine to a problem
    obj = objxi + objxj + objyi + objyj #+ penalty # may need to scale y
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(verbose=True) #solver="ECOS_BB" runs forever, solver="SCIP" takes forever, qcp=True if problem is DQCP
     
    print("Status: ", prob.status)
    print("The optimal value is", prob.value)
    # print("A solution x is")
    # print(x.value)

    tracks[0]["timestamp"] = ti
    tracks[0]["x_position"] = xi.value
    tracks[0]["y_position"] = yi.value
    
    tracks[1]["timestamp"] = tj
    tracks[1]["x_position"] = xj.value
    tracks[1]["y_position"] = yj.value
    # print(np.min(t.value))

    return tracks

def configuration(p1, p2):
    '''
    p1, p2: [centerx, cnetery, l, w]
    return configuration
    [1 is behind 2, 1 is in front of 2, 1 is above 2, 1 is below 2]
    '''
    xpad, ypad = 5,1
    
    return [p1[0]+0.5*p1[2]+xpad<p2[0]-0.5*p2[2], # x1>>x2
            p1[0]-0.5*p1[2]-xpad>p2[0]+0.5*p2[2], # x2>>x1
            p1[1]-0.5*p1[3]-ypad>p2[1]+0.5*p2[3], # y1>>y2
            p1[1]+0.5*p1[3]+ypad<p2[1]-0.5*p2[3]] # y2>>y1
    
def separatum(snapshots):
    '''
    snpashot = {
        i:[centerx, centery, l, w],
        j:[...],
        ...
        }
    beta is non-overlap configuration for all overalpped timestamps
    beta = {
        t1: {
            (i,j): [T,T,F,F,],
            (i,k): []
            },
        t2: {...}..
        }
    beta = { # this schema is easier to build constraints
        (i,j): { "overlap_t": [t1,t2,...],
                "configuration": [[T,T,F,F], [F,F,T,T],...]
                }
        
        }
    '''
    scale = 0.1
    beta = defaultdict(dict) # TODO: consider only two tracks for now
    for t, snapshot in snapshots.items():
        overlap = True
        while overlap:
            overlap = False
            vec = {key: np.array([0,0]) for key in snapshot}
            for i,j in combinations(snapshot.keys(), 2):
                conf = configuration(snapshot[i], snapshot[j])
                if not any(conf): # if all are true
                    overlap = True # reset
                    # TODO: this vector could be based on iou shape
                    vector = snapshot[i][:2] - snapshot[j][:2]
                    vec[i] = vec[i] + scale * vector
                    vec[j] = vec[j] - scale * vector
                else:
                    beta[t][(i,j)] = conf
            # move them! 
            for i, v in vec.items():
                snapshot[i][:2] += v

    return snapshots, beta
            
def time_transform(tracks, dt=0.1):
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
            lru[_id]["x"].append(t)
            lru[_id]["y"].append(t)
            lru[_id]["l"].append(t)
            lru[_id]["w"].append(t)
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
 
    

def _build_obj_fcn(tracks, beta, lam2_x, lam2_y, lam3_x, lam3_y, lam1_x, lam1_y):
    '''
    X = [xi, xj, xk, ..., yi, yj, yk, ...]
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
    for track in tracks:
        n = len(track["t"])
        dt = track["t"][1]-track["t"][0]
        zx = track["x"]
        zy = track["y"]
        x = cp.Variable(n)
        y = cp.Variable(n)
        ex = cp.Variable(n)
        ey = cp.Variable(n)
        # X.append()
        # Y.append(cp.Variable(n))
        # get objective functions for tracki
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
    num_cars = len(tracks)
    for i,j in combinations(num_cars, 2):
        tracki, trackj = tracks[i], tracks[j]
        beta_ij = beta[(tracki["_id"], trackj["_id"])]
        s,t = beta_ij["overlap_t"][0], beta_ij["overlap_t"][-1] # start, end of the overlap time
        si, ei, sj, ej = find_overlap_idx(tracki["t"], trackj["t"])
        
        
        
        
    # all decision variables and all obj functions
    return X,Y,EX,EY,OBJX,OBJY

def _build_constr(X,Y,beta,idList):
    '''
    X = [xi, xj, ...]
    beta is non-overlap configuration for all overalpped timestamps
    beta = {
        t1: {
            (i,j): [T,T,F,F,],
            (i,k): []
            },
        t2: {...}..
        }
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
    
        
    return constraints

def solve_collision_avoidance3(tracks, lam2_x, lam2_y, lam3_x, lam3_y, lam1_x, lam1_y):
    
    # resolve conflict first -> may need to iterate
    snapshots = time_transform(tracks, dt=0.04)
    snapshots, beta = separatum(snapshots)
    tracks_s = snapshots_2_tracks(snapshots)
    
    # get decision vars and objective functions
    X,Y,EX,EY,OBJX,OBJY = _build_obj_fcn(tracks_s, lam2_x, lam2_y, lam3_x, lam3_y, lam1_x, lam1_y)
    constr = _build_constr(X,Y, beta, [track["_id"] for track in tracks])
    
    prob.solve()
    
    # modify tracks
    return


if __name__ == '__main__':
    
    # initialize parameters
    # with open("config/parameters.json") as f:
    #     parameters = json.load(f)
        
    with open(os.path.join(os.environ["USER_CONFIG_DIRECTORY"], "db_param.json")) as f:
        db_param = json.load(f)
        
    
    # for key in ["lam3_x","lam3_y", "lam2_x", "lam2_y", "lam1_x", "lam1_y"]:
    #     reconciliation_args[key] = parameters[key]
    reconciliation_args = {
        "lam2_x": 1e-3,
        "lam2_y": 1e-1,
        "lam3_x": 1e-4,
        "lam3_y": 1e-1,
        "lam1_x": 1e-3,
        "lam1_y": 1e-3
    }
    
    test_dbr = DBClient(**db_param, database_name = "trajectories", collection_name = "sanctimonious_beluga--RAW_GT1")
    ids = [ObjectId('62fd2a29b463d2b0792821c1'), ObjectId('62fd2a2bb463d2b0792821c6')] # raw
    
    # test_dbr = DBClient(**db_param, database_name = "reconciled", collection_name = "sanctimonious_beluga--RAW_GT1__administers")
    # ids = [ObjectId('62fd2a7dd913d95fd3282359'), ObjectId('62fd2a7dd913d95fd328235d')] # rec
    
    docs = []

    for doc in test_dbr.collection.find({"_id": {"$in": ids}}):
        # doc = resample(doc)
        docs.append(doc)
        
    
    # %% time transform
    snapshots = time_transform(docs, dt=0.2)
    snapshots, beta = separatum(snapshots)
    # anim = animate_tracks(snapshots, save=False)
    
    #%% plot
    # plot_track(docs)
    # for doc in docs:
    #     doc = resample(doc)
    
    
    
    # conf, confb = plot_configuration(docs)
    
    # #%% solve
    # for doc in docs:
    #     doc = resample(doc)
    # conf, confb = plot_configuration(docs)
    # plot_track(docs)
    # ani = animate_tracks(docs, save=True, name="before")
    # rec_docs = solve_collision_avoidance2(docs, **reconciliation_args)
    # ani = animate_tracks(docs, save=True, name="after")
    

    # conf, confb = plot_configuration(docs)
    # plot_track(rec_docs)
    # 
    
    