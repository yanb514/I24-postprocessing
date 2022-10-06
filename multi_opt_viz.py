#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 09:49:18 2022

@author: yanbing_wang
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np
from datetime import datetime


dt = 0.04

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
    
  