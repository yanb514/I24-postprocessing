#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 09:49:18 2022

@author: yanbing_wang

Visualize with the new transformed schema

"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np
from datetime import datetime
import os
import json
from i24_database_api import DBClient

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

def animate_collection(dbc, offset = 0, save=False, name="before"):
    '''
    resample tracks, make to the dataframe, join the dataframe to get the overlapped time
    make an animation
    '''

    # animate it!
    fig, ax = plt.subplots(1,1, figsize=(25,5))

    ax.set(ylim=[0,120])
    ax.set(xlim=[0,2000])
    ax.set_aspect('equal', 'box')
    
    # start_time = min(snapshots.keys())
    # frames = sorted([key for key in snapshots.keys() if key > start_time+offset]) # sort time index
    # dt = frames[1]-frames[0]
    # print("dt: ",dt)
    
    time_cursor = dbc.collection.find({}).sort("timestamp",1)
    start_time = dbc.get_min("timestamp")

    def animate(i):
        # plot the ith row in df3
        # remove all car_boxes 
        time_doc = time_cursor.next()
        t = time_doc["timestamp"] 
        if time_doc["timestamp"] > start_time + offset:
        
            time_text = datetime.utcfromtimestamp(int(t)).strftime('%m/%d/%Y, %H:%M:%S')
            plt.suptitle(time_text, fontsize = 20)
            
            for pc in ax._children:
                pc.remove()
            
            # pos = [centerx, centery, l, w, dir,v]
            eb_bbx = [patches.Rectangle(xy=(pos[0]-pos[4]*0.5*pos[2], pos[1]-0.5*pos[3]), width=pos[2], height=pos[3]) for _,pos in time_doc["eb"].items()]
            wb_bbx = [patches.Rectangle(xy=(pos[0]-pos[4]*0.5*pos[2], pos[1]-0.5*pos[3]), width=pos[2], height=pos[3]) for _,pos in time_doc["wb"].items()]
            
            # boxes = [patches.Rectangle(xy=(pos[0]-pos[4]*0.5*pos[2], pos[1]-0.5*pos[3]), width=pos[2], height=pos[3]) for _, pos in snapshot.items()]
            pc = PatchCollection(eb_bbx+wb_bbx, alpha=1,
                             edgecolor="blue")
            ax.add_collection(pc)
            return ax,
        
    # Init only required for blitting to give a clean slate.
    def init():
        print("init")
        return ax,

    anim = animation.FuncAnimation(fig, func=animate,
                                        init_func= init,
                                        frames=time_cursor,
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


    
    
if __name__=="__main__":
    with open(os.path.join(os.environ["USER_CONFIG_DIRECTORY"], "db_param.json")) as f:
        db_param = json.load(f)
    
    dbc = DBClient(**db_param, database_name="transformed_beta", collection_name = "abandoned_wookie--RAW_GT2__lionizes")
    ani = animate_collection(dbc, offset=0, save=False, name="separatum_smooth_raw")
    
    
    
    