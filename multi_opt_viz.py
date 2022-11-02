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
from collections import OrderedDict
import os
import json
from i24_database_api import DBClient
import time
import requests

dt = 0.04

class LRUCache:
    """
    A least-recently-used cache with integer capacity
    To roll out of the cache for vehicle color and dimensions
    get(): return the key-value in cache if exists, otherwise return -1
    put(): (no update) 
    """
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
 
    def get(self, key):
        if len(self.cache) > self.capacity:
            self.cache.popitem(last = False)
            
        try:
            self.cache.move_to_end(key)
            return self.cache[key]
        except KeyError: # key not in cache
            self.cache[key] = np.random.rand(3,)*0.8 # already at end when creating
            return self.cache[key]
 
    # def put(self, key, value, update = False):
    #     if key not in self.cache: # do not update with new value
    #         self.cache[key] = value
    #     elif update:
    #         self.cache[key] = value
    #     self.cache.move_to_end(key)
        
            
            
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



def animate_collection(dbc, database_name="transformed_beta", collection_name = None, offset = 0, duration=None, save=False, upload=True, extra=""):
    '''
    resample tracks, make to the dataframe, join the dataframe to get the overlapped time
    make an animation
    '''
    # check if collection exists in database
    if not collection_name:
        print("collection_name has to be specified")
        return
    
    if collection_name not in dbc.client[database_name].list_collection_names():
        print(f"{collection_name} not in {database_name}. Start transform")
        if "__" in collection_name:
            read_database_name = "reconciled"
        else:
            read_database_name = "trajectories"
        dbc.transform2(read_database_name=read_database_name, read_collection_name=collection_name,
                      write_database_name=database_name, write_collection_name=collection_name)
        
    # time-based
    dbc.db = dbc.client[database_name]
    dbc.collection = dbc.client[database_name][collection_name]
    
    # traj-based
    if "__" not in collection_name:
        veh = DBClient(**db_param, database_name = "trajectories", collection_name = collection_name)
    else:
        veh = DBClient(**db_param, database_name = "reconciled", collection_name = collection_name)
    
    # animate it!
    fig, ax = plt.subplots(1,1, figsize=(25,5))
    
    # get plotting ranges
    t_min = dbc.get_min("timestamp")
    t_max = dbc.get_max("timestamp")
    
    if offset:
        t_min += offset  
    if duration: 
        t_max = min(t_max, t_min + duration)
        
    # time_cursor = dbc.collection.find({}).sort("timestamp",1)
    time_cursor = dbc.get_range("timestamp", t_min, t_max)
    start_time = dbc.get_min("timestamp")
    color_cache = LRUCache(500)

    # init
    # OVERHEAD VIEW SETUP
    ax.set_title(collection_name)
    ax.set_aspect('equal', 'box')
    # ax.set(xlim=[4000,6000])
    # ax.set(ylim=[0,120])
    # ax.set(ylim=[-80, 80])
    try:
        ax.set(xlim=[veh.get_min("starting_x"), veh.get_max("ending_x")])
    except:
        pass
    
    ax.set_ylabel("EB         WB")
    ax.set_xlabel("Distance in feet")
        
    # plot lanes on overhead view
    # for j in range(-1, 12):
    #     if j in (-1, 5, 11):
    #         ax.axhline(y=j*12, linewidth=0.5, color='k')
    #     else:
    #         ax.axhline(y=j*12, linewidth=0.1, color='k')

    for j in range(-6, 7):
        if j in (-6, 0, 6):
            ax.axhline(y=j*12, linewidth=0.5, color='k')
        else:
            ax.axhline(y=j*12, linewidth=0.1, color='k')

    def animate(i):
        # plot the ith row in df3
        # remove all car_boxes 
        time_doc = time_cursor.next()
        t = time_doc["timestamp"] 
        if time_doc["timestamp"] > start_time + offset:
        
            time_text = datetime.utcfromtimestamp(int(t)).strftime('%m/%d/%Y, %H:%M:%S')
            plt.suptitle(time_text, fontsize = 20)
            
            # remove patch collections
            for col in ax.collections:
                col.remove()
            
            # pos = [centerx, centery, l, w, dir,v]
            try:
                eb_bbx = [patches.Rectangle(xy=(pos[0]-0.5*pos[2], pos[1]-0.5*pos[3]), width=pos[2], height=pos[3]) for _,pos in time_doc["eb"].items()]
            except KeyError:
                eb_bbx = []
            try:
                wb_bbx = [patches.Rectangle(xy=(pos[0]-0.5*pos[2], pos[1]-0.5*pos[3]), width=pos[2], height=pos[3]) for _,pos in time_doc["wb"].items()]
            except KeyError:
                wb_bbx = []
                
            # boxes = [patches.Rectangle(xy=(pos[0]-pos[4]*0.5*pos[2], pos[1]-0.5*pos[3]), width=pos[2], height=pos[3]) for _, pos in snapshot.items()]
            try:
                ce = [color_cache.get(_id) for _id in time_doc["eb"]]
            except KeyError:
                ce = []
            try:
                cw = [color_cache.get(_id) for _id in time_doc["wb"]]
            except KeyError:
                cw = []
            
            pc = PatchCollection(eb_bbx+wb_bbx, alpha=1,
                             # color=["red"]*len(eb_bbx)+["blue"]*len(wb_bbx)
                             color=ce+cw
                             )
            ax.add_collection(pc)
            return ax,
        
    # Init only required for blitting to give a clean slate.
    def init():
        return ax,
    
    frame = None
    anim = animation.FuncAnimation(fig, func=animate,
                                        init_func= init,
                                        frames=time_cursor,
                                        repeat=False,
                                        interval=0.04*1000, # in ms
                                        fargs=(frame ),
                                        blit=False,
                                        cache_frame_data = False,
                                        save_count = 1)
    if save:
        file_name = collection_name+ ".mp4"
        anim.save(file_name, writer='ffmpeg', fps=25)
        # self.anim.save('{}.gif'.format(file_name), writer='imagemagick', fps=self.framerate)
        print("saved.")
        
    # if save: # TODO: only one frame is saved
    #     now = datetime.utcfromtimestamp(int(time.time())).strftime('%Y-%m-%d_%H-%M-%S')
    #     file_name = now+"_" + collection_name +extra+".mp4"
    #     print(file_name)
    #     anim.save(file_name, writer='ffmpeg', fps=25)
    #     # self.anim.save('{}.gif'.format(file_name), writer='imagemagick', fps=self.framerate)
    #     print("saved.")
        
    #     if upload:
    #         url = 'http://viz-dev.isis.vanderbilt.edu:5991/upload?type=video'
    #         files = {'upload_file': open(file_name,'rb')}
    #         ret = requests.post(url, files=files)
    #         if ret.status_code == 200:
    #             print('Uploaded!')
    
    print("complete")
    
    # fig.tight_layout()
    # plt.show()
    return anim


    
    
if __name__=="__main__":
    with open(os.path.join(os.environ["USER_CONFIG_DIRECTORY"], "db_param.json")) as f:
        db_param = json.load(f)
    
    dbc = DBClient(**db_param)
    ani = animate_collection(dbc, database_name="transformed_beta", 
                             collection_name = "635997ddc8d071a13a9e5293", 
                             offset=0, duration=1500, save=False)
    
    
    
    
    