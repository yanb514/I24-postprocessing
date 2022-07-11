#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:22:38 2022

@author: yanbing_wang

- Make time space range selection automatic upon object construction
- listener to the database?
- add trajectory shape information
- seperate lanes
- put it on the video wall
- run tracking_v1 and reconciled side by side
"""

from i24_database_api.db_reader import DBReader
from i24_configparse import parse_cfg
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.animation as animation
import matplotlib.ticker as mticker
from datetime import datetime
from i24_logger.log_writer import logger, catch_critical
import warnings


class SpaceTimePlot():
    """
    Create a time-space diagram for a specified time-space window
    Query from database
    Plot on matplotlib
    """
    
    def __init__(self, config, collection_name, window_size = 10):
        """
        Initializes an SpaceTimePlot object
        
        Parameters
        ----------
        config : object
        """
        self.dbr = DBReader(config, collection_name=collection_name)
        self.anim = None
        self.collection_name = collection_name
        self.window_size = window_size
        
        
    @catch_critical(errors = (Exception))
    def get_collection_info(self):
        """
        To set the bounds on query and on visualization
        Returns
        -------
        None.

        """
        dbr = self.dbr
        collection_info = {
            "count": dbr.count(),
            "tmin": dbr.get_min("first_timestamp"),
            # "tmax": dbr.get_max("last_timestamp"),
            "tmax": dbr.get_min("first_timestamp") + 60,
            "xmin": min(dbr.get_min("starting_x"), dbr.get_min("ending_x"),dbr.get_max("starting_x"), dbr.get_max("ending_x")),
            # "xmin": -100,
            "xmax": max(dbr.get_max("starting_x"), dbr.get_max("ending_x"),dbr.get_min("starting_x"), dbr.get_min("ending_x")),
            "ymin": -5,
            "ymax": 200
            }
        self.lanes = [i*12 for i in range(-1,12)]
        # self.lanes = [-500, 0,12,24,36,48,60,72]
        self.lane_name = ["EBRS", "EB4", "EB3", "EB2", "EB1", "EBLS", "WBLS", "WB1", "WB2", "WB3", "WB4", "WBRS"]
        self.left = collection_info["tmin"]
        self.right = self.left + self.window_size
        self.collection_info = collection_info
        
        
    @catch_critical(errors = (Exception))
    def animate(self, tmin=None, tmax=None, increment=0.3, save = False):
        """
        Advance time window by delta second, update left and right pointer, and cache
        """     
        # Initialize ax
        # TODO: initialize time range
        print("in animate")
        self.get_collection_info()
        if not tmin:
            tmin = self.collection_info["tmin"]
        if not tmax:
            tmax = self.collection_info["tmax"]
        steps = np.arange(tmin, tmax, increment, dtype=float)
        
        # set figures: two rows. Top: east, bottom: west. 4 lanes in each direction
        fig, axs = plt.subplots(2,6,figsize=(30,8))
        plt.gcf().autofmt_xdate()
        # ax.set_aspect('equal', 'box')

        for i,row in enumerate(axs):
            for j, ax in enumerate(row):
                ax.set_aspect("auto")
                ax.set(ylim=[self.collection_info["xmin"], self.collection_info["xmax"]])
                ax.set(xlim=[self.left, self.right])
                axs[i,j].set_title(self.lane_name[i*6+j])
                
                labels = ax.get_xticks()
                labels = [datetime.utcfromtimestamp(int(t)).strftime('%H:%M:%S') for t in labels]
                ax.set_xticklabels(labels)
                

        # frame_text = ax1.text(max(ax1.get_xlim()), max(ax1.get_ylim()), "range {:.2f}-{:.2f}".format(steps[0], steps[1]), fontsize=12)
        
        # initialize qeury
        traj_data_e = self.dbr.read_query(query_filter= { "first_timestamp" : {"$gte" : self.left, "$lt" : self.right}, "direction": {"$eq": 1}},
                                        query_sort = [("last_timestamp", "ASC")])
        traj_data_w = self.dbr.read_query(query_filter= { "first_timestamp" : {"$gte" : self.left, "$lt" : self.right}, "direction": {"$eq": -1}},
                                        query_sort = [("last_timestamp", "ASC")])
        
        
        @catch_critical(errors = (Exception))
        def init():
            return axs
              

        @catch_critical(errors = (Exception))
        def update_cache(i, traj_data_e, traj_data_w, frame_text):
            """
            Returns
            -------
            delta : increment in time (sec)
                DESCRIPTION.
            """
            # roll time window forward
            self.left = steps[i]
            old_right = self.right
            self.right = self.left + self.window_size
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for i,row in enumerate(axs):
                    for j, ax in enumerate(row):
                        ax.set_aspect("auto")
                        ax.set(xlim=[self.left, self.right])
                        axs[i,j].set_title(self.lane_name[i*6+j])
                        labels = ax.get_xticks()
                        labels = [datetime.utcfromtimestamp(int(t)).strftime('%H:%M:%S') for t in labels]                
                        ax.set_xticklabels(labels)
                    
            i += 1
            
            # re-query for those whose first_timestamp is in the incremented time window
            traj_data_e = self.dbr.read_query(query_filter= { "first_timestamp" : {"$gte" : old_right, "$lt" : self.right},  "direction": {"$eq": 1}},
                                            query_sort = [("last_timestamp", "DSC")])
            traj_data_w = self.dbr.read_query(query_filter= { "first_timestamp" : {"$gte" : old_right, "$lt" : self.right},  "direction": {"$eq": -1}},
                                            query_sort = [("last_timestamp", "DSC")])

        
            
            # remove trajectories whose last_timestamp is below left
            # lines are ordered by DESCENDING last_timestamp
            axs_flatten = [ax for row in axs for ax in row]
            for ax in axs_flatten: # first row
                for line in ax.get_lines():
                    if line.get_xdata()[-1] < self.left:
                        line.remove()

            # add new lines east
            for traj in traj_data_e:
                # select sub-document for each lane
                lane_idx = np.digitize(traj["y_position"], self.lanes)-1 # should be between 1-6
                # print("east", traj["y_position"][:5])
                # print(lane_idx[:5])
                for idx in np.unique(lane_idx):
                    # print("east lane idx, ", idx)
                    select = lane_idx == idx # select only lane i
                    time = np.array(traj["timestamp"])[select]
                    x = np.array(traj["x_position"])[select]
                    try:
                        line, = axs[0,idx].plot(time, x)
                        mid = int(len(time)/2)
                        axs[0,idx].annotate(text="{}".format(traj["_id"]), xy=(time[mid],x[mid]))
                        print(traj["_id"])
                        
                    except:
                        print("lane idx {} is out of bound for EB".format(idx))
                        pass
            
            # add new lines west
            for traj in traj_data_w:
                # dx = np.diff(np.array(traj["x_position"]))
                # if len(np.unique(np.sign(dx))) > 1:
                    # try:
                    #     print(traj["_id"], len(traj["x_position"]), traj["fragment_ids"])
                        
                    # except:
                    #     print(traj["_id"])
                # print(traj["direction"])
                # select sub-document for each lane
                lane_idx = np.digitize(traj["y_position"], self.lanes)-1 # should be between 1-6
                # print("west", traj["y_position"][:5])
                # print(lane_idx[:5])
                for idx in np.unique(lane_idx):
                    # print("west lane idx, ", idx)
                    select = lane_idx == idx # select only lane i
                    time = np.array(traj["timestamp"])[select]
                    x = np.array(traj["x_position"])[select]
                    # dx = np.diff(np.array(traj["x_position"]))
                    # print(x)
                    try:
                        axs[1,idx-6].plot(time, x)
                        mid = int(len(time)/2)
                        axs[1,idx-6].annotate(text="{}".format(traj["_id"]), xy=(time[mid],x[mid]))
                        print(traj["_id"])
                    except:
                        print("lane idx {} out of bound west".format(idx-6))
                        pass
            return axs
        
        frame_text = None
        self.anim = animation.FuncAnimation(fig, func=update_cache,
                                            init_func= init,
                                            frames=len(steps),
                                            repeat=False,
                                            interval=increment * 1000, # in ms
                                            fargs=(traj_data_e, traj_data_w, frame_text), # specify time increment in sec to update query
                                            blit=False)
        self.paused = False

        fig.canvas.mpl_connect('button_press_event', self.toggle_pause_button)
        fig.canvas.mpl_connect('key_press_event', self.toggle_pause)
        
        if save:
            self.anim.save('{}.mp4'.format(self.collection_name), writer='ffmpeg', fps=int(1/increment))
            
        plt.show()
        print("complete")
        

    def toggle_pause(self, event):
        if event.key == " ":
            if self.paused:
                self.anim.resume()
                print("Animation Resumed")
            else:
                self.anim.pause()
                print("Animation Paused")
            self.paused = not self.paused
            
    def toggle_pause_button(self, *args, **kwargs):
        if self.paused:
            self.anim.resume()
        else:
            self.anim.pause()
        self.paused = not self.paused
        
        
    
if True and __name__=="__main__":
    
    
    config_path = os.path.join(os.getcwd(),"../config")
    os.environ["user_config_directory"] = config_path
    os.environ["my_config_section"] = "TEST"
    parameters = parse_cfg("my_config_section", cfg_name = "test_param.config")
    
    stp = SpaceTimePlot(parameters, "batch_5_07072022", window_size = 5)
    stp.animate(increment=0.1, save=True)
    
    