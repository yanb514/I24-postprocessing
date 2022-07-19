#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:22:38 2022

@author: yanbing_wang

TODOs
- clean up config
- increment vs. framerate?
- xticklabel does not show up
- if transformed collection is not available, plot time-space instead
"""

from i24_database_api.db_reader import DBReader
from i24_configparse import parse_cfg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import matplotlib.animation as animation
import matplotlib.ticker as mticker
from datetime import datetime
from i24_logger.log_writer import logger, catch_critical
import queue
import mplcursors
from collections import OrderedDict

 
class LRUCache:
 
    # initialising capacity
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
 
    # we return the value of the key
    # that is queried in O(1) and return -1 if we
    # don't find the key in out dict / cache.
    # And also move the key to the end
    # to show that it was recently used.
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        else:
            self.cache.move_to_end(key)
            return self.cache[key]
 
    # first, we add / update the key by conventional methods.
    # And also move the key to the end to show that it was recently used.
    # But here we will also check whether the length of our
    # ordered dictionary has exceeded our capacity,
    # If so we remove the first key (least recently used)
    def put(self, key: int, value: int) -> None:
        if key not in self.cache: # do not update with new value
            self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last = False)
            
            
            
            
class Plotter():
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
        # TODO: clean up config
        # combine get_collection_info with __init__
        # add default parameters
        
        self.dbr = DBReader(config, username = "readonly",  collection_name=collection_name)
        if collection_name not in self.dbr.db.list_collection_names():
            print(f"{collection_name} not in dbr")
            raise NameError
            
        # add default collection_name for transformed
        self.dbr_t = DBReader(config, username = "readonly",  database_name = parameters.transformed, collection_name=collection_name+"_transformed")
        if collection_name+"_transformed" not in self.dbr_t.db.list_collection_names():
            print(f"{collection_name}_transformed not in dbr_t")
            self.overhead = False # no overhead view
        else:
            self.overhead = True # intialize as a place holder
            
        self.anim = None
        self.collection_name = collection_name
        self.window_size = window_size
        
        self.lanes = [i*12 for i in range(-1,12)]   
        self.lane_name = [ "EBRS", "EB4", "EB3", "EB2", "EB1", "EBLS", "WBLS", "WB1", "WB2", "WB3", "WB4", "WBRS"]
        self.lane_idx = [i for i in range(12)]
        self.lane_ax = [[1,5],[1,4],[1,3],[1,2],[1,1],[1,0],[0,0],[0,1],[0,2],[0,3],[0,4],[0,5]]
        
        self.vl_queue = queue.Queue() # for updating vertical lines
        self.annot_queue = queue.Queue()
        self.cursor = None
        
        
        
        
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
            # "xmin": min(dbr.get_min("starting_x"), dbr.get_min("ending_x"),dbr.get_max("starting_x"), dbr.get_max("ending_x")),
            "xmin": -100,
            "xmax": max(dbr.get_max("starting_x"), dbr.get_max("ending_x"),dbr.get_min("starting_x"), dbr.get_min("ending_x")),
            "ymin": -5,
            "ymax": 200
            }
        
        self.left = collection_info["tmin"] - self.window_size/2
        self.right = self.left + self.window_size
        self.old_right = collection_info["tmin"]
        self.collection_info = collection_info
        
        self.x_start = self.collection_info["xmin"]
        self.x_end = self.collection_info["xmax"]
        
        
    @catch_critical(errors = (Exception))
    def animate(self, tmin=None, tmax=None, increment=0.3, frames=100, save = False):
        """
        Advance time window by delta second, update left and right pointer, and cache
        """     
        # Initialize ax
        self.get_collection_info()
        
        if not tmin:
            tmin = self.collection_info["tmin"]
        if not tmax:
            tmax = self.collection_info["tmax"]
        # steps = np.arange(tmin, tmax, increment, dtype=float)
        
        # set figures: two rows. Top: east, bottom: west. 4 lanes in each direction
        if self.overhead:
            fig, axs = plt.subplots(3,6,figsize=(34,8))
        else:
            fig, axs = plt.subplots(2,6,figsize=(30,8))
        
        # TODO: make size parameters
        cache_vehicle = LRUCache(100)
        cache_colors = LRUCache(100)
        
        
        # OVERHEAD VIEW SETUP
        if self.overhead:
            ax_o = plt.subplot(313) # overhead view
            ax_o.set_aspect('equal', 'box')
            ax_o.set(ylim=[self.lanes[0], self.lanes[-1]])
            ax_o.set(xlim=[self.x_start, self.x_end])
            ax_o.set_ylabel("EB    WB")
            ax_o.set_xlabel("Distance in feet")
              
            def on_xlims_change(event_ax):
                # print("updated xlims: ", event_ax.get_xlim())
                new_xlim = event_ax.get_xlim()
                # ax1.set(xlim=new_xlim)
                self.x_start = new_xlim[0]
                self.x_end = new_xlim[1]
            ax_o.callbacks.connect('xlim_changed', on_xlims_change)
            
        
        # TIME-SPACE VIEW SETUP
        self.time_cursor = self.dbr_t.collection.find().sort([("timestamp", 1)]).limit(0) # no limit
        plt.gcf().autofmt_xdate()
        
        for i in self.lane_idx:
            ax = axs[self.lane_ax[i][0], self.lane_ax[i][1]]
            ax.set_aspect("auto")
            ax.set(ylim=[self.collection_info["xmin"], self.collection_info["xmax"]])
            ax.set(xlim=[self.collection_info["tmin"]-self.window_size/2, self.collection_info["tmin"]+self.window_size/2])
            ax.set_title(self.lane_name[i])
            ax.yaxis.set_visible(False)
            if i <= 5: # bottom
                ax.set_xlabel("Time")
            if i in [5,6]: # left
                ax.set_ylabel("Distance in feet")
                ax.yaxis.set_visible(True)
            # TODO: labels don't show
            # labels = ax.get_xticks()
            # labels = [datetime.utcfromtimestamp(int(t)).strftime('%H:%M:%S') for t in labels]
            # ax.set_xticklabels(labels)
                    
        
        
        @catch_critical(errors = (Exception))
        def init():
            if self.overhead:
                # plot lanes on overhead view
                for i in range(-1, 12):
                    if i in (-1, 5, 11):
                        ax_o.axhline(y=i*12, linewidth=0.5, color='k')
                    else:
                        ax_o.axhline(y=i*12, linewidth=0.1, color='k')
            
            return axs,
              

        @catch_critical(errors = (Exception))
        def update_cache(frame_text):
            """
            Returns
            -------
            delta : increment in time (sec)
                DESCRIPTION.
            """
            if self.overhead:
                # --------------- OVERHEAD VIEW ---------------------
                doc = self.time_cursor.next()
                curr_time = doc["timestamp"]
                time_text = datetime.utcfromtimestamp(int(curr_time)).strftime('%m/%d/%Y, %H:%M:%S')
                ax_o.set_title(time_text)
                
                # remove all car_boxes and verticle lines
                for box in list(ax_o.patches):
                    box.set_visible(False)
                    box.remove()
                while not self.vl_queue.empty():
                    self.vl_queue.get(block=False).remove()
    
                while not self.annot_queue.empty():
                    self.annot_queue.get(block=False).remove()
                    
                # Add vehicle ids in cache_colors             
                for veh_id in doc['id']:
                    cache_colors.put(veh_id, np.random.rand(3,))
                    
                # query for vehicle dimensions if not in doc
                if "dimensions" not in doc:
                    traj_cursor = self.dbr.collection.find({"_id": {"$in": doc["id"]} }, 
                                                                    {"width":1, "length":1, "coarse_vehicle_class": 1})
                    # add vehicle dimension to cache
                    for index, traj in enumerate(traj_cursor):
                        # print("index: {} is {}".format(index, traj))
                        # { ObjectId('...') : [length, width, coarse_vehicle_class] }
                        
                        cache_vehicle.put(traj["_id"], [traj["length"], traj["width"], traj["coarse_vehicle_class"]])
                else:
                    for index, veh_id in enumerate(doc['id']):
                        cache_vehicle.put(veh_id, doc['dimensions'][index])
                
            
                # plot vehicles
                for index in range(len(doc["position"])):
                    car_x_pos = doc["position"][index][0]
                    car_y_pos = doc["position"][index][1]
    
                    car_length, car_width, _ = cache_vehicle.get(doc["id"][index])
                
                    
                    # print("index {} at ({},{})".format(index, car_x_pos, car_y_pos))
                    # if car_x_pos <= self.x_start and car_x_pos >= self.x_end:
                    # print(cache_colors.get(doc["id"][index]))
                    box = patches.Rectangle((car_x_pos, car_y_pos),
                                            car_length, car_width, 
                                            color=cache_colors.get(doc["id"][index]),
                                            # color = np.array([str_to_float(str(doc["id"])[i*8:i*8+8]) for i in range(3)]),
                                            label=doc["id"][index])
                    ax_o.add_patch(box)   
                    # add annotation
                    annot = ax_o.annotate(doc['_id'], xy=(car_x_pos,car_y_pos))
                    annot.set_visible(False)
                    self.annot_queue.put(annot)
                
                
            # --------------- TIME-SPACE VIS ---------------------
            # update time range
            for i in self.lane_idx:
                ax = axs[self.lane_ax[i][0], self.lane_ax[i][1]]
                ax.set(xlim=[self.left, self.right])
                # add vertical line
                if self.overhead:
                    vl = ax.axvline(x=curr_time, c='k', linewidth='0.5', linestyle='--')
                    self.vl_queue.put(vl)
                    
                # TODO: labels don't show?
                # labels = ax.get_xticks()
                # labels = [datetime.utcfromtimestamp(int(t)).strftime('%H:%M:%S') for t in labels]
                # ax.set_xticklabels(labels)
                ax.xaxis.set_major_locator(mticker.MaxNLocator(1))
                ticks_loc = ax.get_xticks().tolist()
                ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                labels = [datetime.utcfromtimestamp(int(t)).strftime('%H:%M:%S') for t in ticks_loc]
                ax.set_xticklabels(labels)
                        
                
            # re-query for those whose first_timestamp is in the incremented time window
            traj_data = self.dbr.read_query(query_filter= { "first_timestamp" : {"$gte" : self.old_right, "$lt" : self.right}},
                                            query_sort = [("last_timestamp", "DSC")])
            
            # roll time window forward
            if self.overhead:
                self.left = doc['timestamp'] - self.window_size/2
                self.old_right = self.right
                self.right = doc['timestamp'] + self.window_size/2
            else:
                self.old_right = self.right
                self.right += increment
                self.left = self.right - self.window_size
            
            # remove trajectories whose last_timestamp is below left
            # lines are ordered by DESCENDING last_timestamp
            axs_flatten = [ax for row in axs[:2] for ax in row]
            for ax in axs_flatten: # first row
                for line in ax.get_lines():
                    if line.get_xdata()[-1] < self.left:
                        line.remove()
                 
            
            # add trajectory lines, assign them to the corresponding lanes
            for traj in traj_data:
            # select sub-document for each lane
                lane_idx = np.digitize(traj["y_position"], self.lanes)-1 # should be between 6-11
                for idx in np.unique(lane_idx):
                    select = lane_idx == idx # select only lane i
                    time = np.array(traj["timestamp"])[select]
                    x = np.array(traj["x_position"])[select]
                    try:
                        # if traj["_id"] not in cache_colors:
                        #     cache_colors[traj["_id"]] = np.random.rand(3,)
                        cache_colors.put(traj["_id"], np.random.rand(3,))
                        # color = np.array([str_to_float(str(traj["_id"])[i*8:i*8+8]) for i in range(3)]),
                        pos = self.lane_ax[idx]
                        line, = axs[pos[0], pos[1]].plot(time, x, c=cache_colors.get(traj["_id"]))
                           
                    except Exception as e:
                        print(e)
                        # print("lane idx {} is out of bound for EB".format(idx))
                        pass
            return axs
        
        
        frame_text = None
        self.anim = animation.FuncAnimation(fig, func=update_cache,
                                            init_func= init,
                                            frames=frames,
                                            repeat=False,
                                            interval=increment * 1000, # in ms
                                            fargs=( frame_text), # specify time increment in sec to update query
                                            blit=False)
        self.paused = False
        fig.canvas.mpl_connect('key_press_event', self.toggle_pause)

        
        if save:
            self.anim.save('{}.mp4'.format(self.collection_name), writer='ffmpeg', fps=int(1/increment))
          
        if self.overhead:
            fig.tight_layout()
        plt.show()
        print("complete")
        


    
    def toggle_pause(self, event):
        """
        press spacebar to pause/resume animation
        """
        printed = set()
        if event.key == " ":
            if self.paused:
                self.anim.resume()
                print("Animation Resumed")
                self.cursor.remove()
            else:
                self.anim.pause()
                print("Animation Paused")
                printed = set()
                self.cursor = mplcursors.cursor(hover=True)
                def on_add(sel):
                    if self.paused and (sel.artist.get_label()[0] != "_"):
                        label = sel.artist.get_label()
                        if label not in printed:
                            print(label)
                            printed.add(label)
                        sel.annotation.set_text(sel.artist.get_label())
                # connect mouse event to hover for car ID
                self.cursor.connect("add", lambda sel: on_add(sel))
            self.paused = not self.paused

        
        
    
if True and __name__=="__main__":
    
    
    # config_path = os.path.join(os.getcwd(),"../config")
    # os.environ["user_config_directory"] = config_path
    # os.environ["my_config_section"] = "TEST"
    parameters = parse_cfg("test_config_section", cfg_name = "test_param.config")
    parameters.transformed = "transformed"
    
    # batch_5_07072022, batch_reconciled, 
    p = Plotter(parameters, "batch_reconciled", window_size = 5)
    p.animate(increment=0.05,  frames = 2000, save=False)
    
    