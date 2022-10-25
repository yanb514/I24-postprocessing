#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:22:38 2022

@author: yanbing_wang

"""

from i24_database_api import DBClient
# from i24_configparse import parse_cfg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.animation as animation
from datetime import datetime
from i24_logger.log_writer import catch_critical
import queue
import mplcursors
from collections import OrderedDict
import json
import time
import requests
import os
from bson.objectid import ObjectId

 
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
        if key not in self.cache:
            return -1
        else:
            self.cache.move_to_end(key)
            return self.cache[key]
 
    def put(self, key, value, update = False):
        if key not in self.cache: # do not update with new value
            self.cache[key] = value
        elif update:
            self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last = False)
            
            
            
            
class OverheadCompare():
    """
    compare the overhead views of two collecctions
    """
    
    def __init__(self, config, gt=None, raw=None, rec=None,
                 framerate = 25, x_min = 0, x_max = 1500, offset = None ,duration = 60, transform = False):
        """
        Initializes a Plotter object
        
        Parameters
        ----------
        config : object or dictionary for database access
        vehicle_database: database name for vehicle ID indexed collection
        vehicle_colleciton: collection name for vehicle ID indexed collection
        timestamp_database: database name for time-indexed documents
        timestamp_collection1 and 2: collection name for time-indexed documents
        framerate: (FPS) rate to query timestamps and to advance the animation
        x_min/x_max: (feet) roadway range for overhead view
        duration: (sec) duration for animation
        """

        collections = [gt, raw, rec] # any one could be None
        attr_names = ["gt", "raw", "rec"]
        list_db = ["trajectories", "trajectories", "reconciled"] # gt, raw, rec
        
        trans = DBClient(**config, database_name = "transformed")
        transformed_collections = trans.list_collection_names()
        
        # first collection is GT
        for i,collection in enumerate(collections):
            if list_db[i] is not None:
                dbr = DBClient(**config, database_name = "transformed", collection_name=collection)
                veh = DBClient(**config, database_name = list_db[i], collection_name=collection)
                dbr.create_index("timestamp")
                setattr(self, attr_names[i], veh) # self.gt = veh
                setattr(self, attr_names[i]+"_t", dbr) # self.gt_t = dbr
            else:
                attr_names[i] = None
                list_db[i] = None # mask out all the none items
            
            if collection not in transformed_collections or transform == True:
                # print("Transform ", collection)
                veh.transform()

        
        # get plotting ranges
        t_min = max([getattr(self, name+"_t").get_min("timestamp") for name in attr_names if name])
        t_max = max([getattr(self, name+"_t").get_max("timestamp") for name in attr_names if name])
        
        if offset:
            t_min += offset  
        if duration: 
            t_max = min(t_max, t_min + duration)
        
        self.x_start = x_min
        self.x_end = x_max
        self.t_min = t_min
        self.t_max = t_max
        
        # Initialize animation
        self.anim = None
        self.framerate = framerate if framerate else 25
        
        self.lanes = [i*12 for i in range(-1,12)]   
        self.lane_name = [ "EBRS", "EB4", "EB3", "EB2", "EB1", "EBLS", "WBLS", "WB1", "WB2", "WB3", "WB4", "WBRS"]
        self.lane_idx = [i for i in range(12)]
        self.lane_ax = [[1,5],[1,4],[1,3],[1,2],[1,1],[1,0],[0,0],[0,1],[0,2],[0,3],[0,4],[0,5]]
        
        self.annot_queue = queue.Queue()
        self.cursor = None
        
        self.collections = collections
        self.attr_names = attr_names
        
        
        
    @catch_critical(errors = (Exception))
    def animate(self, save = False, upload = False, extra=""):
        """
        Advance time window by delta second, update left and right pointer, and cache
        """     
        # set figures: two rows. Top: dbr1 (ax_o), bottom: dbr2 (ax_o2). 4 lanes in each direction
        num = 2
        fig, axs = plt.subplots(num,1,figsize=(16,3*num))
        self.labels = None
        
        def on_xlims_change(event_ax):
            # print("updated xlims: ", event_ax.get_xlim())
            new_xlim = event_ax.get_xlim()
            # ax1.set(xlim=new_xlim)
            self.x_start = new_xlim[0]
            self.x_end = new_xlim[1]
            
        # OVERHEAD VIEW SETUP
        for i,ax in enumerate(axs):
            ax.set_title(self.collections[i+1])
            ax.set_aspect('equal', 'box')
            ax.set(ylim=[self.lanes[0], self.lanes[-1]])
            ax.set(xlim=[self.x_start, self.x_end])
            ax.set_ylabel("EB    WB")
            ax.set_xlabel("Distance in feet")
            # ax.callbacks.connect('xlim_changed', on_xlims_change)
      
        try:
            self.time_cursor = self.gt_t.get_range("timestamp", self.t_min, self.t_max)
        except:
            self.time_cursor = self.rec_t.get_range("timestamp", self.t_min, self.t_max)
        # plt.gcf().autofmt_xdate()
        
        
        @catch_critical(errors = (Exception))
        def init():
            
            # initialize caches
            self.by_label = LRUCache(10)
            # self.veh_cache =  [LRUCache(400) for _ in self.list_dbr] #key:_id, val:{dim}
            
            # create a cache for each type (gt, rec, raw) to keep dimensions
            for attr in self.attr_names:
                if attr:
                    setattr(self, attr+"_cache", LRUCache(400)) 

            # keep all the gt dimensions
            # NIXIPIN
            if hasattr(self, "gt"):
                gt_query = self.gt.collection.aggregate([
                    {"$match": {"$and" : [{"first_timestamp": {"$lte": self.t_max}},{"last_timestamp": {"$gte": self.t_min}}]}},
                    {'$project':{ 'width':1, 'length':1}}])
                for doc in gt_query:
                    val = {"dim": [doc["length"], doc["width"]],
                           "kwargs": {
                                "color": [0.8]*3, # light grey
                               # "color": np.random.rand(3,)*0.5,
                               "fill": True
                               # "label": "GT"}
                               }
                           }
    
                    self.gt_cache.put(doc["_id"], val, update=False)
                
            # plot lanes on overhead view
            for ax in axs:  
                for i in range(-1, 12):
                    if i in (-1, 5, 11):
                        ax.axhline(y=i*12, linewidth=0.5, color='k')
                    else:
                        ax.axhline(y=i*12, linewidth=0.1, color='k')
            return axs,
              

        @catch_critical(errors = (Exception))
        def update_cache(curr_time):
            """
            Update the cache for each collection (except for GT)
            """
            if hasattr(self, "raw"):
                doc = self.raw_t.find_one("timestamp", curr_time) # doc has dim
                if not doc:
                    doc = {"id": [], "position":[], "dimensions":[]}
                
                query = self.raw.collection.find({"_id": {"$in": doc["id"]} }, 
                                           {"feasibility": 1, "fragment_ids": 1, "merged_ids": 1})
                for d in query:
                    if "feasibility" in d and d["feasibility"]["conflict"]<1: # bad flag 
                        kwargs = {
                            "color": [0,1,0], # green
                            "fill": False,
                            "linewidth": 2,
                            # "label": "conflicts",
                            "label": d["_id"]
                            }
                    else:
                        kwargs = {
                            "color": np.random.rand(3,)*0.6,
                            "fill": True,
                            "linewidth": 0,
                            "alpha": 0.7,
                            # "label": "stitched",
                            "label": d["_id"]
                            }

                    val = {"kwargs": kwargs} 
                    self.raw_cache.put(d["_id"], val, update=False)
                    
            if hasattr(self, "rec"):
                doc = self.rec_t.find_one("timestamp", curr_time) # doc does not have dim
                if not doc:
                    doc = {"id": [], "position":[], "dimensions":[]}
                    
                query = self.rec.collection.find({"_id": {"$in": doc["id"]} }, 
                                           {"width":1, "length":1, "feasibility": 1, "fragment_ids": 1, "merged_ids": 1})
                    
                for d in query:
                    if "feasibility" in d and d["feasibility"]["conflict"]<1: # bad flag 
                        kwargs = {
                            "color": [0,1,0], # green
                            "fill": False,
                            "linewidth": 2,
                            # "label": "conflicts",
                            "label": d["_id"]
                            }
                    else:
                        kwargs = {
                            "color": np.random.rand(3,)*0.6,
                            "fill": True,
                            "linewidth": 0,
                            "alpha": 0.7,
                            # "label": "stitched",
                            "label": d["_id"]
                            }
                    
                    
                    val = {"dim": [d["length"], d["width"]],
                           "kwargs": kwargs,
                          } 
                    self.rec_cache.put(d["_id"], val, update=False)
                    

                    
                    
        @catch_critical(errors = (Exception))    
        def update_plot(frame):
            '''
            Advance time cursor and update the artist
            '''
            # Stop criteria - should not reach this line
            try:
                doc0 = self.time_cursor.next()
            except StopIteration:
                print("Reach the end of time. Exit.")
                return
            
            # Update title
            curr_time = doc0["timestamp"]
            time_text = datetime.utcfromtimestamp(int(curr_time)).strftime('%m/%d/%Y, %H:%M:%S')
            plt.suptitle(time_text, fontsize = 20)
            
            update_cache(curr_time)
            
               
            # remove all car_boxes and verticle lines
            for ax in axs:
                for box in list(ax.patches):
                    box.set_visible(False)
                    box.remove()
            while not self.annot_queue.empty():
                self.annot_queue.get(block=False).remove()
             
                
            # plot GT on all axis
            doc = self.gt_t.find_one("timestamp", curr_time)
            for i in range(2):
                for index in range(len(doc["position"])):
                    car_x_pos = doc["position"][index][0]
                    car_y_pos = doc["position"][index][1]
                
                    d = self.gt_cache.get(doc["id"][index])
                    car_length, car_width = d["dim"] # rec
    
                    car_y_pos -= 0.5 * car_width
                    if car_y_pos >= 60: # west bound
                        car_x_pos -= car_length
                
                    box = patches.Rectangle(xy = (car_x_pos, car_y_pos),
                                            width = car_length, height=car_width,
                                            **d["kwargs"])
                    axs[i].add_patch(box)
            
                
            # plot boxes - should not care about GT or rec or raw
            for name in ["raw", "rec"]:
                if name == "raw":
                    i = 0 # plot on axs[i]
                else:
                    i = 1
                dbr = getattr(self, name+"_t")
                doc = dbr.find_one("timestamp", curr_time)
                
                for index in range(len(doc["position"])):
                    car_x_pos = doc["position"][index][0]
                    car_y_pos = doc["position"][index][1]
                
                    d = getattr(self, name+"_cache").get(doc["id"][index])

                    try:
                        car_length, car_width = d["dim"] # rec
                    except:
                        car_length, car_width = doc['dimensions'][index][:2] # raw

                    car_y_pos -= 0.5 * car_width
                    if car_y_pos >= 60: # west bound
                        car_x_pos -= car_length
                
                    box = patches.Rectangle(xy = (car_x_pos, car_y_pos),
                                            width = car_length, height=car_width,
                                            **d["kwargs"])
                    axs[i].add_patch(box)   
                
            
            return axs
        
        frame = None
        self.anim = animation.FuncAnimation(fig, func=update_plot,
                                            init_func= init,
                                            frames=int(self.t_max-self.t_min)*self.framerate,
                                            repeat=False,
                                            interval=1/self.framerate * 1000, # in ms
                                            fargs=(frame ),
                                            blit=False,
                                            cache_frame_data = False,
                                            save_count = 1)
        self.paused = False
        fig.canvas.mpl_connect('key_press_event', self.toggle_pause)

        
        if save:
            now = datetime.utcfromtimestamp(int(time.time())).strftime('%Y-%m-%d_%H-%M-%S')
            file_name = now+"_" + self.rec.collection._Collection__name +extra+".mp4"
            print(file_name)
            self.anim.save(file_name, writer='ffmpeg', fps=self.framerate)
            # self.anim.save('{}.gif'.format(file_name), writer='imagemagick', fps=self.framerate)
            print("saved.")
            
            if upload:
                url = 'http://viz-dev.isis.vanderbilt.edu:5991/upload?type=video'
                files = {'upload_file': open(file_name,'rb')}
                ret = requests.post(url, files=files)
                if ret.status_code == 200:
                    print('Uploaded!')
        
        
        else:
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
                # print("Animation Resumed")
                self.cursor.remove()
            else:
                self.anim.pause()
                # print("Animation Paused")
                printed = set()
                self.cursor = mplcursors.cursor(hover=True)
                def on_add(sel):
                    if self.paused:
                        if sel.artist.get_label() and (sel.artist.get_label()[0] != "_"):
                            label = sel.artist.get_label()
                            if label not in printed:
                                print(label)
                                printed.add(label)
                            sel.annotation.set_text(sel.artist.get_label())
                # connect mouse event to hover for car ID
                self.cursor.connect("add", lambda sel: on_add(sel))
            self.paused = not self.paused

    

def main(gt = "groundtruth_scene_1_130", rec=None, framerate = 25, x_min=-100, x_max=2200, offset=0, duration=90, 
         save=False, upload=False, extra="", transform=False):
    
    with open(os.path.join(os.environ["USER_CONFIG_DIRECTORY"], "db_param.json")) as f:
        db_param = json.load(f)
    
    raw = rec.split("__")[0]
    print("Generating a video for {}...".format(rec))
    p = OverheadCompare(db_param, 
                gt = gt,
                raw = raw,
                rec = rec,
                framerate = framerate, x_min = x_min, x_max=x_max, offset = offset, duration=duration, transform=transform)
    p.animate(save=save, upload=upload, extra=extra)
    
    
if __name__=="__main__":
# "young_ox--RAW_GT2__calibrates"
    main(rec = "sanctimonious_beluga--RAW_GT1__test", save=True, upload = True, offset = 0, transform=False, extra="rolling_conflict_res")
    
        
    # with open(os.environ["USER_CONFIG_DIRECTORY"]+"db_param.json") as f:
    #     db_param = json.load(f)
        
    # rec = "young_ox--RAW_GT2__calibrates"
    # # rec = "zonked_cnidarian--RAW_GT2__articulates"
    # raw = rec.split("__")[0]
    # print("Generating a video for {}...".format(rec))
    
    # framerate = 25
    # x_min=-100
    # x_max=2200
    # offset=0
    # duration=60
    # save=False
    # extra=""
    # gt = "groundtruth_scene_2_57"
    
    # p = OverheadCompare(db_param, 
    #             collections = [gt, raw, rec],
    #             framerate = framerate, x_min = x_min, x_max=x_max, offset = offset, duration=duration)
    # p.animate(save=save, extra=extra)


    