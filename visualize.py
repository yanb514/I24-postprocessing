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
from copy import copy
import time
import requests

 
class LRUCache:
    """
    A least-recently-used cache with integer capacity
    To roll out of the cache for vehicle color and dimensions
    get(): return the key-value in cache if exists, otherwise return -1
    put(): (no update) 
    """
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
 
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        else:
            self.cache.move_to_end(key)
            return self.cache[key]
 
    def put(self, key: int, value: int, update = False) -> None:
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
    
    def __init__(self, config, collections = None,
                 framerate = 25, x_min = 0, x_max = 1500, offset = None ,duration = 60):
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
        list_dbr = [] # time indexed
        list_veh = [] # vehicle indexed
        list_db = ["trajectories", "trajectories", "reconciled"] # gt, raw, rec
        trans = DBClient(**config, database_name = "transformed")
        transformed_collections = trans.list_collection_names()
        
        # first collection is GT
        for i,collection in enumerate(collections):
            dbr = DBClient(**config, database_name = "transformed", collection_name=collection)
            veh = DBClient(**config, database_name = list_db[i], collection_name=collection)
            dbr.create_index("timestamp")
            list_dbr.append(dbr)
            list_veh.append(veh)
            
            if collection not in transformed_collections:
                # print("Transform ", collection)
                veh.transform()
            
        
        if len(list_dbr) == 0:
            raise Exception("at least one collection must be specified.")
        
        # get plotting ranges
        t_min = max([dbr.get_min("timestamp") for dbr in list_dbr])
       
        if offset:
            t_min += offset
        if duration: t_max = t_min+duration 
        else: t_max = min([dbr.get_max("timestamp") for dbr in list_dbr])
        
        self.x_start = x_min
        self.x_end = x_max
        self.t_min = t_min
        self.t_max = t_max
        
        # Initialize animation
        self.anim = None
        # TODO: framerate for timestmap
        self.framerate = framerate if framerate else 25
        
        self.lanes = [i*12 for i in range(-1,12)]   
        self.lane_name = [ "EBRS", "EB4", "EB3", "EB2", "EB1", "EBLS", "WBLS", "WB1", "WB2", "WB3", "WB4", "WBRS"]
        self.lane_idx = [i for i in range(12)]
        self.lane_ax = [[1,5],[1,4],[1,3],[1,2],[1,1],[1,0],[0,0],[0,1],[0,2],[0,3],[0,4],[0,5]]
        
        self.annot_queue = queue.Queue()
        self.cursor = None
        
        self.list_dbr =  list_dbr
        self.list_veh = list_veh
        
        # put flagged vehicles in cache_colors
        conflicts = set()
        for doc in self.list_veh[-1].collection.find({}):
            if 'conflicts' in doc:
                conflicts.add(doc["_id"])
        
        # print(conflicts)
        self.conflicts = conflicts
        
        

    
        
    @catch_critical(errors = (Exception))
    def animate(self, save = False, extra=""):
        """
        Advance time window by delta second, update left and right pointer, and cache
        """     
        # set figures: two rows. Top: dbr1 (ax_o), bottom: dbr2 (ax_o2). 4 lanes in each direction
        num = len(self.list_dbr)-1
        fig, axs = plt.subplots(num,1,figsize=(16,3*num))
             
        # TODO: make size parameters
        cache_vehicle = LRUCache(200*num)
        cache_colors = LRUCache(200*num)

        def on_xlims_change(event_ax):
            # print("updated xlims: ", event_ax.get_xlim())
            new_xlim = event_ax.get_xlim()
            # ax1.set(xlim=new_xlim)
            self.x_start = new_xlim[0]
            self.x_end = new_xlim[1]
            
        # OVERHEAD VIEW SETUP
        for i,ax in enumerate(axs):
            ax.set_title(self.list_veh[i+1].collection._Collection__name)
            ax.set_aspect('equal', 'box')
            ax.set(ylim=[self.lanes[0], self.lanes[-1]])
            ax.set(xlim=[self.x_start, self.x_end])
            ax.set_ylabel("EB    WB")
            ax.set_xlabel("Distance in feet")
            ax.callbacks.connect('xlim_changed', on_xlims_change)
      
        
        self.time_cursor = self.list_dbr[0].get_range("timestamp", self.t_min, self.t_max)
        plt.gcf().autofmt_xdate()
        
        
        @catch_critical(errors = (Exception))
        def init():
            for ax in axs:
                # plot lanes on overhead view
                for i in range(-1, 12):
                    if i in (-1, 5, 11):
                        ax.axhline(y=i*12, linewidth=0.5, color='k')
                    else:
                        ax.axhline(y=i*12, linewidth=0.1, color='k')
            
            return axs,
              

        @catch_critical(errors = (Exception))
        def update_cache(frame_text):
            """
            Returns
            -------
            delta : increment in time (sec)
                DESCRIPTION.
            """
            # Stop criteria
            try:
                doc0 = self.time_cursor.next()
            except StopIteration:
                return
            curr_time = doc0["timestamp"]
            # if curr_time >= self.t_max:
            #     print("Reach the end of time. Exit.")
            #     return
            docs = []
            for dbr in self.list_dbr[1:]:
                doc = dbr.find_one("timestamp", curr_time)
                if not doc:
                    doc = {"id": [], "position":[], "dimensions":[]}
                docs.append(doc)
            
            time_text = datetime.utcfromtimestamp(int(curr_time)).strftime('%m/%d/%Y, %H:%M:%S')
            plt.suptitle(time_text, fontsize = 20)
            
            # remove all car_boxes and verticle lines
            for ax in axs:
                for box in list(ax.patches):
                    box.set_visible(False)
                    box.remove()
            while not self.annot_queue.empty():
                self.annot_queue.get(block=False).remove()
                
            # Add vehicle ids in cache_colors 
            for doc in docs:   
                for veh_id in doc['id']:
                    if veh_id in self.conflicts:
                        cache_colors.put(veh_id, [1,0,0], update=False)
                    else:
                        cache_colors.put(veh_id, np.random.rand(3,)*0.8, update=False)
                
            # GT
            traj_cursor = self.list_veh[0].collection.find({"_id": {"$in": doc0["id"]} }, 
                                       {"width":1, "length":1, "coarse_vehicle_class": 1})
            # add vehicle dimension to cache
            for traj in traj_cursor:
                # print("** in curosr")
                cache_vehicle.put(traj["_id"], [traj["length"], traj["width"], traj["coarse_vehicle_class"]], update=True)
                    
            
            # query for vehicle dimensions if not in doc (GT or reconciled)
            for i,doc in enumerate(docs):
                if "dimensions" not in doc:
                    traj_cursor = self.list_veh[i+1].collection.find({"_id": {"$in": doc["id"]} }, 
                                               {"width":1, "length":1, "coarse_vehicle_class": 1})
                    # add vehicle dimension to cache
                    for traj in traj_cursor:
                        # print("** in curosr")
                        # print("*put ", i, traj["_id"])
                        cache_vehicle.put(traj["_id"], [traj["length"], traj["width"], traj["coarse_vehicle_class"]], update=True)
                else:
                    for index, veh_id in enumerate(doc['id']):    
                        cache_vehicle.put(veh_id, doc['dimensions'][index], update=True)  
                        # print("**put ", veh_id, doc['dimensions'][index])
             
            # plot GT
            for index in range(len(doc0["position"])):
                car_x_pos = doc0["position"][index][0]
                car_y_pos = doc0["position"][index][1]

                # print("** ",cache_vehicle.get(doc0["id"][index]))
                # print(len(cache_vehicle.cache))
                # print(len(cache_colors.cache))
                car_length, car_width, _ = cache_vehicle.get(doc0["id"][index])
                car_y_pos -= 0.5 * car_width
                if car_y_pos >= 60: # west bound
                    car_x_pos -= car_length
                    
                if doc0["id"][index] in self.conflicts:
                    fill = False
                else:
                    fill = True
                box = patches.Rectangle(xy = (car_x_pos, car_y_pos),
                                        width = car_length, height=car_width,
                                        color=[0.8,0.8,0.8],  fill = fill) # light grey
                for i in range(num):
                    axs[i].add_patch(copy(box)) 
                    
                    
            # plot vehicles
            for i, doc in enumerate(docs):
                for index in range(len(doc["position"])):
                    car_x_pos = doc["position"][index][0]
                    car_y_pos = doc["position"][index][1]

                    # print("** ",cache_vehicle.get(doc["id"][index]))
                    # print(i,doc["id"][index])
                    car_length, car_width, _ = cache_vehicle.get(doc["id"][index])
                    car_y_pos -= 0.5 * car_width
                    if car_y_pos >= 60: # west bound
                        car_x_pos -= car_length

                    if doc["id"][index] in self.conflicts:
                        fill = False
                    else:
                        fill = True
                    box = patches.Rectangle(xy=(car_x_pos, car_y_pos),
                                            width=car_length, height=car_width, 
                                            color=cache_colors.get(doc["id"][index]),
                                            label=doc["id"][index], fill = fill)
                    axs[i].add_patch(box)   
                    # add annotation
                    annot = axs[i].annotate(doc['_id'], xy=(car_x_pos,car_y_pos))
                    annot.set_visible(False)
                    self.annot_queue.put(annot)

            return axs
        
        
        frame_text = None
        self.anim = animation.FuncAnimation(fig, func=update_cache,
                                            init_func= init,
                                            frames=int(self.t_max-self.t_min)*self.framerate,
                                            repeat=False,
                                            interval=1/self.framerate * 1000, # in ms
                                            fargs=( frame_text), # specify time increment in sec to update query
                                            blit=False,
                                            cache_frame_data = False,
                                            save_count = 1)
        self.paused = False
        fig.canvas.mpl_connect('key_press_event', self.toggle_pause)

        
        if save:
            now = datetime.utcfromtimestamp(int(time.time())).strftime('%Y-%m-%d_%H-%M-%S')
            file_name = now+"_" + self.list_veh[2].collection._Collection__name +extra+".mp4"
            print(file_name)
            self.anim.save(file_name, writer='ffmpeg', fps=self.framerate)
            # self.anim.save('{}.gif'.format(file_name), writer='imagemagick', fps=self.framerate)
            print("saved.")
            url = 'http://10.2.219.208:5991/upload?type=video'
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

    

def main(rec, gt = "groundtruth_scene_1_130", framerate = 25, x_min=0, x_max=2000, offset=0, duration=90, save=False, extra=""):
    with open("config/parameters.json") as f:
        config = json.load(f)
        parameters = config["db_param"]
    
    raw = rec.split("__")[0]
    print("Generating a video for {}...".format(rec))
    p = OverheadCompare(parameters, 
                collections = [gt, raw, rec],
                framerate = framerate, x_min = x_min, x_max=x_max, offset = offset, duration=duration)
    p.animate(save=save, extra=extra)
    
    
if __name__=="__main__":

    main(rec = "sanctimonious_beluga--RAW_GT1__administers", save=False, offset=70)


    