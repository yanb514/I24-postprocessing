# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 10:45:24 2021

@author: wangy79
Data association class consists of 
- METHODS
    - time_space_matching
    - time_space_matching_online
    - time_space_matching_online_2
    
- DA evaluation with GT
    - fragments and ID switches
"""
import numpy as np
import utils
# import pandas as pd
import utils_vis as vis
import matplotlib.pyplot as plt
import utils_evaluation as ev
import utils_data_association as uda
from collections import defaultdict
import time


class Data_Association():
    
    def __init__(self, data_path, gt_path, params = None):
        '''

        '''
        
        self.params = params 

        self.df = utils.read_data(data_path)
        try:
            self.df = self.df[(self.df["Frame #"] >= params["start"]) & (self.df["Frame #"] <= params["end"])]
        except: pass
        self.original = self.df.copy()
        print("Total Frames: ", max(self.df["Frame #"].values)-min(self.df["Frame #"].values))
        self.gt = utils.read_data(gt_path)
        try:
            self.gt = self.gt[(self.gt["Frame #"] >= params["start"]) & (self.gt["Frame #"] <= params["end"])]
        except: pass
   
    def stitch(self):

        THRESHOLD_MIN, THRESHOLD_MAX, VARX, VARY, time_out = self.params["args"]
        # self = uda.stitch_objects_tsmn_ll(self, THRESHOLD_MAX, VARX, VARY, time_out)
        # self = uda.stitch_objects_tsmn_online(self, THRESHOLD_MIN, THRESHOLD_MAX, VARX, VARY, time_out)
        # start = time.time()
        self = uda.stitch_objects_tsmn_online_3(self, THRESHOLD_MAX, VARX, VARY, time_out)
        # end = time.time()
        # print('total runtime: ', end-start)
        return

    def da_evaluate(self, synth=True):
        '''

        '''
        # build an inverse path: {newid:{oldids}}
        inv_path = defaultdict(set)
        for oldid, newid in self.path.items():
            inv_path[newid].add(oldid)
        
        # FRAG = defaultdict(set)
        FRAG = set()
        IDS = defaultdict(set)
        matched = defaultdict(set)
        nfrags = 0
        for newid, oldids in inv_path.items():
            trueID = newid if newid<1000 else newid//1000
            if (newid>=1000) and (newid in self.past_tracks):
                # FRAG[trueID].add(newid)
                FRAG.add(newid)
                nfrags += 1
            for oldid in oldids:
                if oldid != newid:
                    matched[newid].add(oldid)
                    if oldid//1000 != trueID:
                        IDS[trueID].add(oldid)
        self.FRAG = FRAG
        self.IDS = IDS
        self.matched = matched
        print("{} Fragments: {}".format(nfrags, FRAG))
        print("ID switches: {}".format(IDS))
        return
        
        
        
    def evaluate(self, synth=True):
        '''
        synth = True: for synthetic experiments only
            compute fragments and ID switches
        get_invalid
        mark_outliers
        '''
        if synth:
            idpath = {}
            FRAG = {}
            IDS = {}
            visited = set()
            for idx, idx_v in self.path.items():
                trueID = self.groupList[int(idx)]
                if trueID in self.empty_id or trueID in visited:
                    continue
                assID = [self.groupList[id] for id in idx_v]
                idpath[trueID] = assID
                if trueID >= 1000 and all([l>=1000 for l in assID]) and (trueID//1000 in idpath.keys()):
                    FRAG[trueID] = assID
                visited.add(trueID)
                visited = visited.union(set(assID))
                if trueID>=1000: trueID = trueID//1000
                if any([l//1000 != trueID and l!= trueID for l in assID]):
                    IDS[trueID] = assID
            self.idpath = idpath
            self.FRAG = FRAG
            self.IDS = IDS
            print("Fragments: {}".format(FRAG))
            print("ID switches: {}".format(IDS))
            return     
        
        print("{} tracklets are stitched".format(self.original.groupby("ID").ngroups-self.df.groupby("ID").ngroups))
        valid0,collision0 = ev.get_invalid(self.original, ratio=0.4)
        valid,collision = ev.get_invalid(self.df, ratio=0.4) # remove obvious invalid tracks from df
        print("{} more valid tracklets are created".format(len(valid)-len(valid0)))
        self.data["valid"] = valid
        self.data["collision"] = collision
        print("Valid tracklets: {}/{}".format(len(valid), self.df.groupby("ID").ngroups))
        print("Collision with valid: {}".format(collision))
        
        # check lane-change vehicles
        groups = self.df.groupby("ID")
        multiple_lane = set()
        for carid, group in groups:
            if group.lane.nunique()>1:
                # frames = group.groupby("Frame #")
                # for frame_id, frame in frames:
                if np.abs(np.max(group[["bbr_y","bbl_y"]].values)-np.min(group[["bbr_y","bbl_y"]].values)) > 12/3.281:
                    if carid in valid:
                        multiple_lane.add(carid)
                        # break
                    
        self.data["lane_change"] = multiple_lane
        print("Possible lane-change tracks:", multiple_lane)
        
        self.df = self.df.groupby("ID").apply(ev.mark_outliers_car).reset_index(drop=True)
        outlier_ratio = {carid: np.count_nonzero(car["Generation method"].values=="outlier")/car.bbr_x.count() for carid, car in self.df.groupby("ID")}
        self.data["outlier_ratio"] = outlier_ratio
        outlier_high = {key: value for key, value in outlier_ratio.items() if (value > self.params["outlier_thresh"]) and (key in valid)}  
        print("Outlier ratio above {}: {}".format(self.params["outlier_thresh"],outlier_high))
        vis.plot_histogram(np.fromiter(outlier_ratio.values(), dtype=float), bins=40,
                           labels="", 
                           xlabel = "Outlier ratio", 
                           ylabel = "Probability", 
                           title = "Outlier ratio distribution")
        return
    

    def visualize_BA(self, lanes=[1,2,3,4,7,8,9,10]):
        
        for lane_idx in lanes:
            fig, axs = plt.subplots(1,2, figsize=(15,5), facecolor='w', edgecolor='k')
            axs = axs.ravel()
            vis.plot_time_space(self.original, lanes=[lane_idx], time="frame", space="x", ax=axs[0])
            vis.plot_time_space(self.df, lanes=[lane_idx], time="frame", space="x", ax=axs[1])
            fig.tight_layout()
    
    
if __name__ == "__main__":

    raw_path = r"E:\I24-postprocess\benchmark\TM_5000_pollute_nojerk.csv"
    gt_path = r"E:\I24-postprocess\benchmark\TM_5000_GT_nojerk.csv"
    
    params = {
              "threshold": (0,0), # 0.3, 0.04 for tsmn
               # "start": 0, # starting frame
               # "end": 1000, # ending frame
              "args": (1, 3, 0.05, 0.02, 80) # THRESHOLD_C=3, VARX=0.03, VARY=0.03, time_out = 500
              }
    
    da = Data_Association(raw_path, gt_path, params)

    da.df = da.df[da.df["ID"].isin([66,66008,66009])] # for debugging purposes
    da.stitch()    
    da.da_evaluate(synth=True) # set to False if absence of GT

    # da.visualize_BA(lanes=[1,2,3,4])


    
    