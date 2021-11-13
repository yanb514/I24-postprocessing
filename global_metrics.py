# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 13:43:23 2021

@author: wangy79
Produce metrics in the absence of ground truth
- Global metrics
	ID counts (Y)
	Space gap distribution
	Valid/invalid (Y)
- Tracklet quality
	Collision
	Lane-change tracks
	Outliers 
		Wlh mean/stdev
	Lengths of tracks
	Missing detection
    # frames that one track has multiple meas
"""

import utils
import utils_evaluation as ev
import utils_vis as vis
import numpy as np
import matplotlib.pyplot as plt

class GlobalMetrics():
    
    def __init__(self, filepath, params, rawpath=None):
        '''
        
        '''
        self.df = utils.read_data(filepath)
        print("Select from Frame {} to {}".format(params["start"], params["end"] ))
        self.df = self.df[(self.df["Frame #"] >= params["start"]) & (self.df["Frame #"] <= params["end"])]
        self.params = params
        self.metrics = {}
        self.data = {} # storing evaluation metrics long data
       
        if rawpath:
            self.raw = utils.read_data(rawpath)
            self.raw = self.raw[(self.raw["Frame #"] >= params["start"]) & (self.raw["Frame #"] <= params["end"])]
            
    def evaluate_by_frame(self):
        # extract spacing distribution
        return
      
    def evaluate(self):
        print("*** Evaluation mode: ", self.params["mode"])
        if self.params["mode"] == "DA":
            groups = self.df.groupby("ID")
            groupList = list(groups.groups)
            self.metrics["Total tracklets"] = self.df.groupby("ID").ngroups
            
            # invalid/valid tracks and collision in time space
            print("Check for valid tracks and collision...")
            valid, collision = ev.get_invalid(self.df, ratio=0.4)
            invalid = set(groupList)-valid-collision
            self.metrics["Valid tracklets"] = valid
            self.metrics["Collision with valid tracks"] = collision
            
            # lane-change tracks
            print("Check for lane change...")
            lane_change = ev.get_lane_change(self.df) - invalid
            self.metrics["Possible lane change"] = lane_change
            
            # tracks that have multiple meas per frame
            print("Check for multiple detections...")
            multiple_frames = ev.get_multiple_frame_track(self.df) # dict
            if hasattr(self,"raw"):
                multiple_frames_raw = ev.get_multiple_frame_track(self.raw) 
                self.metrics["Tracks with multiple detections"] = {key:value for key,value in multiple_frames.items() if (key not in multiple_frames_raw or value != multiple_frames_raw[key])}
            else:
                self.metrics["Tracks with multiple detections"] = multiple_frames
            
            # outliers
            print("Checking for outliers...")
            self.raw = self.raw.groupby("ID").apply(ev.mark_outliers_car).reset_index(drop=True)
            outlier_ratio_raw = {carid: np.count_nonzero(car["Generation method"].values=="outlier")/car.bbr_x.count() for carid, car in self.raw.groupby("ID")}
            self.df = self.df.groupby("ID").apply(ev.mark_outliers_car).reset_index(drop=True)
            outlier_ratio = {carid: np.count_nonzero(car["Generation method"].values=="outlier")/car.bbr_x.count() for carid, car in self.df.groupby("ID")}
            outlier_high = {key: value for key, value in outlier_ratio.items() if (value > self.params["outlier_thresh"]) and (key in valid)}
            name = "Tracks > " + str(self.params["outlier_thresh"]*100) + "% outliers"
            self.metrics[name] = outlier_high
            self.data["outlier_ratio"] = [outlier_ratio_raw, outlier_ratio]
            
            # xrange covered
            xranges0 = ev.get_x_covered(self.raw, ratio=True)
            xranges1 = ev.get_x_covered(self.df, ratio=True)
            self.data['xrange'] = [xranges0, xranges1]
            
            # change of xrange covered
            # xrange_delta = {key: value-xranges0[key] for key,value in xranges1.items() if key in xranges0}
            
            # # w,h,y distribution carid: (mean, std)
            # w_dist0 = ev.get_distribution(self.raw, "width")
            # w_dist1 = ev.get_distribution(self.df, "width")
            
            # l_dist0 = ev.get_distribution(self.raw, "length")
            # l_dist1 = ev.get_distribution(self.df, "length")
            
            # y_dist0 = ev.get_distribution(self.raw, "y")
            # y_dist1 = ev.get_distribution(self.df, "y")
            
        return

    def visualize_metrics(self):
        for name in self.data:
            data = self.data[name]
            if "xrange" in name:
                data_list = [np.fromiter(data_item.values(), dtype=float) for data_item in data]
                vis.plot_histogram(data_list, bins=40,
                                   labels=["raw", self.params["mode"]], 
                                   xlabel="FOV covered (%)", 
                                   ylabel="Probability", 
                                   title="X range (%) distribution")
            elif "outlier" in name:
                data_list = [np.fromiter(data_item.values(), dtype=float) for data_item in data]
                vis.plot_histogram(data_list, bins=40,
                                   labels=["raw", self.params["mode"]], 
                                   xlabel="Outlier ratio", 
                                   ylabel="Probability", 
                                   title="Outlier ratio distribution")
      
    def print_metrics(self):
        print("\n")
        for name in self.metrics:
            if "Valid tracklets" in name: 
                print("{:<30}: {}".format(name,len(self.metrics[name])))

            else:
                if (not isinstance(self.metrics[name], int)) and (len(self.metrics[name])==0):
                    continue
                print("{:<30}: {}".format(name,self.metrics[name]))
        return
 
    def evaluate_single_track(self, carid, plot=True, dashboard=True):
        '''
        identify a problematic track
        '''
        car = self.df[self.df["ID"]==carid]
        if plot:
            vis.plot_track_df(car)
        if dashboard:
            vis.dashboard([car])
        return
    
    
if __name__ == "__main__":
    

    data_path = r"E:\I24-postprocess\MC_tracking" 
    file_path = data_path+r"\DA\MC_tsmn.csv"
    mode = "DA"
    
    raw_file_path = data_path+r"\MC_reinterpolated.csv"
    # mode = "raw"
    
    params = {
              "start": 1000,
              "end": 2000,
              "outlier_thresh": 0.25,
              "mode": mode
              }

    
    gm = GlobalMetrics(file_path, params, rawpath=raw_file_path)
    gm.evaluate()
    gm.print_metrics()
    gm.visualize_metrics()
    
    # gm.evaluate_single_track(133, plot=True, dashboard=True)
    
   
    