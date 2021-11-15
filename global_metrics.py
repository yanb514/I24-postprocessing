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
import warnings
warnings.filterwarnings("ignore")

class GlobalMetrics():
    
    def __init__(self, params, raw_path, da_path=None, rec_path=None):
        '''
        '''
        self.raw = utils.read_data(raw_path)
        print("Select from Frame {} to {}".format(params["start"], params["end"] ))
        self.raw = self.raw[(self.raw["Frame #"] >= params["start"]) & (self.raw["Frame #"] <= params["end"])]
        
        if da_path:
            self.da = utils.read_data(da_path)
            self.da = self.da[(self.da["Frame #"] >= params["start"]) & (self.da["Frame #"] <= params["end"])]
        else: self.da = None
        if rec_path:
            self.rec = utils.read_data(rec_path)
            self.rec = self.rec[(self.rec["Frame #"] >= params["start"]) & (self.rec["Frame #"] <= params["end"])]
        else: self.rec = None   
        self.params = params
        self.metrics = {}
        self.data = {} # storing evaluation metrics long data
                
            
    def evaluate_by_frame(self):
        # extract spacing distribution
        return
      
    def evaluate(self):
        # evaluation raw data
        self.metrics["Total tracklets"] = []
        self.metrics["Valid tracklets"] = []
        self.metrics["Collision with valid tracks"] = []
        self.metrics["Possible lane change"] = []
        self.metrics["Tracks with multiple detections"] = []
        name = "Tracks > " + str(self.params["outlier_thresh"]*100) + "% outliers"
        self.metrics[name] = []
        
        self.data["outlier_ratio"] = []
        self.data['xrange'] = []
        self.data['Width variance'] = []
        self.data['Length variance'] = []
        self.data['y variance'] = []
        self.data["correction_score"] = ev.get_correction_score(self.da, self.rec)
        
        for df in [self.raw, self.da, self.rec]:
            if df is None:
                continue
            groupList = list(df.groupby("ID").groups)
            valid, collision = ev.get_invalid(df, ratio=self.params["outlier_thresh"])      
            invalid = set(groupList)-valid-collision
            lane_change = ev.get_lane_change(df) - invalid
            multiple_frames = ev.get_multiple_frame_track(df) # dict
            outlier_ratio = {carid: np.count_nonzero(car["Generation method"].values=="outlier")/car.bbr_x.count() for carid, car in df.groupby("ID")}
            outlier_high = {key: value for key, value in outlier_ratio.items() if (value > self.params["outlier_thresh"]) and (key in valid)}  
            xranges = ev.get_x_covered(df, ratio=True)  
            w_var = ev.get_variation(df, "width")
            l_var = ev.get_variation(df, "length")
            y_var = ev.get_variation(df, "y")
            df = df.groupby("ID").apply(ev.mark_outliers_car).reset_index(drop=True)
            
            # metrics are to be printed (in print_metrics)
            self.metrics["Total tracklets"].append(df.groupby("ID").ngroups)
            self.metrics["Valid tracklets"].append(valid)
            self.metrics["Collision with valid tracks"].append(collision)
            self.metrics["Possible lane change"].append(lane_change)
            self.metrics["Tracks with multiple detections"].append(multiple_frames)
            self.metrics[name].append(outlier_high)
            
            # data is to be plotted (in visualize)
            self.data["outlier_ratio"].append(outlier_ratio)
            self.data['xrange'].append(xranges)
            self.data['Width variance'].append(w_var)
            self.data['Length variance'].append(l_var)
            self.data['y variance'].append(y_var)

        self.metrics["Score > " + str(self.params["score_thresh"])] = {carid:score for carid,score in self.data["correction_score"].items() if score>self.params["score_thresh"]}
        return

    def visualize_metrics(self):
        for name in self.data:
            data = self.data[name]
            if isinstance(data, list):
                data_list = [np.fromiter(data_item.values(), dtype=float) for data_item in data]
            else:
                data_list = np.fromiter(data.values(), dtype=float)
            if  name == "xrange":
               xlabel = "FOV covered (%)"
               ylabel = "Probability"
               title = "X range (%) distribution"
               
            elif "delta" in name:
                xlabel = name
                ylabel = "Probability"
                title = "{} distribution".format(name)
                
            elif name == "outlier_ratio":
                xlabel = "Outlier ratio"
                ylabel = "Probability"
                title = "Outlier ratio distribution"
            
            elif "variance" in name:
                xlabel = name
                ylabel = "Probability"
                title = "{} variance distribution".format(name[0])
            
            elif "correction_score" in name:
                xlabel = "Correction score"
                ylabel = "Probability"
                title = "Correction score distribution"

            vis.plot_histogram(data_list, bins=40,
                                   labels="" if len(data_list)==1 else ["raw", "da", "rec"], 
                                   xlabel= xlabel, 
                                   ylabel= ylabel, 
                                   title= title)
         
        # plot correction score vs. DA's outlier ratio
        plt.figure()
        for carid, score in self.data["correction_score"].items():
            try:
                plt.scatter(score,self.data["outlier_ratio"][1][carid], s=2, c='b')
            except:
                pass
        plt.xlabel("correction score")
        plt.ylabel("outlier ratio")  
        plt.title("Correction score vs. outlier ratio")
        return
         
    def print_metrics(self):
        print("\n")
        for name in self.metrics:
            if "Valid tracklets" in name: 
                print("{:<30}: {}".format(name,[len(item) for item in self.metrics[name]]))

            else:
                if (not isinstance(self.metrics[name], int)) and (len(self.metrics[name])==0):
                    continue
                print("{:<30}: {}".format(name,self.metrics[name]))
        return
 
    def evaluate_single_track(self, carid, plot=True, dashboard=True):
        '''
        identify a problematic track
        '''
        # raw = self.raw[self.raw["ID"]==carid]
        da = self.da[self.da["ID"]==carid]
        rec = self.rec[self.rec["ID"]==carid]
        if plot:
            vis.plot_track_compare(da,rec)
        if dashboard:
            vis.dashboard([da, rec],["da","rectified"])
        return
    
    
if __name__ == "__main__":
    
    data_path = r"E:\I24-postprocess\MC_tracking" 
    raw_path = data_path+r"\MC_reinterpolated.csv"
    da_path = data_path+r"\DA\MC_tsmn.csv"
    rec_path = data_path+r"\rectified\MC_tsmn.csv"
    
    params = {
              "start": 0,
              "end": 1000,
              "outlier_thresh": 0.25,
              "score_thresh": 3
              }

    
    gm = GlobalMetrics(params, raw_path, da_path, rec_path)
    gm.evaluate()
    gm.print_metrics()
    gm.visualize_metrics()
    
    # gm.evaluate_single_track(216, plot=True, dashboard=True)
    
   
    