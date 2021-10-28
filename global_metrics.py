# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 13:43:23 2021

@author: wangy79
Histograms (GT, raw, rec)
	- X,y
	- Length
	- Width
	- Speed

Scatter plots
 Positional error (GT vs. raw & rec) wrt.

	- Speed
	- Lanes
	- Vehicle dimension
	- L1 l2 norm
	
Dimension error wrt.
	- Position (x)
    - Vehicle class
"""
import torch
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from homography import Homography, load_i24_csv
import utils
from data_association import count_overlaps
import numpy.linalg as LA
import matplotlib.pyplot as plt
import utils_vis as vis
import scipy.stats as sps
import pickle
import seaborn as sns
from rec_evaluator import MOT_Evaluator

class GlobalMetrics():
    
    def __init__(self, ev1, ev2, ev3):
        '''

        Parameters
        ----------
        evi : TYPE
            rec_evaluator object.
            [raw, DA, rectified]

        Returns
        -------
        None.
        '''
        self.gt = ev1.gt
        self.raw = ev1.rec
        self.da = ev2.rec
        self.rec = ev3.rec
        
        self.ev_raw = ev1
        self.ev_da = ev2
        self.ev_rec = ev3
        
        self.gt = self.gt.groupby("ID").apply(utils.constant_speed).reset_index(drop=True)
        self.gt["length"] = np.abs(self.gt["bbr_x"] - self.gt["fbr_x"])
        self.gt["width"] = np.abs(self.gt["bbr_y"] - self.gt["bbl_y"])
    
    def histogram(self, data_list, xlabel, ylabel, title, mn, mx):
        '''
        produce histograms on a single plot using KDE smoothing
        '''
        if len(data_list)==4:
            gt, raw, da, rec = data_list
        else:
            raw, da, rec = data_list
        plt.figure()
        try:
            gt = gt[~np.isnan(gt)]
        except:
            pass
            
        raw = raw[~np.isnan(raw)]
        da = da[~np.isnan(da)]
        rec = rec[~np.isnan(rec)]
        
        if mn == None:
            all_value = np.concatenate([raw,da,rec])
            mn, mx = min(all_value), max(all_value)
            r =mx-mn
            mn -= 0.3*r
            mx+= 0.3*r
           
        try:
            x = np.linspace(mn, mx, len(gt))
            kde = sps.gaussian_kde(gt)
            plt.plot(x, kde.pdf(x), color = 'black', label='GT')
        except:
            pass
            
        x = np.linspace(mn, mx, len(raw))
        kde = sps.gaussian_kde(raw)
        plt.plot(x, kde.pdf(x), color = 'r', label='raw')
        
        x = np.linspace(mn, mx, len(da))
        kde = sps.gaussian_kde(da)
        plt.plot(x, kde.pdf(x), color = 'y', label='DA')
        
        x = np.linspace(mn, mx, len(rec))
        kde = sps.gaussian_kde(rec)
        plt.plot(x, kde.pdf(x), color = 'b', label='rectified')
        
        plt.legend()
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        sns.despine()
        
        
    def vis_state_distribution(self):
        
        # histogram of spacing
        gt = np.array(self.ev_raw.m["space_gap_gt"])
        raw = np.array(self.ev_raw.m["space_gap_rec"])
        da = np.array(self.ev_da.m["space_gap_rec"])
        rec = np.array(self.ev_rec.m["space_gap_rec"])
        self.histogram([gt, raw, da, rec], "Spacing (m)", "PDF", "Spacing histogram", 0, 230)

        # histogram of speed
        gt = np.array(self.gt.speed.values)
        raw = np.array(self.raw.speed.values)
        da = np.array(self.da.speed.values)
        rec = np.array(self.rec.speed.values)
        self.histogram([gt, raw, da, rec], "Speed (m/s)", "PDF", "Speed histogram", 0, 50)
        
        # histogram of X
        gt = np.array(self.gt.x.values)
        raw = np.array(self.raw.x.values)
        da = np.array(self.da.x.values)
        rec = np.array(self.rec.x.values)
        self.histogram([gt, raw, da, rec], "Position (m)", "PDF", "Position-x histogram", None, None)

        # histogram of length
        gt = np.array(self.gt.length.values)
        raw = np.array(self.raw.length.values)
        da = np.array(self.da.length.values)
        rec = np.array(self.rec.length.values)
        self.histogram([gt, raw, da, rec], "Length (m)", "PDF", "Length histogram", None, None)
        
        # histogram of width
        gt = np.array(self.gt.width.values)
        raw = np.array(self.raw.width.values)
        da = np.array(self.da.width.values)
        rec = np.array(self.rec.width.values)
        self.histogram([gt, raw, da, rec], "Width (m)", "PDF", "Width histogram", None, None)

    def vis_state_error(self):
        # states: L,W,H,x,y,velocity
        raw = np.array(torch.stack(self.ev_raw.m["state_err"]))
        da = np.array(torch.stack(self.ev_da.m["state_err"]))
        rec = np.array(torch.stack(self.ev_rec.m["state_err"]))
        
        self.histogram([raw[:,0], da[:,0], rec[:,0]], "Length MAE (m)", "PDF", "Length MAE distribution", None, None)
        self.histogram([raw[:,3], da[:,3], rec[:,3]], "Position MAE (m)", "PDF", "Position MAE distribution", None, None)
        self.histogram([raw[:,6], da[:,6], rec[:,6]], "Speed MAE (m/s)", "PDF", "Speed MAE distribution", -20, 50)
        
        
        
if __name__ == "__main__":
    
    camera_name = "p1c3"
    sequence_idx = 0

    file = "{}_{}".format(camera_name,sequence_idx)
    
    # load mutiple pickle data
    modes = ["raw","da","rec"]
    ev_list = []
    for mode in modes:
        file_mode = file+"_{}.pkl".format(mode)
        with open(file_mode, "rb") as f:
            ev_list.append(pickle.load(f))
        
    ev1, ev2, ev3 = ev_list # unpack
    gm = GlobalMetrics(ev1,ev2,ev3)
    gm.vis_state_distribution()
    gm.vis_state_error()
    
    