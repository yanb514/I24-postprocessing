# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 10:00:07 2021

@author: wangy79
Synthetic experiment for 1 car

Control variables
- # Frames
- dynamics (state)
- missing rate
- noise level (on measurements)

Performance metrics
- state recovery accuracy
- correction score

"""
import utils
import utils_optimization as opt
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import utils_vis as vis
import numpy.linalg as LA


class Experiment():
    
    def __init__(self, states, params):
        '''
        state: x,y,w,l,v,theta
        '''
        for state in states:
            if "0" not in state:
                setattr(self,state,states[state])
        
        # create ground truth data
        Y, x, y, a = opt.generate(self.width, self.length, states["x0"], states["y0"], self.theta, self.speed, outputall=True)
        states.pop("x0")
        states.pop("y0")
        self.states = states
        self.states["x"] = x
        self.states["y"] = y
        # self.states["acceleration"] = a
        
        self.params = params
        
        # write into dataframe
        N = len(self.states["x"])
        self.pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
        self.gt = pd.DataFrame(Y, columns = self.pts)
        self.gt['Timestamp'] = np.arange(0,N/30,1/30)
        self.gt['Frame #'] = np.arange(0,N)
        self.gt["speed"] = self.states["speed"]
        self.gt["x"] = x
        self.gt["y"] = y
        self.gt["acceleration"] = a
        self.gt["theta"] = self.states["theta"]
        self.gt["width"] = self.states["width"]
        self.gt["length"] = self.states["length"]
        self.gt['direction'] = np.sign(np.cos(self.states["theta"]))
        self.gt['ID'] = 0
        

    def downgrade(self, missing_rate, noise_std):
        '''
        create synthetically downgraded data based on downgrade parameters
        '''
        meas = self.gt.copy()
        # add noise
        Y = meas[self.pts]
        Y = Y + np.random.normal(0, noise_std, Y.shape)
        meas.loc[:, self.pts] = Y
        
        # mask missing
        if missing_rate == 0:
            missing_idx = []
        else:
            if missing_rate < 0.5:
                step = int(1/missing_rate)
                nans = np.array([False] * len(meas))
                nans[::step] = True # True is missing
                missing_idx = [i for i in range(len(nans)) if nans[i]]
            else:
                step = int(1/(1-missing_rate))
                nans = np.array([False] * len(meas))
                nans[::step] = True # True is missing
                missing_idx = [i for i in range(len(nans)) if ~nans[i]]
        meas.loc[missing_idx,["speed","acceleration","x","y","theta","width","length"]] = np.nan
        meas.loc[missing_idx, self.pts] = np.nan
        # vis.plot_track_df(meas)
        return meas
        
    def mae(self, true_state, rec_state):
        try:
            AE = [abs(true_state[i] - rec_state[i]) for i in range(len(true_state))]
        except TypeError:
            AE = abs(true_state-rec_state)
        return np.nanmean(AE)
    
    def evaluate(self, rec):
        '''
        compute errors between self.gt and rec
        return MAE for each state
        '''
        state_err = {}
        for state in self.states:
            try:
                error = self.mae(self.states[state], rec[state].values)
            except TypeError:
                error = -1
            # state_err.append(error)
            state_err[state] = error
        return state_err
    
    def score(self, meas, rec, norm='l21'):
        '''
        commpute the correction score
        '''
        pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
        Y1 = np.array(meas[pts])    
        Yre = np.array(rec[pts])
        score = opt.loss(Yre, Y1, norm)

        return score
    
    def grid_evaluate(self):
        '''
        create a 2D array to store the accuracy metrics according to missing rate and noise std
        '''
        missing_list = self.params["missing"]
        noise_list = self.params["noise"]
        
        grid = np.zeros((len(missing_list), len(noise_list))) # dummy 2D array to store the error of each state
        self.err_grid = {state: grid.copy() for state in self.states} # dict of 2D-array
        
        correction_score = np.zeros((len(missing_list), len(noise_list))) 
        
        for i,m in enumerate(missing_list):
            for j, n in enumerate(noise_list):
                print("\rEvaluating {}/{}".format(i*len(noise_list)+j+1,len(missing_list)*len(noise_list)),end = "\r",flush = True)
                for e in range(self.params["epoch"]):
                    meas = self.downgrade(m,n)
                    rec = meas.copy()
                    rec = opt.rectify_single_camera(rec, self.params["lams"])
                    state_err = self.evaluate(rec)
                    correction_score[i,j] = self.score(meas, rec,'l2')
                    vis.plot_track_compare(meas, rec)
                    vis.dashboard([meas,rec],["meas","rec"])
                    # print(m,n,state_err)
                    for state in state_err:
                        # grid[i][j] += err
                        self.err_grid[state][i][j] += state_err[state]
                    del meas, rec
        for state in state_err:# get the mean
            self.err_grid[state] /= self.params["epoch"]
        # self.err_grid = np.array(grid)/self.params["epoch"]
        self.correction_score = correction_score
        
    def runtime_analysis(self):
        missing = 0.2
        noise = 0.2
        run_time = []
        for n in self.N:
            meas = self.downgrade(missing, noise)
            start = time.time()
            rec = opt.rectify_single_camera(meas, self.params["lams"])
            end = time.time()
            run_time.append(end-start)
        self.run_time = run_time
        return
        
    def visualize(self):
        '''
        visualize the error of each state in heatmaps
        '''
        ns = len(self.states) # number of states
        
        fig, axs = plt.subplots(2,math.ceil(ns/2), figsize=(15,9), facecolor='w', edgecolor='k')
        # fig.subplots_adjust(hspace = .5, wspace=.001)
        
        axs = axs.ravel()
        fs = 16 # fontsize
        for i, state in enumerate(self.states):
            im = axs[i].imshow(self.err_grid[state], cmap='hot', interpolation='nearest')
            axs[i].set_title(state+" error",fontsize=fs)
            if i>=3:
                axs[i].set_xlabel("noise", fontsize=fs)
            axs[i].set_xticks(list(np.arange(len(self.params["noise"])))[::2])
            xlabel = ["{:.1f}".format(x) for x in self.params["noise"]]
            axs[i].set_xticklabels(xlabel[::2],fontsize=fs)
            if i%3 == 0:
                axs[i].set_ylabel("missing rate", fontsize=fs)
            axs[i].set_yticks(list(np.arange(len(self.params["missing"])))) 
            ylabel = ["{:.1f}".format(x) for x in self.params["missing"]]
            axs[i].set_yticklabels(ylabel,fontsize=fs) 

            fig.colorbar(im, ax=axs[i])
            
        if hasattr(self, "run_time"):
            plt.figure()
            plt.plot(self.N, self.run_time)
            plt.xlabel("# Frames")
            plt.ylabel("Run time (sec)")
            
        
    
if __name__ == "__main__":
    N = 40 # number of frames
    state = {"width": 2,
             "length": 4,
             "x0": 0,
             "y0": 10,
             "theta": [0] * N,
             "speed": 10*np.sin(np.arange(0,1/30*N,1/30))+30 # [30] * N
             }
    params = {"missing": [0.4], #np.arange(0,0.7,0.1), # missing rate
               "noise": [0], # np.arange(0,0.7,0.1), # gaussian noise variance on measurement boxes
              "N": N,
              "epoch": 1, # number of random runs of generate
              "lams" : (1,0,0,0,0,0)
        }
    
    ex = Experiment(state, params)
    ex.grid_evaluate()
    if isinstance(N,list): # trigger run time analysis
        ex.runtime_analysis()
        # %%
    print(N, ex.correction_score)
    ex.visualize()

    