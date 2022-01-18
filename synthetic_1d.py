# -*- coding: utf-8 -*-
"""
Toy experiment 1d

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
        # Y, x, y, a = opt.generate(self.width, self.length, states["x0"], states["y0"], self.theta, self.speed, outputall=True)
        x,vx,ax,jx = opt.generate_1d([states["x0"], states["v0_x"], states["a0_x"]], self.jerk_x, self.dt, self.order)
        y,vy,ay,jy = opt.generate_1d([states["x0"], states["v0_y"], states["a0_y"]], self.jerk_y, self.dt, self.order)
        
        self.states = {}
        self.states["x"] = x
        self.states["speed_x"] = vx
        self.states["acceleration_x"] = ax
        self.states["jerk_x"] = jx
        self.states["y"] = y
        self.states["speed_y"] = vy
        self.states["acceleration_y"] = ay
        self.states["jerk_y"] = jy
        
        v = np.sqrt(vx**2+vy**2)
        self.states["theta"] = np.arcsin(vy/v) # preserve the direction
        
        self.params = params
        
        # write into dataframe
        N = len(self.states["x"])
        self.gt = pd.DataFrame()
        self.gt['Timestamp'] = np.arange(0,N/30,1/30)
        self.gt['Frame #'] = np.arange(0,N)
        self.gt["x"] = x
        self.gt["speed_x"] = vx
        self.gt["acceleration_x"] = ax
        self.gt["jerk_x"] = jx
        self.gt["y"] = y
        self.gt["speed_y"] = vy
        self.gt["acceleration_y"] = ay
        self.gt["jerk_y"] = jy
        
        self.gt['direction'] = np.sign(np.cos(self.states["theta"]))
        self.gt['ID'] = 0
        
        self.gt['theta'] = self.states["theta"]
        

    def downgrade(self, missing_rate, noise_std):
        '''
        create synthetically downgraded data based on downgrade parameters
        '''
        meas = self.gt.copy()
        # add noise
        meas["x"] += np.random.normal(0, noise_std, len(meas))
        
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
        meas.loc[missing_idx,["speed","acceleration","x","jerk"]] = np.nan
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
            except:
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
        
        # correction_score = np.zeros((len(missing_list), len(noise_list))) 
        
        for i,m in enumerate(missing_list):
            for j, n in enumerate(noise_list):
                print("\rEvaluating {}/{}".format(i*len(noise_list)+j+1,len(missing_list)*len(noise_list)),end = "\r",flush = True)
                for e in range(self.params["epoch"]):
                    meas = self.downgrade(m,n)
                    rec = meas.copy()
                    
                    rec = opt.rectify_2d(rec, self.params["args"])
                    self.rec = rec
                    state_err = self.evaluate(rec)
                    vis.dashboard([meas,rec],self.states.keys(),["gt","rectified"])

                    for state in state_err:
                        # grid[i][j] += err
                        self.err_grid[state][i][j] += state_err[state]
                    del meas, rec
        for state in state_err:# get the mean
            self.err_grid[state] /= self.params["epoch"]
        # self.err_grid = np.array(grid)/self.params["epoch"]
        # self.correction_score = correction_score
        
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
    N = 50 # number of frames
    # specify GT initial states and highest_order dynamics s.t. GT states can be simulated 
    state = {"width": 2,
             "length": 4,
             "x0": 0,
             "y0": 10,
             "v0_x": 32,
             "v0_y": 0,
             "a0_x": 0,
             "a0_y": 0,
             "jerk_x": 2*np.sin(np.arange(0,1/10*N,1/10)),
             "jerk_y": 0.1*np.sin(np.arange(0,1/10*N,1/10)),
             # "theta": [0] * N, # theta is calculated using vx and vy
             "dt": 1/30,
             "order": 3 # highest order of derivatives in dynamics
             }
    
    params = {"missing": [0], #np.arange(0,0.7,0.1), # missing rate
              "noise": [0], # np.arange(0,0.7,0.1), # gaussian noise variance on measurement boxes
              "N": N,
              "epoch": 1, # number of random runs of generate
              "args" : (0.7,0,state["order"]) # lam,niter 
        }
    
    ex = Experiment(state, params)
    ex.grid_evaluate()
    if isinstance(N,list): # trigger run time analysis
        ex.runtime_analysis()
        # %%
    # print(N, ex.correction_score)
    # ex.visualize()

    