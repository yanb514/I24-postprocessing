# -*- coding: utf-8 -*-
"""
Toy experiment 2d
- generate GT with nonlinear dynamics
- manual pollution
- rectify on x and y independently
- plot pareto curve for lambda tuning

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
import matplotlib.cm as cm


class Experiment():
    
    def __init__(self, states, params):
        '''
        state: x,y,w,l,v,theta
        '''
        for state in states:
            if "0" not in state:
                setattr(self,state,states[state])
        
        # create ground truth using simple steering dynamics up to jerk
        initial_state = [states["x0"], states["y0"], states["v0"], states["a0"]]
        x,y,theta,v,a,j = opt.generate_2d(initial_state, self.jerk, self.theta, self.dt, self.order)
        self.pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
        
        self.states = {}
        self.states["x"] = x
        self.states["y"] = y
        self.states["theta"] = theta
        self.states["speed"] = v
        self.states["acceleration"] = a
        self.states["jerk"] = j
        
        self.params = params
        
        # write into dataframe
        N = len(self.states["x"])
        self.gt = pd.DataFrame()
        self.gt['Timestamp'] = np.arange(0,N/30,1/30)
        self.gt['Frame #'] = np.arange(0,N)
        self.gt["x"] = x
        self.gt["speed"] = v
        self.gt["acceleration"] = a
        self.gt["jerk"] = j
        self.gt["y"] = y
        self.gt['direction'] = np.sign(np.cos(self.states["theta"]))
        self.gt['ID'] = 0
        self.gt['theta'] = self.states["theta"]
        
        Y = opt.generate_box(self.width,self.length, x, y, self.theta)
        self.gt.loc[:, self.pts] = Y
        
        # auxiliary states for debuggin
        vx = v * np.cos(theta)
        vy = v * np.sin(theta)
        ax = np.append(np.diff(vx)/self.dt,np.nan)
        jx = np.append(np.diff(ax)/self.dt,np.nan)

        ay = np.append(np.diff(vy)/self.dt,np.nan)
        jy = np.append(np.diff(ay)/self.dt,np.nan)
        
        self.gt.loc[:,'speed_x'] = vx
        self.gt.loc[:,'jerk_x'] = jx
        self.gt.loc[:,'acceleration_x'] = ax
        self.gt.loc[:,'speed_y'] = vy
        self.gt.loc[:,'jerk_y'] = jy
        self.gt.loc[:,'acceleration_y'] = ay 
        
        self.states['speed_x'] = vx
        self.states['jerk_x'] = jx
        self.states['acceleration_x'] = ax
        self.states['speed_y'] = vy
        self.states['jerk_y'] = jy
        self.states['acceleration_y'] = ay 
    
    def downgrade(self, missing_rate, noise_std):
        '''
        create synthetically downgraded data based on downgrade parameters
        missing_rate: float between 0 -1
        noise_std: [x_noise_std, y_noise_std]
        '''
        meas = self.gt.copy()
        # add noise
        meas["x"] += np.random.normal(0, noise_std[0], len(meas))
        meas["y"] += np.random.normal(0, noise_std[1], len(meas))
        
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
        meas.loc[missing_idx,["x","speed","acceleration","jerk","theta"]] = np.nan
        
        Y0 = opt.generate_box(self.width, self.length, meas.x.values, meas.y.values, meas.theta.values)
        pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
        meas.loc[:, pts] = Y0
        
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
        Y1 = np.array(meas[self.pts])    
        Yre = np.array(rec[self.pts])
        score = opt.loss(Yre, Y1, norm)

        return score
    
    def grid_evaluate(self):
        '''
        create a 2D array to store the accuracy metrics according to missing rate and noise std
        '''
        missing_list = self.params["missing"]
        noise_list = [list(l) for l in zip(self.params["noise_x"],self.params["noise_y"])]
        
        grid = np.zeros((len(missing_list), len(noise_list))) # dummy 2D array to store the error of each state
        self.err_grid = {state: grid.copy() for state in self.states} # dict of 2D-array
        
        # correction_score = np.zeros((len(missing_list), len(noise_list))) 
        
        for i,m in enumerate(missing_list):
            for j, n in enumerate(noise_list):
                print("\rEvaluating {}/{}".format(i*len(noise_list)+j+1,len(missing_list)*len(noise_list)),end = "\r",flush = True)
                for e in range(self.params["epoch"]):
                    meas = self.downgrade(m,n)
                    rec = meas.copy()
                    
                    rec = opt.rectify_2d(rec, self.width, self.length, self.params["args"])
                    
                    self.rec = rec
                    state_err = self.evaluate(rec)
                    vis.dashboard([self.gt,meas,rec],self.states.keys(),["gt","meas","rectified"])
                    vis.plot_track_compare(self.gt, rec, legends=["gt","rec"])
                    vis.plot_track_compare(meas, rec, legends=["meas","rec"])
                    for state in state_err:
                        # grid[i][j] += err
                        self.err_grid[state][i][j] += state_err[state]
                    del meas, rec
        for state in state_err:# get the mean
            self.err_grid[state] /= self.params["epoch"]
        # self.err_grid = np.array(grid)/self.params["epoch"]
        # self.correction_score = correction_score
        return
    
    def pareto_curve(self, missing_rate, noise_std, axis="x"):
        '''
        for lambda tuning
        for each lambda between [0,1], solve rectify_1d
        record the final objective function value for both perturbation and regularization term
        '''
        # make a copy of gt
        meas = self.downgrade(missing_rate, noise_std)
        
        x = meas[axis].values
        notNan = ~np.isnan(x)
        pert = []
        reg = []
        dt = 1/30
        N = len(meas)
        
        lam_x, lam_y, order = self.params["args"]
        lambdas = np.arange(0, 1, 0.01)
        
        for lam in lambdas:
            xhat, vxhat, axhat, jxhat = opt.rectify_1d(meas, (lam,order), axis)
            X_opt = np.concatenate((xhat,vxhat[:-1],axhat[:-2],jxhat[:-3]), axis=0)
            pert.append(opt.obj_1d(X_opt, x, order, N, dt, notNan, 1))
            reg.append(opt.obj_1d(X_opt, x, order, N, dt, notNan, 0))
        
        # compute GT point - on the selected axis only
        x_gt = self.gt[axis].values
        v_gt = np.diff(x_gt)/dt
        a_gt = np.diff(v_gt)/dt
        j_gt = np.diff(a_gt)/dt
        X_gt = np.concatenate((x_gt,v_gt,a_gt, j_gt), axis=0)
        gt_pert = opt.obj_1d(X_gt, x, order, N, dt, notNan, 1)
        gt_reg = opt.obj_1d(X_gt, x, order, N, dt, notNan, 0)
        
        print(gt_pert, gt_reg)
        
        # plot the pareto curve
        colors = cm.rainbow(np.linspace(0, 1, len(lambdas)+1))
        fig, ax = plt.subplots()
        ax.plot(pert, reg, linewidth = 0.2, color = "blue", label="rectified")

        for i in range(len(lambdas)):
            ax.scatter(pert[i], reg[i], s = 20, color =colors[i], marker = "x",label="lam={:.1f}".format(lambdas[i]) if i%10==0 else "")
        
        ax.scatter(gt_pert, gt_reg, s = 20, color = "red", marker = "o", label="GT")
        # ax.scatter(m_pert, m_reg, s = 20, color = "blue", marker = "o", label="meas")

        ax.set_xlabel("Perturbation f1")
        ax.set_ylabel("Jerk regularization f2")
        ax.legend()
        
        # plot the pareto curve wihtout lamda=0
        fig, ax = plt.subplots()
        ax.plot(pert[1:], reg[1:], linewidth = 0.2, color = "blue", label="rectified")

        for i in range(1,len(lambdas)):
            ax.scatter(pert[i], reg[i], s = 20, color =colors[i], marker = "x",label="lam={:.1f}".format(lambdas[i]) if i%10==0 else "")
        
        ax.scatter(gt_pert, gt_reg, s = 20, color = "red", marker = "o", label="GT")
        # ax.scatter(m_pert, m_reg, s = 20, color = "blue", marker = "o", label="meas")

        ax.set_xlabel("Perturbation f1")
        ax.set_ylabel("Jerk regularization f2")
        ax.legend()
        
        return
            
        
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
             "v0": 32,
             "a0": 0,
             "jerk": 1*np.sin(np.arange(0,1/10*N,1/10)),
             "theta": 0.1*np.cos(np.arange(0,1/20*N,1/20)), # theta is calculated using vx and vy
             "dt": 1/30,
             "order": 3 # highest order of derivatives in dynamics
             }
    
    params = {"missing": [0], #np.arange(0,0.7,0.1), # missing rate
              "noise_x": [0.2], # np.arange(0,0.7,0.1), # gaussian noise variance on measurement boxes
              "noise_y": [0.05], # np.arange(0,0.7,0.1),
              "N": N,
              "epoch": 1, # number of random runs of generate
              "args" : (0.1, 0.1, state["order"]) # lam,order 
        }
    
    ex = Experiment(state, params)
    ex.grid_evaluate()
    # ex.pareto_curve(params["missing"][0], [params["noise_x"],params["noise_y"]], axis="y")
    if isinstance(N,list): # trigger run time analysis
        ex.runtime_analysis()
        # %%
    # print(N, ex.correction_score)
    # ex.visualize()

    