# -*- coding: utf-8 -*-
"""
Build upon synthetic_2d.py
- read TM_xxxx_gt.csv
- pullute it on different rate
- recover using rectify_2d

- accuracy
- runtime analysis

"""
import utils
import utils_optimization as opt
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import utils_vis as vis
# import numpy.linalg as LA
import matplotlib.cm as cm
import random

# import warnings
# warnings.simplefilter("error")

pd.options.mode.chained_assignment = None # default "warn"
    
class Experiment():
    
    def __init__(self, file_path, params):
        '''
        specify GT
        downgrade to meas
        '''
        self.params = params
        # select GT from csv
        df = pd.read_csv(file_path, nrows=params["nrows"])
        # find a lane-change vehicle
        # cars = df.groupby("ID")
        # for carid, car in cars:
        #     if len(car) > 700:
        #         print(carid)
        #     if car.lane.nunique()>1:
        #         print(carid)
        self.gt = df[df["ID"]==params["carid"]]
        print(len(self.gt))
        self.gt = self.gt.reset_index(drop=True)
        
        self.pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
        vx,vy,ax,ay,jx,jy = opt.decompose_2d(self.gt)
        self.gt.loc[:,"speed_x"] = vx
        self.gt.loc[:,"speed_y"] = vy
        self.gt.loc[:,"acceleration_x"] = ax
        self.gt.loc[:,"acceleration_y"] = ay
        self.gt.loc[:,"jerk_x"] = jx
        self.gt.loc[:,"jerk_y"] = jy
        
        if isinstance(params["N"],list): # if N is a list, trigger run time analysis
            self.N = [x for x in params["N"] if x < len(self.gt)]
        else:
            self.N = params["N"] # as default length for evaluate_single
            self.gt = self.gt[:self.N]
        self.meas = self.gt.copy()
        
        # states to be plotted
        # self.states = ["x","y","speed","speed_x","speed_y","acceleration_x","acceleration_y","jerk_x","jerk_y","theta","acceleration","jerk"]
        self.states = ["x","y","speed","acceleration","jerk","theta"]
        self.units = {
                  **dict.fromkeys(['x', 'y'], "m"), 
                  **dict.fromkeys(['speed', 'speed_x', 'speed_y'], 'm/s'),
                  **dict.fromkeys(['acceleration', 'acceleration_x', 'acceleration_y'], 'm/s2'),
                  **dict.fromkeys(['jerk', 'jerk_x', 'jerk_y'], 'm/s3'),
                  **dict.fromkeys(['theta'], 'rad'),
                }
    
    def pollute_car(self, car):
        '''
        AVG_CHUNK_LENGTH: Avg length (in # frames) of missing chunk mask
        OUTLIER_RATIO: ratio of bbox in each trajectory are outliers (noisy measurements)
        Assume each trajectory is manually chopped into 0.01N fragments, where N is the length of the trajectory
            Mark the IDs of the fragments as xx000, e.g., if the GT ID is 9, the fragment IDs obtained from track 9 are 9000, 9001, 9002, etc.
            This is assign a unique ID to each fragments.
            
        '''
        car = self.downgrade(car, self.params["missing"],[self.params["noise_x"],self.params["noise_y"]])
        AVG_CHUNK_LENGTH,OUTLIER_RATIO = self.params["AVG_CHUNK_LENGTH"],self.params["OUTLIER_RATIO"]
        
        if AVG_CHUNK_LENGTH ==0 and OUTLIER_RATIO==0: # no pollution
            return car
        
        car=car.reset_index(drop=True)
        l = car["length"].iloc[0]
        w = car["width"].iloc[0]
        id = car["ID"].iloc[0] # original ID
        
        # mask chunks
        if AVG_CHUNK_LENGTH > 0:
            n_chunks = int(len(car)*0.01)
            for index in sorted(random.sample(range(0,len(car)),n_chunks)):
                to_idx = max(index, index+AVG_CHUNK_LENGTH+np.random.normal(0,20)) # The length of missing chunks follow Gaussian distribution N(AVG_CHUNK_LENGTH, 20)
                car.loc[index:to_idx, self.pts] = np.nan # Mask the chunks as nan to indicate missing detections
                # if id>=1000: id+=1 # assign unique IDs to fragments
                # else: id*=1000
                # car.loc[to_idx:, ["ID"]] = id
            
        # add outliers (noise)
        if OUTLIER_RATIO > 0:
            outlier_idx = random.sample(range(0,len(car)),int(OUTLIER_RATIO*len(car))) # randomly select 0.01N bbox for each trajectory to be outliers
            for idx in outlier_idx:
                noise = np.random.multivariate_normal([0,0,0,0,0,0,0,0], np.diag([0.3*l, 0.1*w]*4)) # add noises to each outlier box
                car.loc[idx, self.pts] += noise
            car.loc[outlier_idx, ["Generation method"]] = "outlier"
        
        # update x and y
        x = (car["bbr_x"].values + car["bbl_x"].values)/2
        y = (car["bbr_y"].values + car["bbl_y"].values)/2
        car.loc[:,"x"] = x
        car.loc[:,"y"] = y
        
        # delete column values
        columns_to_delete = ["speed","acceleration","width","length","theta"]
        car.loc[:,columns_to_delete] = ""
        return car
    
    def downgrade(self, meas, missing_rate, noise_std):
        '''
        create synthetically downgraded data based on downgrade parameters
        missing_rate: float between 0 -1
        noise_std: [x_noise_std, y_noise_std]
        '''
        meas=meas.reset_index(drop=True)
        # meas = self.gt.copy()
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
        meas.loc[missing_idx,["x","y","speed","acceleration","jerk","theta"]] = np.nan
        
        Y0 = opt.generate_box(meas.width.values[0], meas.length.values[0], meas.x.values, meas.y.values, meas.theta.values)
        pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
        meas.loc[:, pts] = Y0
        
        # vis.plot_track_df(meas)
        return meas
        
    def mae(self, true_state, rec_state, max = False):
        try:
            AE = [abs(true_state[i] - rec_state[i]) for i in range(len(true_state))]
        except TypeError:
            AE = abs(true_state-rec_state)
        if max:
            return np.nanmax(AE)
        return np.nanmean(AE)
    
    def evaluate(self, rec):
        '''
        compute errors between self.gt and rec
        return MAE for each state
        '''
        state_err = {}
        for state in self.states:
            try:
                error = self.mae(self.gt[state].values, rec[state].values, max=False)
            except:
                error = -1
            # state_err.append(error)
            state_err[state] = error
        return state_err
    
    def evaluate_single(self):
        if isinstance(self.N,list):
            print("In runtime analysis mode")
            return
        self.gt = self.gt[:self.N]
        self.meas = self.meas[:self.N]
        self.meas = self.pollute_car(self.meas)
        # newmeas = self.meas.copy()
        
        # modify meas first with box fitting
        width_array = np.abs(self.meas["bbr_y"].values - self.meas["bbl_y"].values)
        length_array = np.abs(self.meas["bbr_x"].values - self.meas["fbr_x"].values)
        
        width = np.nanmedian(width_array)
        length = np.nanmedian(length_array)

        # newmeas = opt.box_fitting(newmeas, width, length) # modify x and y
        # newmeas = utils.mark_outliers(newmeas)
        # newmeas.loc[newmeas["Generation method"]=="outlier1",self.pts+["x","y"]] = np.nan
        self.meas = opt.decompose_2d(self.meas, write_to_df=True)
        
        rec = self.meas.copy()
        # rec = opt.rectify_single_car(rec, self.params["args"])
        start = time.time()
        rec = opt.rectify_2d(rec, width, length, self.params["args"])
        print("Rectify_2d: ", time.time()-start)
        self.rec = rec

        vis.dashboard([self.gt,self.meas,rec],self.states,["gt","meas","rectified"])
        # vis.plot_track_compare(self.gt, rec, legends=["gt","rec"])
        # vis.plot_track_compare(self.newmeas, rec, legends=["meas_no_outlier","rec"])
        # vis.plot_track_compare(self.meas, rec, legends=["meas","rec"])
        return
        
    def receding_horizon_rectify(self):
        
        
        width_array = np.abs(self.meas["bbr_y"].values - self.meas["bbl_y"].values)
        length_array = np.abs(self.meas["bbr_x"].values - self.meas["fbr_x"].values)
        
        width = np.nanmedian(width_array)
        length = np.nanmedian(length_array)
        
        lamx,lamy,order  = self.params["args"]
        args = (lamx, lamy, self.params["PH"], self.params["IH"]) #lam, order, PH, IH
        
        rec = self.meas.copy()
        start = time.time()
        rec = opt.receding_horizon_2d(rec,width, length, args)
        print("Receding horizon 2d: ", time.time()-start)
        # rec.loc[:,[axis,"speed_"+axis,"acceleration_"+axis,"jerk_"+axis]] = np.array([xfinal, vfinal, afinal, jfinal]).T
        
        vis.dashboard([self.gt[:self.N],self.meas[:self.N],rec],self.states,["gt","meas","rectified"])
        self.rec = rec
        return
            
    def grid_evaluate(self):
        '''
        create a 2D array to store the accuracy metrics according to missing rate and noise std
        '''
        missing_list = self.params["missing"]
        noise_list = [list(l) for l in zip(self.params["noise_x"],self.params["noise_y"])]
        
        grid = np.zeros((len(missing_list), len(noise_list))) # dummy 2D array to store the error of each state
        self.err_grid = {state: grid.copy() for state in self.states} # dict of 2D-array
        
        width_array = np.abs(self.meas["bbr_y"].values - self.meas["bbl_y"].values)
        length_array = np.abs(self.meas["bbr_x"].values - self.meas["fbr_x"].values)
        
        width = np.nanmedian(width_array)
        length = np.nanmedian(length_array)
        
        # correction_score = np.zeros((len(missing_list), len(noise_list))) 
        
        for i,m in enumerate(missing_list):
            for j, n in enumerate(noise_list):
                print("\rEvaluating {}/{}".format(i*len(noise_list)+j+1,len(missing_list)*len(noise_list)),end = "\r",flush = True)
                for e in range(self.params["epoch"]):
                    meas = self.gt.copy()
                    meas = self.downgrade(meas, m,n)
                    rec = opt.rectify_2d(meas, width, length, self.params["args"])
                    
                    self.rec = rec
                    state_err = self.evaluate(rec)
                    # vis.dashboard([self.gt,meas,rec],self.states,["gt","meas","rectified"])
                    # vis.plot_track_compare(self.gt, rec, legends=["gt","rec"])
                    # vis.plot_track_compare(meas, rec, legends=["meas","rec"])
                    for state in state_err:
                        # grid[i][j] += err
                        self.err_grid[state][i][j] += state_err[state]
                    del meas, rec
        for state in state_err:# get the mean
            self.err_grid[state] /= self.params["epoch"]
        # self.err_grid = np.array(grid)/self.params["epoch"]
        # self.correction_score = correction_score
        return
    
    def pareto_curve(self, axis="x"):
        '''
        for lambda tuning
        for each lambda between [0,1], solve rectify_1d
        record the final objective function value for both perturbation and regularization term
        '''
        # make a copy of gt
        # meas = self.downgrade(missing_rate, noise_std)
        def _l2(z):
            return np.linalg.norm(z, 2)**2
        
        try:
            meas = self.newmeas
        except:
            meas = self.meas
        x = meas[axis].values
        notNan = ~np.isnan(x)
        pert = []
        reg = []
        dt = 1/30
        
        lam_x, lam_y = self.params["args"]
        if axis == "x":
            lambdas = np.logspace(-1,1.5, num=20)
        else:
            lambdas = np.logspace(-3,1, num=20)
            # lambdas = np.arange(1e-3,5,0.2)
        # lambdas = [1]
        for lam in lambdas:
            xhat, vxhat, axhat, jxhat = opt.rectify_1d(meas, lam, axis)
            xhat,jxhat = np.reshape(xhat,-1), np.reshape(jxhat,-1)
            c1 = _l2(x[notNan]-xhat[notNan])
            c2 = _l2(jxhat)
            pert.append(c1)
            reg.append(c2)
        # compute GT point - on the selected axis only
        x_gt = self.gt[axis].values
        v_gt = np.diff(x_gt)/dt
        a_gt = np.diff(v_gt)/dt
        j_gt = np.diff(a_gt)/dt

        gt_pert = _l2(x[notNan]-x_gt[notNan])
        gt_reg = _l2(j_gt)
        
        # plot the pareto curve
        colors = cm.rainbow(np.linspace(0, 1, len(lambdas)+1))
        # fig, ax = plt.subplots()
        # ax.plot(pert, reg, linewidth = 0.2, color = "blue", label="rectified")

        # for i in range(len(lambdas)):
        #     ax.scatter(pert[i], reg[i], s = 20, color =colors[i], marker = "x",label="lam={:.1f}".format(lambdas[i]) if i%5==0 else "")
        
        # ax.scatter(gt_pert, gt_reg, s = 30, color = "black", marker = "o", label="GT")

        # ax.set_xlabel("||x-\hat{x}||_2^2")
        # ax.set_ylabel("||\hat{j}||_2^2")
        # ax.legend()
        
        # plot the pareto curve wihtout lamda=1
        fig, ax = plt.subplots()
        ax.plot(pert[1:-1], reg[1:-1], linewidth = 0.2, color = "blue", label="rectified")

        for i in range(1,len(lambdas)-1):
            ax.scatter(pert[i], reg[i], s = 20, color =colors[i], marker = "x",label="lam={:.1f}".format(lambdas[i]) if i%2==0 else "")
        
        ax.scatter(gt_pert, gt_reg, s = 30, color = "black", marker = "o", label="GT")

        ax.set_xlabel("||x-\hat{x}||_2^2")
        ax.set_ylabel("||\hat{j}||_2^2")
        ax.legend()
        
        return
            
        
    def runtime_analysis(self):

        run_time = []
        for n in self.N:
            meas = self.gt[:n].copy()
            meas = self.pollute_car(meas)
            start = time.time()
            rec = opt.rectify_single_car(meas, self.params["args"])
            end = time.time()
            run_time.append(end-start)
            del meas
        self.run_time = run_time
        
        # plot
        fig, ax = plt.subplots()
        ax.plot(self.N, run_time, color = "blue")
        ax.scatter(self.N, run_time, s = 10, color = "blue")
        ax.set_xlabel("Length in Frame #")
        ax.set_ylabel("Run time in sec")
        
        return
        
    def visualize(self):
        '''
        visualize the error of each state in heatmaps
        '''
        ns = len(self.states) # number of states
        
        if hasattr(self, "err_grid"):
            fig, axs = plt.subplots(2,math.ceil(ns/2), figsize=(15,9), facecolor='w', edgecolor='k')
            # fig.subplots_adjust(hspace = .5, wspace=.001)
            
            axs = axs.ravel()
            fs = 12 # fontsize
            for i, state in enumerate(self.states):
                im = axs[i].imshow(self.err_grid[state], cmap='hot', interpolation='nearest')
                axs[i].set_title(state+"("+self.units[state]+")",fontsize=fs)
                if i>=3:
                    axs[i].set_xlabel("noise std x/y", fontsize=fs)
                axs[i].set_xticks(list(np.arange(len(self.params["noise_x"])))[::2])
                xlabel = ["{:.1f}/{:.2f}".format(self.params["noise_x"][i],self.params["noise_y"][i]) for i in range(len(self.params["noise_x"]))]
                axs[i].set_xticklabels(xlabel[::2],fontsize=fs, rotation=45)
                if i%3 == 0:
                    axs[i].set_ylabel("missing rate", fontsize=fs)
                axs[i].set_yticks(list(np.arange(len(self.params["missing"])))) 
                ylabel = ["{:.1f}".format(x) for x in self.params["missing"]]
                axs[i].set_yticklabels(ylabel,fontsize=fs) 
    
                fig.colorbar(im, ax=axs[i])
            
        
    
if __name__ == "__main__":
    # N = list(np.arange(10,1400, 50)) # number of frames
    N = 500
    file_path = r"E:\I24-postprocess\benchmark\TM_1000_GT.csv"
    
    params = {"missing": np.arange(0,0.7,0.1), # missing rate
               "noise_x": np.arange(0,0.7,0.1), # gaussian noise variance on measurement boxes
               "noise_y": .1*np.arange(0,0.7,0.1),
              "N": N,
              "epoch": 1, # number of random runs of generate
                "args" : (7,1e-1), # lamx,lamy - no l1 reg
              # "args" : (7,7, 1), # deltax, deltay, lam - for l1 reg (delta=0 for no outliers)
              "carid": 16, # 16, 38, for lane change
              "nrows": 20000,
              "AVG_CHUNK_LENGTH": 0,
              "OUTLIER_RATIO": 0,
              "PH": 200,
              "IH": 100
        }
    
    ex = Experiment(file_path, params)
    # ex.evaluate_single()
    ex.grid_evaluate()
    # ex.pareto_curve(axis="y")
    # ex.receding_horizon_rectify()
    if isinstance(N,list): # trigger run time analysis
        ex.runtime_analysis()
    #     # %%
    ex.visualize()

    