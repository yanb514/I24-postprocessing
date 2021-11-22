# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 14:52:57 2021

@author: wangy79
"""
import numpy as np
from numpy import sin,cos
from scipy.optimize import basinhopping
import numpy.linalg as LA
import utils
from tqdm import tqdm
import utils_vis as vis
import matplotlib.pyplot as plt
import utils_evaluation as ev

class Rectification():
    
    def __init__(self, data_path, params = None):
        self.df = utils.read_data(data_path)
        self.df = self.df[(self.df["Frame #"] >= params["start"]) & (self.df["Frame #"] <= params["end"])]
        self.original = self.df.copy()
        self.params = params
        self.data = {}
        return
        
    def obj1(self, X, Y1,N,dt,notNan, lam1,lam2,lam3,lam4,lam5):
        """The cost function
            X = [a,theta,v0,x0,y0,w,l]^T
            penalize only theta, correction and accel
            pretty accurate and faster than previous formulation
        """ 
        nvalid = np.count_nonzero(notNan)
        # unpack
        v = X[:N]
        theta = X[N:2*N]
        x0,y0,w,l = X[2*N:]
        
        Yre,x,y,a = self.generate(w,l,x0, y0, theta,v, outputall=True)
    
        # min perturbation
        diff = Y1-Yre[notNan,:]
        # c1 = lam1*np.nanmean(LA.norm(diff,axis=1)) # L2 norm
        # c1 = lam1*np.nanmean(LA.norm(diff,ord=1,axis=1)) # L1 norm
        # weighted distance
        mae_x = np.abs(diff[:,[0,2,4,6]])
        mae_y = np.abs(diff[:,[1,3,5,7]]) 
        alpha = 0.3
        mae_xy = alpha*mae_x + (1-alpha)*mae_y
        c1 = lam1*LA.norm(mae_xy,ord=1)
        
        # regularize acceleration # not the real a or j, multiply a constant dt
        c2 = lam2*LA.norm(a,2)/nvalid/30
        
        # regularize jerk
        j = np.diff(a)/dt
    
        c3 = lam3*LA.norm(j,2)/nvalid /900
        # regularize angle
        st = sin(theta)
        c4 = lam4*LA.norm(st,2)/nvalid
        
        # regularize angular velocity
        o = np.diff(theta)/dt
        c5 = lam5*LA.norm(o,2)/nvalid/30
    
        return c1+c2+c3+c4+c5
        
    def get_costs(self, Yre, Y1, x,y,v,a,theta,notNan, cmax=None):
        if cmax: # using normalization
            c1m, c2m, c3m, c4m, c5m = cmax
        else: # not using normalization
            c1m, c2m, c3m, c4m, c5m = 1,1,1,1,1 # directly calculate 2-norm
        dt = 1/30
        diff = Y1-Yre[notNan,:]
        c1 = np.nanmean(LA.norm(diff,axis=1))/c1m
        c2 = LA.norm(a,2)/np.count_nonzero(notNan)/30/c2m
        j = np.diff(a)/dt
        c3 = LA.norm(j,2)/np.count_nonzero(notNan)/900/c3m
        st = sin(theta)
        c4 = LA.norm(st,2)/np.count_nonzero(notNan)/c4m
        o = np.diff(theta)/dt
        c5 = LA.norm(o,2)/np.count_nonzero(notNan)/30/c5m
        return c1,c2,c3,c4,c5
    
        
    def unpack1(self, res,N,dt):
        # extract results
        # unpack variables
                            
        x = np.zeros(N)            
        y = np.zeros(N) 
        
        # ver2 unpack
        v = res.x[:N]
        theta = res.x[N:2*N]
        x0,y0,w,l = res.x[2*N:]
        
        Yre, x, y, a = self.generate(w,l,x0, y0, theta,v, outputall=True)    
        return Yre, x,y,v,a,theta,w,l
                                
    
    def rectify_single_camera(self, df, args):
        '''                        
        df: a single track in one camera view
        '''             
        lam1, lam2, lam3,lam4,lam5,niter = args
    
        # optimization parameters
        lam1 = lam1# modification of measurement 1000
        lam2 = lam2 # acceleration 0
        lam3 = lam3 # jerk 0.1      
        lam4 = lam4 # theta 1000      
        lam5 = lam5 # omega 2     
        niter = niter
        
        timestamps = df.Timestamp.values
        dt = np.diff(timestamps)
        # sign = df["direction"].iloc[0]
        sign = np.sign(df.x.values[-1]-df.x.values[0])
        
        # get bottom 4 points coordinates
        pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
        Y1 = np.array(df[pts])    
    
        N = len(Y1)                
        notNan = ~np.isnan(np.sum(Y1,axis=-1))
        Y1 = Y1[notNan,:]
    
        if (len(Y1) <= 3):        
            print('Not enough valid measurements: ', df['ID'].iloc[0])
            # df.loc[:,pts] = np.nan
            return None 
        
        # reorder Y1 to deal with backward traveling measurements
        # new_order = np.argsort(np.sum(Y1[:, [0,2,4,6]],axis=1))[::int(sign)]
        # Y1 = Y1[new_order,:]
        
        first_valid = np.where(notNan==True)[0][0]
        
        temp = df[~df["bbr_x"].isna()]
        v_bbr = (max(temp.bbr_x.values)-min(temp.bbr_x.values))/(max(temp.Timestamp.values)-min(temp.Timestamp.values))
        v_fbr = (max(temp.fbr_x.values)-min(temp.fbr_x.values))/(max(temp.Timestamp.values)-min(temp.Timestamp.values))
        # avgv = max(min(v_bbr,50), min(v_fbr,50))
        avgv = (v_bbr+v_fbr)/2
        # print(avgv)
        v0 = np.array([np.abs(avgv)]*N)            
    
        x0 = (Y1[0,0]+Y1[0,6])/2- sign*avgv*first_valid*1/30
        y0 = (Y1[0,1]+Y1[0,7])/2
        dy = Y1[-1,1]-Y1[0,1]
        dx = Y1[-1,0]-Y1[0,0]
        theta0 = np.ones((N))*np.arccos(sign) # parallel to lane
        # theta0 = np.ones((N))*np.arctan2(dy,dx) # average angle
        
        # no perfect box exists    
        w0 = np.nanmean(np.sqrt((Y1[:,1]-Y1[:,7])**2+(Y1[:,0]-Y1[:,6])**2))
        l0 = np.nanmean(np.sqrt((Y1[:,2]-Y1[:,0])**2+(Y1[:,1]-Y1[:,3])**2))
        X0 = np.concatenate((v0.T, theta0.T, \
                     [x0,y0,w0,l0]),axis=-1)
    
        bnds = [(0,50) for i in range(0,N)]+\
                [(-np.pi/8+np.arccos(sign),np.pi/8+np.arccos(sign)) for i in range(N)]+\
                [(-np.inf,np.inf),(0,np.inf),(1,4),(2,np.inf)]        
        Y0 = self.generate(w0,l0,x0, y0, theta0,v0)
        diff = Y1-Y0[notNan,:]
        c1max = np.nanmean(LA.norm(diff,axis=1))
        c1max = max(c1max, 1e-4)
        
        # SOLVE FOR MAX C2-C5 BY SETTING LAM2-5 = 0
        lams = (100,0,0,0,0)
        minimizer_kwargs = {"method":"L-BFGS-B", "args":(Y1,N,dt,notNan,*lams),'bounds':bnds,'options':{'disp': False}}
        res = basinhopping(self.obj1, X0, minimizer_kwargs=minimizer_kwargs,niter=0)
    
        # extract results    
        Yre, x,y,v,a,theta,w,l = self.unpack1(res,N,dt)
        _,c2max,c3max,c4max,c5max = self.get_costs(Yre, Y1, x,y,v,a,theta,notNan)
        c2max,c3max,c4max,c5max = max(c2max, 1e-4), max(c3max, 1e-4), max(c4max, 1e-4), max(c5max, 1e-4)
        # SOLVE AGAIN - WITH NORMALIZED OBJECTIVES
        lams = (lam1/c1max,lam2/c2max,lam3/c3max,lam4/c4max,lam5/c5max)
        minimizer_kwargs = {"method":"L-BFGS-B", "args":(Y1,N,dt,notNan,*lams),'bounds':bnds,'options':{'disp': False}}
        res = basinhopping(self.obj1, X0, minimizer_kwargs=minimizer_kwargs,niter=niter)
        Yre, x,y,v,a,theta,w,l = self.unpack1(res,N,dt)
        
        df.loc[:,pts] = Yre        
        df.loc[:,'acceleration'] = a
        df.loc[:,'speed'] = v    
        df.loc[:,'x'] = x        
        df.loc[:,'y'] = y        
        df.loc[:,'theta'] = theta
        df.loc[:,'width'] = w    
        df.loc[:,'length'] = l       
        
        return df
                                
    
                                
    def rectify(self):            
        '''                        
        apply solving obj1 for each objects in the entire dataframe
        '''                        
        print('Rectifying...')    
      
        self.df = self.df.groupby("ID").filter(lambda x: len(x)>=2)
        tqdm.pandas()            
       
        self.df = utils.applyParallel(self.df.groupby("ID"), self.rectify_single_camera, args = self.params["lambda"]).reset_index(drop=True)
        # df = df.groupby('ID').apply(rectify_single_camera, args=lams).reset_index(drop=True)
        return                
                                

    def generate(self,w,l,x0, y0, theta,v,outputall=False):
        # extract results
        # unpack variables
        N = len(theta)
        dt = [1/30]*N
        
        vx = v*cos(theta)
        vy = v*sin(theta)
        a = np.diff(v)
        a = np.append(a,a[-1])
        a = a/dt
        
        x = np.zeros(N)
        y = np.zeros(N)
        x[0] = x0
        y[0] = y0
        for k in range(0,N-1):
            x[k+1] = x[k] + vx[k]*dt[k]
            y[k+1] = y[k] + vy[k]*dt[k]
    
        # compute positions
        xa = x + w/2*sin(theta)
        ya = y - w/2*cos(theta)
        xb = xa + l*cos(theta)
        yb = ya + l*sin(theta)
        xc = xb - w*sin(theta)
        yc = yb + w*cos(theta)
        xd = xa - w*sin(theta)
        yd = ya + w*cos(theta)
    
        Yre = np.stack([xa,ya,xb,yb,xc,yc,xd,yd],axis=-1)
        if outputall:
            return Yre, x, y, a
        return Yre
    
    
    def visualize_time_space(self, lanes=[1,2,3,4,7,8,9,10]):
        
        for lane_idx in lanes:
            fig, axs = plt.subplots(1,2, figsize=(15,5), facecolor='w', edgecolor='k')
            axs = axs.ravel()
            vis.plot_time_space(self.original, lanes=[lane_idx], time="frame", space="x", ax=axs[0])
            vis.plot_time_space(self.df, lanes=[lane_idx], time="frame", space="x", ax=axs[1])
            fig.tight_layout() 
        return
    
    
    def visualize(self,TIME_SPACE = True, SPEED_DIST = True, ACCELERATION_DIST = True, SCORE_DIST = True):
        if TIME_SPACE:
            self.visualize_time_space()
            
        if SPEED_DIST:
            vis.plot_histogram(self.df.speed.values, bins=40,
                                   labels=[""], 
                                   xlabel= "Speed (m/s)", 
                                   ylabel= "Probability", 
                                   title= "Speed distribution")
        if ACCELERATION_DIST:
            vis.plot_histogram(self.df.acceleration.values, bins=40,
                                   labels=[""], 
                                   xlabel= "Acceleration (m/s2)", 
                                   ylabel= "Probability", 
                                   title= "Acceleration distribution")
        if SCORE_DIST:
            score = ev.get_correction_score(self.original, self.df)
            self.data["correction_score"]=score
            vis.plot_histogram(score.values(), bins=40,
                                   labels=[""], 
                                   xlabel= "Score", 
                                   ylabel= "Probability", 
                                   title= "Score distribution")
        return
     
    def postprocess(self, REMOVE_COLLISION=False, EXTEND=False, SAVE=""):
        if not self.params["postprocess"]:
            return
        
        if REMOVE_COLLISION:
            print('remove overlapped cars...')
            # id_rem = get_id_rem(df, SCORE_THRESHOLD=0) # TODO: untested threshold
            # df = df.groupby(['ID']).filter(lambda x: (x['ID'].iloc[-1] not in id_rem))
            # print('cap width at 2.59m...')
            # self.df = self.df.groupby("ID").apply(width_filter).reset_index(drop=True)
            
            # print('remove overlaps, before: ', len(df['ID'].unique()))
            # df = da.remove_overlaps(df)
            # print('remove overlaps, after: ', len(df['ID'].unique()))
            
        if EXTEND:
            print('extending tracks to edges of the frame...')
            xmin, xmax = np.min(self.df.x.values), np.max(self.df.x.values)
            maxFrame = max(self.df['Frame #'])
            args = (xmin, xmax, maxFrame)
            tqdm.pandas()
            # df = df.groupby('ID').apply(extend_prediction, args=args).reset_index(drop=True)
            self.df = utils.applyParallel(self.df.groupby("ID"), utils.extend_prediction, args=args).reset_index(drop=True)
            
        print('standardize format for plotter...')
        # if ('lat' in df):
            # df = df.drop(columns=['lat','lon'])
        self.df = self.df[['Frame #', 'Timestamp', 'ID', 'Object class', 'BBox xmin','BBox ymin','BBox xmax','BBox ymax',
                'vel_x','vel_y','Generation method',
                'fbrx','fbry','fblx','fbly','bbrx','bbry','bblx','bbly','ftrx','ftry','ftlx','ftly','btrx','btry','btlx','btly',
                'fbr_x','fbr_y','fbl_x','fbl_y','bbr_x','bbr_y','bbl_x','bbl_y',
                'direction','camera','acceleration','speed','x','y','theta','width','length','height',"ts_bias for cameras ['p1c2', 'p1c3', 'p1c4']",'lane']]
        
        if len(SAVE)>0:
            self.df.to_csv(SAVE, index = False)
            
        return

    def evaluate_single_track(self, carid, plot=True, dashboard=True):
        '''
        identify a problematic track
        '''
        car = self.original[self.original["ID"]==carid]
        carre = self.df[self.df["ID"]==carid]
        
        if plot:
            vis.plot_track_compare(car,carre)
        if dashboard:
            vis.dashboard([car, carre], legends=["before","rectified"])
        return
    
if __name__ == "__main__":

    data_path = r"E:\I24-postprocess\MC_tracking" 
    file_path = data_path+r"\DA\MC_tsmn.csv"
    
    params = {"norm": "l1",
              "start": 0,
              "end": 1000,
              "plot_start": 0, # for plotting tracks in online methods
              "plot_end": 10,
              "postprocess": True,
              "lambda": (1,0,0,0.1,0.1,0)
              }
    
    re = Rectification(file_path, params)
    re.rectify()
    #%%
    re.postprocess(REMOVE_COLLISION=False, 
                   EXTEND = True,
                    SAVE = data_path+r"\rectified\MC_tsmn.csv"
                    # SAVE = ""
                    )
    #%%
    # re.visualize(TIME_SPACE = True,
    #               SPEED_DIST = True,
    #               ACCELERATION_DIST = True,
    #               SCORE_DIST = True)
    # %%
    # re.evaluate_single_track(216, plot=True, dashboard=True)
    # # %%
    # car = re.original[re.original["ID"]==17]
    # carre = car.copy()
    # carre = re.rectify_single_camera(carre, (1,0,0,0.1,0.1,0))
    # vis.plot_track_compare(car,carre)
    # vis.dashboard([car,carre],["original","rectified"])