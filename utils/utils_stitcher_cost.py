'''
data_association module connected with database
3/25: first pass of spatial_temporal_match_online ready
- parallel processes?
- how to set up queues?
'''
import numpy as np
# import torch
# from scipy import stats
from i24_logger.log_writer import catch_critical
# from utils.misc import calc_fit_select, calc_fit_select_ransac
import statsmodels.api as sm
import warnings
warnings.filterwarnings('error')

dt=0.04

def bhattacharyya_distance(mu1, mu2, cov1, cov2):
    mu = mu1-mu2
    cov = (cov1+cov2)/2
    det = np.linalg.det(cov)
    det1 = np.linalg.det(cov1)
    det2 = np.linalg.det(cov2)
    return 0.125 * np.dot(np.dot(mu.T, np.linalg.inv(cov)), mu) + 0.5 * np.log(det/np.sqrt(det1 * det2))
    

def bhattacharyya_coeff(bhatt_dist):
    return np.exp(-bhatt_dist)

def weighted_least_squares(t,x,y,weights=None):
    '''
    '''
    t = sm.add_constant(t)
    modelx = sm.WLS(x, t, weights=weights)
    resx = modelx.fit()
    fitx = [resx.params[1],resx.params[0]]
    modely = sm.WLS(y, t, weights=weights)
    resy = modely.fit()
    fity = [resy.params[1],resy.params[0]]
    return fitx, fity

@catch_critical(errors = (Exception))
def stitch_cost(track1, track2, TIME_WIN,residual_threshold_x, residual_threshold_y):
    '''
    use bhattacharyya_distance
    vectorize bhattacharyya_distance
    track t,x,y must not have nans!
    '''

    t1 = track1["timestamp"] #[filter1]
    t2 = track2["timestamp"] #[filter2]
    
    gap = t2[0] - t1[-1] 
    if gap < 0 or gap > TIME_WIN:
        return 1e6
    
    x1 = track1["x_position"]#[filter1]
    x2 = track2["x_position"]#[filter2]
    
    y1 = track1["y_position"]#[filter1]
    y2 = track2["y_position"]#[filter2]

    n1 = min(len(t1), int(1/dt)) # for track1
    n2 = min(len(t2), int(1/dt)) # for track2
        
    if len(t1) >= len(t2):
        anchor = 1 # project forward in time
        # find the new fit for anchor1 based on the last ~1 sec of data
        t1 = t1[-n1:]
        x1 = x1[-n1:]
        y1 = y1[-n1:] # TODO: could run into the danger that the ends of a track has bad speed estimate
        # fitx, fity = calc_fit_select_ransac(t1,x1,y1,residual_threshold_x, residual_threshold_y)
        # fitx, fity = calc_fit_select(t1,x1,y1)
        weights = np.linspace(1e-6, 1, len(t1)) # put more weights towards end
        fitx, fity = weighted_least_squares(t1,x1,y1,weights)
        # print(fitx, fity)
        
        # get the first chunk of track2
        meast = t2[:n2]
        measx = x2[:n2]
        measy = y2[:n2]
        pt = t1[-1] # cone starts at the end of t1

        dir = 1 # cone should open to the +1 direction in time (predict track1 to future)
        
        
    else:
        anchor = 2
        # find the new fit for anchor2 based on the first ~1 sec of track2
        t2 = t2[:n2]
        x2 = x2[:n2]
        y2 = y2[:n2]
        # fitx, fity = calc_fit_select_ransac(t2,x2,y2,residual_threshold_x, residual_threshold_y)
        # fitx, fity = calc_fit_select(t2,x2,y2)
        weights = np.linspace(1, 1e-6, len(t2)) # put more weights towards front
        fitx, fity = weighted_least_squares(t2,x2,y2,weights)
        pt = t2[0]
        # get the last chunk of tarck1
        meast = t1[-n1:]
        measx = x1[-n1:]
        measy = y1[-n1:]
        dir = -1 # use the fit of track2 to "predict" back in time
    # print(anchor, fity)
    
        
    # find where to start the cone
    try:
        tdiff = (meast - pt) * dir # tdiff should be positive
    except TypeError:
        meast = np.array(meast)
        tdiff = (meast - pt) * dir 
        
    slope, intercept = fitx
    targetx = slope * meast + intercept
    slope, intercept = fity
    targety = slope * meast + intercept
    
    sigmax = (0.1 + tdiff * 0.01) * fitx[0] # 0.1,0.1, sigma in unit ft
    varx = sigmax**2
    # sigmay = 1.5 + 2*tdiff * fity[0]
    sigmay = (1 + tdiff *0.1) * fity[0]
    # print("sigmay ", sigmay)
    # print(fity)
    vary_pred = sigmay**2
    # vary_pred = max(vary_pred, 2) # lower bound
    vary_meas = np.var(measy)
    vary_meas = max(vary_meas, 2) # lower bound 

    # vectorize!
    n = len(meast)
    mu1 = np.hstack([targetx, targety]) # 1x 2n
    mu2 = np.hstack([measx, measy]) # 1 x 2n
    cov1 = np.diag(np.hstack([varx, vary_pred])) # 2n x 2n
    cov2 = np.diag(np.hstack([np.ones(n)*varx[0], np.ones(n)*vary_meas])) 
    try:
        bd = bhattacharyya_distance(mu1, mu2, cov1, cov2)
        nll = bd/n # mean
    except:
        return 1e6
    
    # time_cost = 0.01* (np.exp(gap) - 1)
    time_cost = 0.1 * gap
    # print("id1: {}, id2: {}, cost:{:.2f}".format(str(track1['_id'])[-4:], str(track2['_id'])[-4:], nll+time_cost))
    # print("")
    
    tot_cost = nll + time_cost 
    # if TIME_WIN > 5:
    #     print(tot_cost)
        
    return tot_cost








