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


def bhattacharyya_distance(mu1, mu2, cov1, cov2):
    mu = mu1-mu2
    cov = (cov1+cov2)/2
    det = np.linalg.det(cov)
    det1 = np.linalg.det(cov1)
    det2 = np.linalg.det(cov2)
    return 0.125 * np.dot(np.dot(mu.T, np.linalg.inv(cov)), mu) + 0.5 * np.log(det/np.sqrt(det1 * det2))
    

def bhattacharyya_coeff(bhatt_dist):
    return np.exp(-bhatt_dist)



@catch_critical(errors = (Exception))
def cost_3(track1, track2, TIME_WIN, VARX, VARY):
    '''
    use bhattacharyya_distance
    '''
    
    cost_offset = 0

    # filter1 = np.array(track1["filter"], dtype=bool) # convert fomr [1,0] to [True, False]
    # filter2 = np.array(track2["filter"], dtype=bool)
    
    t1 = np.array(track1["timestamp"])#[filter1]
    t2 = np.array(track2["timestamp"])#[filter2]
    
    x1 = np.array(track1["x_position"])#[filter1]
    x2 = np.array(track2["x_position"])#[filter2]
    
    y1 = np.array(track1["y_position"])#[filter1]
    y2 = np.array(track2["y_position"])#[filter2]

    
    # if time_gap > TIME_WIN, don't stitch
    gap = t2[0] - t1[-1] 
    if gap > TIME_WIN:
        return 1e6
            
       
    if len(t1) >= len(t2):
        anchor = 1
        fitx, fity = track1["fitx"], track1["fity"]
        meast = t2
        measx = x2
        measy = y2
        pt = t1[-1]
        # if gap < -2: # if overlap in tiem for more than 2 sec, get all the overlaped range
        #     n = 0
        #     while meast[n] <= pt:
        #         n+= 1
        # else:
        n = min(len(meast), 30) # consider n measurements
        meast = meast[:n]
        measx = measx[:n]
        measy = measy[:n]
        dir = 1
        
        
    else:
        anchor = 2
        fitx, fity = track2["fitx"], track2["fity"]
        meast = t1
        measx = x1
        measy = y1
        pt = t2[0]
        # if gap < -2 or t1[0] > t2[0]: # if overlap in time is more than 2 sec, or t1 completely overlaps with t2, get all the overlaped range
        #     i = 0
        #     while meast[i] < pt:
        #         i += 1
        #     n = len(meast)-i
        # else:
        n = min(len(meast), 30) # consider n measurements
        meast = meast[-n:]
        measx = measx[-n:]
        measy = measy[-n:]
        dir = -1
        
    
    # find where to start the cone
    # 
    if anchor==2 and t1[0] > t2[0]: # t1 is completely overlap with t2
        pt = t1[-1]
        tdiff = meast * 0 # all zeros
    else:
        tdiff = (meast - pt) * dir


    # tdiff = meast - pt
    tdiff[tdiff<0] = 0 # cap non-negative

    
    
    slope, intercept = fitx
    targetx = slope * meast + intercept
    slope, intercept = fity
    targety = slope * meast + intercept
    
    sigmax = (0.05 + tdiff * 0.01) * fitx[0] #0.1,0.1, sigma in unit ft
    varx = sigmax**2
    # vary_pred = np.var(y1) if anchor == 1 else np.var(y2)
    sigmay = 1.5 + tdiff* 2 * fity[0]
    vary_pred = sigmay**2
    # vary_pred = max(vary_pred, 2) # lower bound
    vary_meas = np.var(measy)
    vary_meas = max(vary_meas, 2) # lower bound 
    
    
    bd = []
    for i, t in enumerate(tdiff):
        mu1 = np.array([targetx[i], targety[i]]) # predicted state
        mu2 = np.array([measx[i], measy[i]]) # measured state
        cov1 = np.diag([varx[i], vary_pred[i]]) # prediction variance - grows as tdiff
        cov2 = np.diag([varx[0], vary_meas])  # measurement variance - does not grow as tdiff
        # mu1 = np.array([targetx[i]]) # predicted state
        # mu2 = np.array([measx[i]]) # measured state
        # cov1 = np.diag([varx[i]]) # prediction variance - grows as tdiff
        # cov2 = np.diag([varx[0]])  # measurement variance - does not grow as tdiff
        bd.append(bhattacharyya_distance(mu1, mu2, cov1, cov2))
    
    nll = np.mean(bd)
    
    # print("id1: {}, id2: {}, cost:{:.2f}".format(str(track1['_id'])[-4:], str(track2['_id'])[-4:], nll))
    # print("")
    
    return nll + cost_offset

    
    








