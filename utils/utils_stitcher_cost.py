'''
data_association module connected with database
3/25: first pass of spatial_temporal_match_online ready
- parallel processes?
- how to set up queues?
'''
import numpy as np
import torch
from scipy import stats
from i24_logger.log_writer import catch_critical


loss = torch.nn.GaussianNLLLoss() 

def _compute_stats(track):
    t,x,y = track['timestamp'],track['x_position'],track['y_position']
    ct = np.nanmean(t)
    if len(t)<2:
        v = np.sign(x[-1]-x[0]) # assume 1/-1 m/frame = 30m/s
        b = x-v*ct # recalculate y-intercept
        fitx = np.array([v,b[0]])
        fity = np.array([0,y[0]])
    else:
        xx = np.vstack([t,np.ones(len(t))]).T # N x 2
        fitx = np.linalg.lstsq(xx,x, rcond=None)[0]
        fity = np.linalg.lstsq(xx,y, rcond=None)[0]
    track['fitx'] = fitx
    track['fity'] = fity
    return track

   
# define cost
def min_nll_cost(track1, track2, TIME_WIN, VARX, VARY):
    '''
    track1 always ends before track2 ends
    999: mark as conflict
    -1: invalid
    '''
    INF = 10e6
    if track2.first_timestamp < track1.last_timestamp: # if track2 starts before track1 ends
        return INF
    if track2.first_timestamp - track1.last_timestamp > TIME_WIN: # if track2 starts TIME_WIN after track1 ends
        return -INF
    
    # predict from track1 forward to time of track2
    xx = np.vstack([track2.t,np.ones(len(track2.t))]).T # N x 2
    targetx = np.matmul(xx, track1.fitx)
    targety = np.matmul(xx, track1.fity)
    pt1 = track1.t[-1]
    varx = (track2.t-pt1) * VARX 
    vary = (track2.t-pt1) * VARY

    input = torch.transpose(torch.tensor(np.array([track2.x,track2.y])),0,1) # if track2.x and track2.y are np.array, this operation is slow (according to tensor's warning)
    target = torch.transpose(torch.tensor(np.array([targetx, targety])),0,1)
    var = torch.transpose(torch.tensor(np.array([varx,vary])),0,1)
    nll1 = loss(input,target,var).item() # get a number from a tensor
    
    # predict from track2 backward to time of track1 
    # xx = np.vstack([track1.t,np.ones(len(track1.t))]).T # N x 2
    # targetx = np.matmul(xx, track2.fitx)
    # targety = np.matmul(xx, track2.fity)
    # pt1 = track2.t[0]
    # varx = (track1.t-pt1) * VARX 
    # vary = (track1.t-pt1) * VARY
    # input = torch.transpose(torch.tensor(np.array([track1.x,track1.y])),0,1)
    # target = torch.transpose(torch.tensor(np.array([targetx, targety])),0,1)
    # var = torch.transpose(torch.tensor(np.array([varx,vary])),0,1)
    # nll2 = loss(input,target,np.abs(var)).item()
    # cost = min(nll1, nll2)
    # cost = (nll1 + nll2)/2
    return nll1



def nll(track1, track2, TIME_WIN, VARX, VARY):
    '''
    negative log likelihood of track2 being a successor of track1
    '''
    INF = 10e6
    if track2.first_timestamp < track1.last_timestamp: # if track2 starts before track1 ends
        return INF
    if track2.first_timestamp - track1.last_timestamp > TIME_WIN: # if track2 starts TIME_WIN after track1 ends
        return INF
    
    n = min(len(track2.t), 30)
    pt1 = track1.t[-1]-1e-6
    tdiff = track2.t[:n] - pt1

    xx = np.vstack([track2.t[:n],np.ones(n)]).T # N x 2
    targetx = np.matmul(xx, track1.fitx)
    targety = np.matmul(xx, track1.fity)
    
    # const = n/2*np.log(2*np.pi)
    nllx =  n/2*np.log(VARX) + 1/2*np.sum(np.log(tdiff)) + 1/2*np.sum(1/(tdiff)*(track2.x[:n]-targetx)**2)
    nlly =  n/2*np.log(VARY) + 1/2*np.sum(np.log(tdiff)) + 1/2*np.sum(1/(tdiff)*(track2.y[:n]-targety)**2)
    
    cost = (nllx + nlly)/n
    return cost



def add_filter(track, thresh = 0.3):
    '''
    remove bad detection from track before linear regression fit
    outlier based on shapes - remove those that are thresh away from the median of length and width
    track: a document
    '''
    l,w = track.length, track.width
    l_low, l_high = np.quantile(l, thresh, method='median_unbiased'), np.quantile(l, 1-thresh, method='median_unbiased')
    w_low, w_high = np.quantile(w, thresh, method='median_unbiased'), np.quantile(w, 1-thresh, method='median_unbiased')

    filter = [i for i in range(len(l)) if l_low<l[i]<l_high and w_low<w[i]<w_high] # keep index
    track.filter = filter
    return track
    
    
    
def line_regress(track, with_filter = True):
    '''
    compute statistics for matching cost
    based on linear vehicle motion (least squares fit constant velocity)
    '''
    if with_filter:
        if not hasattr(track, "filter"):
            track = add_filter(track, thresh = 0.3)    
        filter = track.filter   
        t = [track.t[i] for i in filter]
        x = [track.x[i] for i in filter]
        y = [track.y[i] for i in filter]
    else:
        t,x,y = track.t, track.x, track.y
    
    try:
        slope, intercept, r, p, std_err = stats.linregress(t,x)
    except:
        print(track.id)
        print(len(t))
        print(track.t)
        print(track.x)
        print(track.length)
        print(track.width)
    fitx = [slope, intercept, r, p, std_err]
    slope, intercept, r, p, std_err = stats.linregress(t,y)
    fity = [slope, intercept, r, p, std_err]
    # track.fitx = fitx
    # track.fity = fity
    return fitx, fity



@catch_critical(errors = (Exception))
def cost_1(track1, track2, TIME_WIN, VARX, VARY, with_filter = True):
    '''
    1. add filters before linear regression fit
    2. cone offset to consider time-overlapped fragments and allow initial uncertainties
    '''

    cone_offset = 5
    cost_offset = -6
    n = 30 # consider n measurements
    
    # add filters to track
    if with_filter:
        if not hasattr(track1, "filter"):
            track1 = add_filter(track1)
        if not hasattr(track2, "filter"):
            track2 = add_filter(track2)
            
    if len(track1.t) >= len(track2.t):
        anchor = 1
        fitx, fity = line_regress(track1, with_filter = with_filter)
        meast = track2.t[track2.filter] if with_filter else track2.t
        measx = track2.x[track2.filter] if with_filter else track2.x
        measy = track2.y[track2.filter] if with_filter else track2.y
        meast = meast[:n]
        measx = measx[:n]
        measy = measy[:n]
        
    else:
        anchor = 2
        fitx, fity = line_regress(track2, with_filter = with_filter)
        meast = track1.t[track1.filter] if with_filter else track1.t
        measx = track1.x[track1.filter] if with_filter else track1.x
        measy = track1.y[track1.filter] if with_filter else track1.y
        meast = meast[-n:]
        measx = measx[-n:]
        measy = measy[-n:]
        
        
    # find where to start the cone
    gap = track2.t[0] - track1.t[-1]
    if gap > cone_offset: # large gap between tracks
        pt = track1.t[-1] if anchor == 1 else track2.t[0]
    else: # tracks are close, but not overlap, or tracks are partially overlap in time
        pt = track1.t[-1]+cone_offset if anchor == 2 else track2.t[0]-cone_offset

    tdiff = abs(meast - pt)
    tdiff = tdiff[:n] if anchor == 1 else tdiff[-n:] # cap length at ~1sec
    
    slope, intercept, r, p, std_err = fitx
    targetx = slope * meast + intercept
    slope, intercept, r, p, std_err = fity
    targety = slope * meast + intercept
    
    const = -n/2*np.log(2*np.pi)
    var_term_x = 1/2*np.sum(np.log(VARX*tdiff))
    dev_term_x = 1/2*np.sum((measx-targetx)**2/(VARX * tdiff))
    var_term_y = 1/2*np.sum(np.log(VARY*tdiff))
    dev_term_y = 1/2*np.sum((measy-targety)**2/(VARY * tdiff))
    
    nllx = var_term_x + dev_term_x + const
    nlly = var_term_y + dev_term_y + const
    
    # nllx =  n/2*np.log(VARX) + 1/2*np.sum(np.log(tdiff)) + 1/2*np.sum(1/(tdiff)*(measx-targetx)**2)
    # nlly =  n/2*np.log(VARY) + 1/2*np.sum(np.log(tdiff)) + 1/2*np.sum(1/(tdiff)*(measy-targety)**2)
    cost = (nllx + nlly)/n
    
    # print(f"id1: {track1.id}, id2: {track2.id}, cost:{cost}")
    # print(f"VARX: {VARX}, VARY: {VARY}, nllx:{nllx}, nlly:{nlly}")
    # print(f"[x_speed, x_intercept, x_rvalue, x_pvalue, x_std_err]:{fitx}, [y...]:{fity}, anchor={anchor}, n={n}")
    
    # cost = nllx + nlly
    return cost + cost_offset

        
        
        
def nll_modified(track1, track2, TIME_WIN, VARX, VARY):
    '''
    negative log likelihood of track2 being a successor of track1
    except that the cost is moved backward, starting at the beginning of track1 to allow overlaps
    '''
    track1 = line_regress(track1)
    track2 = line_regress(track2)
    
    INF = 10e6
    # if track2.first_timestamp < track1.last_timestamp: # if track2 starts before track1 ends
    #     return INF
    if track2.first_timestamp - track1.last_timestamp > TIME_WIN: # if track2 starts TIME_WIN after track1 ends
        return INF
    
    if len(track1.t) >= len(track2.t):
        # cone from beginning of track1
        n = min(len(track2.t), 2)
        pt1 = track2.t[-1]-1 # cone stems from the first_timestamp of track1
        tdiff = abs(track2.t[:n] - pt1) # has to be positive otherwise log(tdiff) will be undefined
    
        # xx = np.vstack([track2.t[:n],np.ones(n)]).T # N x 2
        # targetx = np.matmul(xx, track1.fitx)
        # targety = np.matmul(xx, track1.fity)
        slope, intercept, r, p, std_err = track1.fitx
        targetx = slope * track2.t[:n] + intercept
        slope, intercept, r, p, std_err = track1.fity
        targety = slope * track2.t[:n] + intercept
        
        # const = n/2*np.log(2*np.pi)
        nllx =  n/2*np.log(VARX) + 1/2*np.sum(np.log(tdiff)) + 1/2*np.sum(1/(tdiff)*(track2.x[:n]-targetx)**2)
        nlly =  n/2*np.log(VARY) + 1/2*np.sum(np.log(tdiff)) + 1/2*np.sum(1/(tdiff)*(track2.y[:n]-targety)**2)
        
        return (nllx + nlly)/n 
    
    else:
        # cone from end of track2
        n = min(len(track1.t), 2)
        pt1 = track1.t[-1]+1 # cone stems from the last_timestamp of track2
        tdiff = abs(track1.t[-n:] - pt1) # has to be positive otherwise log(tdiff) will be undefined
    
        xx = np.vstack([track1.t[-n:],np.ones(n)]).T # N x 2
        targetx = np.matmul(xx, track2.fitx)
        targety = np.matmul(xx, track2.fity)
        
        # const = n/2*np.log(2*np.pi)
        nllx =  n/2*np.log(VARX) + 1/2*np.sum(np.log(tdiff)) + 1/2*np.sum(1/(tdiff)*(track1.x[-n:]-targetx)**2)
        nlly =  n/2*np.log(VARY) + 1/2*np.sum(np.log(tdiff)) + 1/2*np.sum(1/(tdiff)*(track1.y[-n:]-targety)**2)
        
        return (nllx + nlly)/n 

