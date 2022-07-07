'''
data_association module connected with database
3/25: first pass of spatial_temporal_match_online ready
- parallel processes?
- how to set up queues?
'''
import numpy as np
import torch


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


def nll_modified(track1, track2, TIME_WIN, VARX, VARY):
    '''
    negative log likelihood of track2 being a successor of track1
    except that the cost is moved backward, starting at the beginning of track1 to allow overlaps
    '''
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
    
        xx = np.vstack([track2.t[:n],np.ones(n)]).T # N x 2
        targetx = np.matmul(xx, track1.fitx)
        targety = np.matmul(xx, track1.fity)
        
        # const = n/2*np.log(2*np.pi)
        nllx =  n/2*np.log(VARX) + 1/2*np.sum(np.log(tdiff)) + 1/2*np.sum(1/(tdiff)*(track2.x[:n]-targetx)**2)
        nlly =  n/2*np.log(VARY) + 1/2*np.sum(np.log(tdiff)) + 1/2*np.sum(1/(tdiff)*(track2.y[:n]-targety)**2)
        
        return (nllx + nlly)/n - 10
    
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
        
        return (nllx + nlly)/n - 10




def nll_headway(track1, track2, TIME_WIN, VARX, VARY):
    '''
    the headway distance of the first point on the shorter track to the fitted line of the longer track
    very bad
    '''
    INF = 10e6
    if track2.first_timestamp - track1.last_timestamp > TIME_WIN: # if track2 starts TIME_WIN after track1 ends
        return INF
    
    if len(track1.t) >= len(track2.t):
        long = track1
        short = track2
    else:
        long = track2
        short = track1
    
    # first point on the shorter track
    pt = [short.t[0], short.x[0], short.y[0]]
    
    # distance to the fitted line of the longer track
    dx = abs(pt[1] - (pt[0] * long.fitx[0] + long.fitx[1]))
    dy = abs(pt[2] - (pt[0] * long.fity[0] + long.fity[1]))
    
    # some scaled distance
    d = 0.1*dx**2 + 0.9*dy**2
    
    # cost = nll(track1, track2, TIME_WIN, VARX, VARY)
    # if cost < 3: 
    #     print(cost)
    # print("*", track1.id, track2.id, d, 0.1*d-3)  
    return 0.8*d-3
