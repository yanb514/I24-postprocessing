'''
data_association module connected with database
3/25: first pass of spatial_temporal_match_online ready
- parallel processes?
- how to set up queues?
'''
import numpy as np
import torch

import heapq
from utils.data_structures import Fragment
from time import sleep


loss = torch.nn.GaussianNLLLoss() 
# TODO: confirm these, put them in parameters
x_bound_max = 31680
x_bound_min = 0
   
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
    if track2.t[0] < track1.t[-1]: # if track2 starts before track1 ends
        return INF
    if track2.t[0] - track1.t[-1] > TIME_WIN: # if track2 starts TIME_WIN after track1 ends
        return -INF
    
    # predict from track1 forward to time of track2
    xx = np.vstack([track2.t,np.ones(len(track2.t))]).T # N x 2
    targetx = np.matmul(xx, track1.fitx)
    targety = np.matmul(xx, track1.fity)
    pt1 = track1.t[-1]
    varx = (np.array(track2.t)-pt1) * VARX 
    vary = (np.array(track2.t)-pt1) * VARY

    input = torch.transpose(torch.tensor([track2.x,track2.y]),0,1)
    target = torch.transpose(torch.tensor([targetx, targety]),0,1)
    var = torch.transpose(torch.tensor([varx,vary]),0,1)
    nll1 = loss(input,target,var).item()
    
    # predict from track2 backward to time of track1 
    xx = np.vstack([track1.t,np.ones(len(track1.t))]).T # N x 2
    targetx = np.matmul(xx, track2.fitx)
    targety = np.matmul(xx, track2.fity)
    pt1 = track2.t[-1]
    varx = (np.array(track1.t)-pt1) * VARX 
    vary = (np.array(track1.t)-pt1) * VARY
    input = torch.transpose(torch.tensor([track1.x,track1.y]),0,1)
    target = torch.transpose(torch.tensor([targetx, targety]),0,1)
    var = torch.transpose(torch.tensor([varx,vary]),0,1)
    nll2 = loss(input,target,np.abs(var)).item()
    cost = min(nll1, nll2)
    # cost = (nll1 + nll2)/2
    # print(cost)
    # if track1.ID == 100165 or track2.ID == 100165:
    #     print("track1 {} track 2 {}, cost={:.2f}".format(track1.ID, track2.ID, cost))
    return cost
    # return nll1
    

def spatial_temporal_match_online(dr, data_q, log_q, TIME_WIN = 500, THRESH=3, VARX=0.03, VARY=0.03):
    '''
    online implementation of greedy matching
    build on ver2 - with object-oriented data structure
    data_q: queue to store temporary matched
    '''
    # TODO: 
    # 1. will the dr reflect the updates if db is being written constantly?-yes
    # 2. global access to data_q and log_q? assume so

    # Initialize
    curr_tracks = deque() # tracks in view. list of tracks. should be sorted by end_time
    past_tracks = OrderedDict() # set of ids indicate end of track ready to be matched
    path = {} # latest_track_id: previous_id. to store matching assignment
    start_times_heap = [] # a heap to order start times

    # keep grabbing fragments from queue TODO: add wait time
    while not dr._is_empty():
        track = dr._get_first('last_timestamp') # get the earliest ended track
        curr_id = track._id # last_track = track['id']
        track = Fragment(curr_id, track['timestamp'], track['x_position'], track['y_position'])
        
        path[curr_id] = curr_id
        right = track.t[-1] # right pointer: current end time
        left = right-1
        while left < right: # get all tracks that started but not ended at "right"
            start_track = dr._get_first('first_timestamp')
            start_time = start_track['first_timestamp']
            if start_track['last_timestamp'] >= right:
                heapq.heappush(start_times_heap,  start_time) # TODO: check for processed documents on database side, avoid repeatedly reading
        # start_times_heap[0] is the left pointer of the moving window
        try: 
            left = max(0, start_times_heap[0] - TIME_WIN)
        except: left = 0

        # compute track statistics
        track._computeStats()

        # print("window size :", right-left)
        # remove out of sight tracks 
        while curr_tracks and curr_tracks[0].t[-1] < left: 
            past_track = curr_tracks.popleft()
            past_tracks[past_track.id] = past_track

        # print("Curr_tracks ", [i.id for i in curr_tracks])
        # print("past_tracks ", past_tracks.keys())
        # compute score from every track in curr to track, update Cost
        for curr_track in curr_tracks:
            cost = _getCost(curr_track, track, TIME_WIN, VARX, VARY)
            if cost > THRESH:
                curr_track._addConflict(track)
            elif cost > 0:
                curr_track._addSuc(cost, track)
                track._addPre(cost, curr_track)
                        
        prev_size = 0
        curr_size = len(past_tracks)
        while curr_size > 0 and curr_size != prev_size:
            prev_size = len(past_tracks)
            remove_keys = set()
            for _, ready in past_tracks.items(): # all tracks in past_tracks are ready to be matched to tail
                best_head = ready._getFirstSuc()
                if not best_head or not best_head.pre: # if ready has no match or best head already matched to other tracks# go to the next ready
                    # past_tracks.pop(ready.id)
                    remove_keys.add(ready.id)
                
                else:
                    try: best_tail = best_head._getFirstPre()
                    except: best_tail = None
                    if best_head and best_tail and best_tail.id == ready.id and best_tail.id not in ready.conflicts_with:
                        # print("** match tail of {} to head of {}".format(best_tail.id, best_head.id))
                        path[best_head.id] = path[best_tail.id]
                        remove_keys.add(ready.id)
                        Fragment._matchTailHead(best_tail, best_head)
                        matched += 1

            [past_tracks.pop(key) for key in remove_keys]
            curr_size = len(past_tracks)

        # check if current track reaches the boundary, if yes, write its path to database 
        if (track.dir == 1 and track.x[-1] > x_bound_max) or (track.dir == -1 and track.x[-1] < x_bound_min):
            # TODO: put to queue
            key = track.id
            fragment_ids = [key]
            while key != path[key]:
                fragment_ids.append(path[key])
                key = path[key]
            data_q.put(fragment_ids)
        else:
            curr_tracks.append(track)        
        # running_tracks.pop(track.id) # remove tracks that ended
    
    return 