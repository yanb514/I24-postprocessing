import numpy as np

import torch
from collections import defaultdict, deque, OrderedDict
import heapq
from data_structures import DoublyLinkedList, UndirectedGraph,Fragment
import time
import sys

   
loss = torch.nn.GaussianNLLLoss()   
   
def _compute_stats(track):
    t,x,y = track['t'],track['x'],track['y']
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
    track['t'] = t
    track['x'] = x
    track['y'] = y
    track['fitx'] = fitx
    track['fity'] = fity
    return track
   

            
def stitch_objects_tsmn_online_2(o, THRESHOLD_MAX=3, VARX=0.03, VARY=0.03, time_out = 500):
        '''
        Dan's online version
        
        can potentially "under" stitch if interior fragments have higher matching cost
        THRESHOLD_MAX: aobve which pairs should never be matched
        online version of stitch_objects_tsmn_ll
        track: dict with key: id, t, x, y
            {"id": 20,
             "t": [frame1, frame2,...],
             "x":[x1,x2,...], 
             "y":[y1,y2...],
             "fitx": [vx, bx], least square fit
             "fity": [vy, by]}
        tracks come incrementally as soon as they end
        '''
        # define cost
        def _getCost(track1, track2):
            '''
            track1 always ends before track2 ends
            999: mark as conflict
            -1: invalid
            '''

            if track2["t"][0] < track1['t'][-1]: # if track2 starts before track1 ends
                return 999
            if track2['t'][0] - track1['t'][-1] > time_out: # if track2 starts TIMEOUT after track1 ends
                return -1
            
            # predict from track1 forward to time of track2
            xx = np.vstack([track2['t'],np.ones(len(track2['t']))]).T # N x 2
            targetx = np.matmul(xx, track1['fitx'])
            targety = np.matmul(xx, track1['fity'])
            pt1 = track1['t'][-1]
            varx = (track2['t']-pt1) * VARX 
            vary = (track2['t']-pt1) * VARY
            input = torch.transpose(torch.tensor([track2['x'],track2['y']]),0,1)
            target = torch.transpose(torch.tensor([targetx, targety]),0,1)
            var = torch.transpose(torch.tensor([varx,vary]),0,1)
            nll1 = loss(input,target,var).item()
            
            # predict from track2 backward to time of track1 
            xx = np.vstack([track1['t'],np.ones(len(track1['t']))]).T # N x 2
            targetx = np.matmul(xx, track2['fitx'])
            targety = np.matmul(xx, track2['fity'])
            pt1 = track2['t'][-1]
            varx = (track1['t']-pt1) * VARX 
            vary = (track1['t']-pt1) * VARY
            input = torch.transpose(torch.tensor([track1['x'],track1['y']]),0,1)
            target = torch.transpose(torch.tensor([targetx, targety]),0,1)
            var = torch.transpose(torch.tensor([varx,vary]),0,1)
            nll2 = loss(input,target,np.abs(var)).item()
            return min(nll1, nll2)
            # return nll1
        
        def _first(s):
            '''Return the first element from an ordered collection
               or an arbitrary element from an unordered collection.
               Raise StopIteration if the collection is empty.
            '''
            return next(iter(s.values()))
        
        df = o.df
        # sort tracks by start/end time - not for real deployment
        
        groups = {k: v for k, v in df.groupby("ID")}
        ids = list(groups.keys())
        ordered_tracks = deque() # list of dictionaries
        all_tracks = {}
        S = []
        E = []
        for id, car in groups.items():
            t = car["Frame #"].values
            x = (car.bbr_x.values + car.bbl_x.values)/2
            y = (car.bbr_y.values + car.bbl_y.values)/2
            notnan = ~np.isnan(x)
            t,x,y = t[notnan], x[notnan],y[notnan]
            if len(t)>1: # ignore empty or only has 1 frame
                S.append([t[0], id])
                E.append([t[-1], id])
                track = {"id":id, "t": t, "x": x, "y": y} 
                # ordered_tracks.append(track)
                all_tracks[id] = track

            
        heapq.heapify(S) # min heap (frame, id)
        heapq.heapify(E)
        EE = E.copy()
        while EE:
            e, id = heapq.heappop(EE)
            ordered_tracks.append(all_tracks[id])
            
        # Initialize
        X = UndirectedGraph() # exclusion graph
        TAIL = defaultdict(list) # id: [(cost, head)]
        HEAD = defaultdict(list) # id: [(cost, tail)]
        curr_tracks = deque() # tracks in view. list of tracks. should be sorted by end_time
        path = {} # oldid: newid. to store matching assignment
        past_tracks = DoublyLinkedList() # set of ids indicate end of track ready to be matched
        TAIL_MATCHED = set()
        HEAD_MATCHED = set()
        matched = 0 # count matched pairs
        
        running_tracks = OrderedDict() # tracks that start but not end at e 
        
        start = time.time()
        for i,track in enumerate(ordered_tracks):
            # print("\n")
            # print('Adding new track {}/{}'.format(i, len(ordered_tracks)))
            # print("Out of view: {}".format(past_tracks.size))

            curr_id = track['id'] # last_track = track['id']
            path[curr_id] = curr_id
            right = track['t'][-1] # right pointer: current time
            
            # get tracks that started but not end - used to define the window left pointer
            while S and S[0][0] < right: # append all the tracks that already starts
                started_time, started_id = heapq.heappop(S)
                running_tracks[started_id] = started_time
                
            # compute track statistics
            track = _compute_stats(track)
            
            try: 
                left = max(0,_first(running_tracks) - time_out)
            except: left = 0
            # print("window size :", right-left)
            # remove out of sight tracks 
            while curr_tracks and curr_tracks[0]['t'][-1] < left:           
                past_tracks.append(curr_tracks.popleft()['id'])
            
            # compute score from every track in curr to track, update Cost
            for curr_track in curr_tracks:
                cost = _getCost(curr_track, track)
                if cost > THRESHOLD_MAX:
                    X._addEdge(curr_track['id'], track['id'])
                elif cost > 0:
                    heapq.heappush(TAIL[curr_track['id']], (cost, track['id']))
                    heapq.heappush(HEAD[track['id']], (cost, curr_track['id']))
            
            # print("TAIL {}, HEAD {}".format(len(TAIL), len(HEAD)))
            # start matching from the first ready tail
            tail_node = past_tracks.head
            if not tail_node:  # no ready tail available: keep waiting
                curr_tracks.append(track)        
                running_tracks.pop(curr_id) # remove tracks that ended
                continue # go to the next track in ordered_tracks

            while tail_node is not None:
                tail = tail_node.data # tail is ready (time-wise)
                
                # remove already matched
                while TAIL[tail] and TAIL[tail][0][1] in HEAD_MATCHED:
                    heapq.heappop(TAIL[tail]) 
                if not TAIL[tail]: # if tail does not have candidate match
                    TAIL.pop(tail)
                    tail_node = tail_node.next # go to the next ready tail
                    continue
                _, head = TAIL[tail][0] # best head for tail
                while HEAD[head] and HEAD[head][0][1] in TAIL_MATCHED:
                    heapq.heappop(HEAD[head]) 
                if not HEAD[head]:
                    HEAD.pop(head)
                    tail_node = tail_node.next
                    continue
                else: _, tail2 = HEAD[head][0]

                # tail and head agrees with each other
                if tail==tail2:
                    if head in X[tail]: # conflicts
                        HEAD.pop(head)
                        TAIL.pop(tail)
                    else: # match tail and head
                        # print("matching {} & {}".format(tail, head))
                        path[head] = path[tail]
                        X._union(head, tail)
                        HEAD.pop(head)
                        TAIL.pop(tail)
                        HEAD_MATCHED.add(head)
                        TAIL_MATCHED.add(tail)
                        matched += 1
                        past_tracks.delete_element(tail)
                        X._remove(tail)
                    
                # match or not, process the next ready tail  
                tail_node = tail_node.next
                
                    
            curr_tracks.append(track)        
            running_tracks.pop(curr_id) # remove tracks that ended
            # print("matched:", matched)
            
        # delete IDs that are empty
        # print("\n")
        # print("{} Ready: ".format(past_tracks.printList()))
        # print("{} Processsed: ".format(len(processed)))
        print("{} pairs matched".format(matched))
        # print("Deleting {} empty tracks".format(len(empty_id)))
        # df = df.groupby("ID").filter(lambda x: (x["ID"].iloc[0] not in empty_id))
        end = time.time()
        print('run time online stitching:', end-start)
        # for debugging only
        o.path = path
        # o.C = C
        o.X = X
        o.groupList = ids
        o.past_tracks = past_tracks.convert_to_set()
        o.TAIL = TAIL
        o.HEAD = HEAD
        # replace IDs
        newids = [v for _,v in path.items()]
        m = dict(zip(path.keys(), newids))
        df = df.replace({'ID': m})
        df = df.sort_values(by=['Frame #','ID']).reset_index(drop=True)         
        
        print("Before DA: {} unique IDs".format(len(ids))) 
        print("After DA: {} unique IDs".format(df.groupby("ID").ngroups))
        print("True: {} unique IDs".format(len([id for id in ids if id<1000])))
      
        o.df = df
        return o


# define cost
def _getCost(track1, track2, time_out, VARX, VARY):
    '''
    track1 always ends before track2 ends
    999: mark as conflict
    -1: invalid
    '''

    if track2.t[0] < track1.t[-1]: # if track2 starts before track1 ends
        return 999
    if track2.t[0] - track1.t[-1] > time_out: # if track2 starts TIMEOUT after track1 ends
        return -1
    
    # predict from track1 forward to time of track2
    xx = np.vstack([track2.t,np.ones(len(track2.t))]).T # N x 2
    targetx = np.matmul(xx, track1.fitx)
    targety = np.matmul(xx, track1.fity)
    pt1 = track1.t[-1]
    varx = (track2.t-pt1) * VARX 
    vary = (track2.t-pt1) * VARY
    input = torch.transpose(torch.tensor([track2.x,track2.y]),0,1)
    target = torch.transpose(torch.tensor([targetx, targety]),0,1)
    var = torch.transpose(torch.tensor([varx,vary]),0,1)
    nll1 = loss(input,target,var).item()
    
    # predict from track2 backward to time of track1 
    xx = np.vstack([track1.t,np.ones(len(track1.t))]).T # N x 2
    targetx = np.matmul(xx, track2.fitx)
    targety = np.matmul(xx, track2.fity)
    pt1 = track2.t[-1]
    varx = (track1.t-pt1) * VARX 
    vary = (track1.t-pt1) * VARY
    input = torch.transpose(torch.tensor([track1.x,track1.y]),0,1)
    target = torch.transpose(torch.tensor([targetx, targety]),0,1)
    var = torch.transpose(torch.tensor([varx,vary]),0,1)
    nll2 = loss(input,target,np.abs(var)).item()
    return min(nll1, nll2)
    # return nll1
    
def _first(s):
        '''Return the first element from an ordered collection
           or an arbitrary element from an unordered collection.
           Raise StopIteration if the collection is empty.
        '''
        return next(iter(s.values()))
        
def stitch_objects_tsmn_online_3(o, THRESHOLD_MAX=3, VARX=0.03, VARY=0.03, time_out = 500):
        '''
        build on ver2 - with object-oriented data structure
        TODO: integrate X with Fragment object
        '''
        
        df = o.df
        # sort tracks by start/end time - not for real deployment
        
        groups = {k: v for k, v in df.groupby("ID")}
        ids = list(groups.keys())
        ordered_tracks = deque() # list of dictionaries
        all_tracks = {}
        S = []
        E = []
        for id, car in groups.items():
            t = car["Frame #"].values
            x = (car.bbr_x.values + car.bbl_x.values)/2
            y = (car.bbr_y.values + car.bbl_y.values)/2
            notnan = ~np.isnan(x)
            t,x,y = t[notnan], x[notnan],y[notnan]
            if len(t)>1: # ignore empty or only has 1 frame
                S.append([t[0], id])
                E.append([t[-1], id])
                track = Fragment(id, t,x,y)
                # ordered_tracks.append(track)
                all_tracks[id] = track

        heapq.heapify(S) # min heap (frame, id)
        heapq.heapify(E)

        while E:
            e, id = heapq.heappop(E)
            ordered_tracks.append(all_tracks[id])
        del all_tracks
            

        # Initialize
        # X = UndirectedGraph() # exclusion graph
        running_tracks = OrderedDict() # tracks that start but not end at e 
        curr_tracks = deque() # tracks in view. list of tracks. should be sorted by end_time
        past_tracks = OrderedDict() # set of ids indicate end of track ready to be matched

        path = {} # oldid: newid. to store matching assignment
        matched = 0 # count matched pairs 
        
        start = time.time()
        for i,track in enumerate(ordered_tracks):
            # print("\n")
            # print('Adding new track {}/{},{}'.format(i, len(ordered_tracks),track.id))
            # print("Past tracks: {}".format(len(past_tracks)))
            # print("Curr tracks: {}".format(len(curr_tracks)))
            # print("running tracks: {}".format(len(running_tracks)))
            # print("path bytes: {}".format(sys.getsizeof(path)))
            curr_id = track.id # last_track = track['id']
            path[curr_id] = curr_id
            right = track.t[-1] # right pointer: current time
            
            # get tracks that started but not end - used to define the window left pointer
            while S and S[0][0] < right: # append all the tracks that already starts
                started_time, started_id = heapq.heappop(S)
                running_tracks[started_id] = started_time
                
            # compute track statistics
            track._computeStats()
            
            try: 
                left = max(0,_first(running_tracks) - time_out)
            except: left = 0
            # print("window size :", right-left)
            # remove out of sight tracks 
            while curr_tracks and curr_tracks[0].t[-1] < left: 
                past_track = curr_tracks.popleft()
                past_tracks[past_track.id] = past_track
            # print("Curr_tracks ", [i.id for i in curr_tracks])
            # print("past_tracks ", past_tracks.keys())
            # compute score from every track in curr to track, update Cost
            for curr_track in curr_tracks:
                cost = _getCost(curr_track, track, time_out, VARX, VARY)
                if cost > THRESHOLD_MAX:
                    curr_track._addConflict(track)
                elif cost > 0:
                    curr_track._addSuc(cost, track)
                    track._addPre(cost, curr_track)
                          
            prev_size = 0
            curr_size = len(past_tracks)
            while curr_size > 0 and curr_size != prev_size:
                prev_size = len(past_tracks)
                remove_keys = set()
                # ready = _first(past_tracks) # a fragment object
                for ready_id, ready in past_tracks.items():
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
                
                
                    
            curr_tracks.append(track)        
            running_tracks.pop(track.id) # remove tracks that ended
            # print("matched:", matched)
            
        # delete IDs that are empty
        # print("\n")
        # print("{} Ready: ".format(past_tracks.printList()))
        # print("{} Processsed: ".format(len(processed)))
        print("{} pairs matched".format(matched))
        # print("Deleting {} empty tracks".format(len(empty_id)))
        # df = df.groupby("ID").filter(lambda x: (x["ID"].iloc[0] not in empty_id))
        end = time.time()
        print('run time online stitching:', end-start)
        # for debugging only
        o.path = path
        # o.C = C
        # o.X = X
        o.groupList = ids
        o.past_tracks = past_tracks.keys()
        # replace IDs
        newids = [v for _,v in path.items()]
        m = dict(zip(path.keys(), newids))
        df = df.replace({'ID': m})
        df = df.sort_values(by=['Frame #','ID']).reset_index(drop=True)         
        
        print("Before DA: {} unique IDs".format(len(ids))) 
        print("After DA: {} unique IDs".format(df.groupby("ID").ngroups))
        print("True: {} unique IDs".format(len([id for id in ids if id<1000])))
      
        o.df = df
        return o