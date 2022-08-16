
# loss = torch.nn.GaussianNLLLoss() 

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



def add_filter(track, thresh):
    '''
    remove bad detection from track before linear regression fit
    outlier based on shapes - remove those that are thresh away from the median of length and width
    track: a document
    '''
    # use detection confidence thresholding
    # print(thresh)
    # print(track.detection_confidence)
    filter = np.where(np.array(track.detection_confidence) > thresh)[0]
    # print(len(track.t), len(filter))
    
    # use shape thresholding
    # l,w = track.length, track.width
    # l_low, l_high = np.quantile(l, thresh, method='median_unbiased'), np.quantile(l, 1-thresh, method='median_unbiased')
    # w_low, w_high = np.quantile(w, thresh, method='median_unbiased'), np.quantile(w, 1-thresh, method='median_unbiased')

    # filter = [i for i in range(len(l)) if l_low<l[i]<l_high and w_low<w[i]<w_high] # keep index
    
    if len(filter) < 3:
        filter = np.arange(len(track.t))    
    track.filter = filter
    return track
    
    
    
def line_regress(track, CONF):
    '''
    compute statistics for matching cost
    based on linear vehicle motion (least squares fit constant velocity)
    '''

    if not hasattr(track, "filter"):
        track = add_filter(track, CONF)   
    filter = track.filter   
    t = [track.t[i] for i in filter]
    x = [track.x[i] for i in filter]
    y = [track.y[i] for i in filter]

        # t,x,y = track.t, track.x, track.y

    slope, intercept, r, p, std_err = stats.linregress(t,x)
    fitx = [slope, intercept, r, p, std_err]
    slope, intercept, r, p, std_err = stats.linregress(t,y)
    fity = [slope, intercept, r, p, std_err]
    # track.fitx = fitx
    # track.fity = fity
    return fitx, fity



@catch_critical(errors = (Exception))
def cost_1(track1, track2, TIME_WIN, VARX, VARY, CONF, with_filter = True):
    '''
    1. add filters before linear regression fit
    2. cone offset to consider time-overlapped fragments and allow initial uncertainties
    '''
    
    cone_offset = 1
    cost_offset = -2
    n = 30 # consider n measurements
    
    # if time_gap > tIME_WIN, don't stitch
    if track2.t[0] - track1.t[-1] > TIME_WIN:
        return 1e6
    
    # add filters to track
    if with_filter:
        if not hasattr(track1, "filter"):
            track1 = add_filter(track1, CONF)
        if not hasattr(track2, "filter"):
            track2 = add_filter(track2, CONF)
            
    if len(track1.t) >= len(track2.t):
        anchor = 1
        fitx, fity = line_regress(track1, CONF)
        meast = track2.t[track2.filter] if with_filter else track2.t
        measx = track2.x[track2.filter] if with_filter else track2.x
        measy = track2.y[track2.filter] if with_filter else track2.y
        meast = meast[:n]
        measx = measx[:n]
        measy = measy[:n]
        
    else:
        anchor = 2
        fitx, fity = line_regress(track2, CONF)
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
    # print("x_rvalue: {},  x_std_err: {}".format(fitx[2], fitx[4]))
    
    # cost = nllx + nlly
    return cost + cost_offset

        
        
@catch_critical(errors = (Exception))
def nll_modified(track1, track2, TIME_WIN, VARX, VARY):
    '''
    negative log likelihood of track2 being a successor of track1
    except that the cost is moved backward, starting at the beginning of track1 to allow overlaps
    '''
    track1 = line_regress(track1, None)
    track2 = line_regress(track2, None)
    
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



@catch_critical(errors = (Exception))
def cost_2(track1, track2, TIME_WIN, VARX, VARY):
    '''
    track already has fit and mask
    track.filter: all good ones are True
    '''
       
    cone_offset = 0
    cost_offset = -3

    # n=30
    
    # if time_gap > tIME_WIN, don't stitch
    if track2.t[0] - track1.t[-1] > TIME_WIN:
        return 1e6
            
    if len(track1.t) >= len(track2.t):
        anchor = 1
        fitx, fity = track1.fitx, track1.fity
        filter = np.array(track2.filter, dtype=bool) # convert fomr [1,0] to [True, False]
        meast = track2.t[filter]
        measx = track2.x[filter]
        measy = track2.y[filter]
        n = min(len(meast), 30) # consider n measurements
        meast = meast[:n]
        measx = measx[:n]
        measy = measy[:n]
        
    else:
        anchor = 2
        fitx, fity = track2.fitx, track2.fity
        filter = np.array(track1.filter, dtype=bool)
        meast = track1.t[filter]
        measx = track1.x[filter]
        measy = track1.y[filter]
        n = min(len(meast), 30) # consider n measurements
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
    
    slope, intercept = fitx
    targetx = slope * meast + intercept
    slope, intercept = fity
    targety = slope * meast + intercept
    
    # const = -n/2*np.log(2*np.pi)
    # var_term_x = 1/(2*n)*np.sum(np.log(VARX + 10*tdiff))
    # dev_term_x = 1/(2*n)*np.sum((measx-targetx)**2/(VARX + 10*tdiff))
    # # var_term_y = 1/(2*n)*np.sum(np.log(VARY*tdiff))
    # # dev_term_y = 1/(2*n)*np.sum((measy-targety)**2/(VARY * tdiff))
    
    var_term_y = 0.5*np.log(VARY) # evaluate y on a single point
    dev_term_y = 0.5*(np.nanmedian(measy)-np.nanmedian(targety))**2/VARY

    # nllx = var_term_x + dev_term_x 
    nlly = var_term_y + dev_term_y #+ -1/2*np.log(2*np.pi)
    
    # multivariate gaussian log likelihood
    # sigma = np.array([VARX, VARY])
    # sigma_d = np.array([5, 1])
    # var_term = 0
    # dev_term = 0
    # for i,td in enumerate(tdiff):
    #     sigma_i = np.diag(sigma + td*sigma_d) # y variance does not grow wrt td
    #     var_term += np.log(np.linalg.det(sigma_i))
    #     mean = np.array([targetx[i], targety[i]])
    #     xx = np.array([measx[i], measy[i]])
    #     md = np.dot(np.dot((xx-mean).T, np.linalg.inv(sigma_i)), xx-mean) # mahalanobis distance
    #     dev_term += md
        
    # nll = (var_term + dev_term)/(2*n)
    
    
    
    # scale based on fit slope
    # first find two points on the line of fitx
    # p1 = np.array([meast[-1], fitx[0]*meast[-1]+fitx[1]])
    # p2 = np.array([meast[0], fitx[0]*meast[0]+fitx[1]])
    # dist = []
    # for i,td in enumerate(tdiff):
    #     p3 = np.array([meast[i], measx[i]])
    #     dist.append(np.abs(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)) # distance from p3 to line (p1,p2)

    # dist = np.array(dist)
    # var_term_x = 1/(2*n)*np.sum(np.log(VARX + tdiff))
    # dev_term_x = 1/(2*n)*np.sum(dist**2/(VARX + tdiff))
    # nllx = var_term_x + dev_term_x 
    
    
    
    # time headway perspective
    # prediction error stdev sigma = (tau0 + tdiff*dtau)*v = s0 + tdiff*ds -> grows linearly wrt tdiff
    # assume extremely close headway s = s0 + tau*v, where s0 = 5, tau = 0.2
    # sigma(tdiff) = s0 + (tau0 + tdiff * dtau) * v
    sigma_arr = 0.1 + (0.05 + tdiff * 0.1) * fitx[0] #0.1,0.1,0.1, sigma in unit ft
    var_arr = sigma_arr**2
    var_term_x = 1/(2*n)*np.sum(np.log(var_arr))
    dev_term_x = 1/(2*n)*np.sum((measx-targetx)**2/var_arr)
    nllx = var_term_x + dev_term_x 
    
    # add time-gap cost
    # dt_cost = 0.5*(np.exp(1*abs(gap))-np.exp(1)) # (gap=1sec, cost=0, np.exp(1)=e)
    dt_cost = 0.5*(abs(gap)-2)
    cost = nllx + nlly + dt_cost
    # cost = nll + dt_cost
    print("id1: {}, id2: {}, cost:{:.2f}".format(track1.id[-4:], track2.id[-4:], cost))
    print("nllx: {:.2f}, nlly: {:.2f} dt: {:.2f}".format(nllx, nlly, dt_cost))
    print("")
    # print("nll ", nll)
    # print(f"VARX: {VARX}, VARY: {VARY}, nllx:{nllx}, nlly:{nlly}")
    # print(f"[x_speed, x_intercept, x_rvalue, x_pvalue, x_std_err]:{fitx}, [y...]:{fity}, anchor={anchor}, n={n}")
    # print("x_rvalue: {},  x_std_err: {}".format(fitx[2], fitx[4]))
    
    # cost = nllx + nlly
    return cost + cost_offset


            



            
        
    def delete_collection(self, collection_list):
        '''
        delete (reset) collections in list
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dbw = DBWriter(self.config, collection_name = "none", schema_file=None)
            all_collections = dbw.db.list_collection_names()
            
            for col in collection_list:
                if col not in all_collections:
                    print(f"{col} not in collection list")
                # dbw.reset_collection() # This line throws OperationFailure, not sure how to fix it
                else:
                    dbw.db[col].drop()
                    if col not in dbw.db.list_collection_names():
                        print("Collection {} is successfully deleted".format(col))
                    
        
    def get_random(self, collection_name):
        '''
        Return a random document from a collection
        '''
        dbr = DBReader(self.config, collection_name=collection_name)
        import random
        doc = dbr.collection.find()[random.randrange(dbr.count())]
        return doc
    
        
    
    def plot_fragments(self, traj_ids):
        '''
        Plot fragments with the reconciled trajectory (if col2 is specified)

        Parameters
        ----------
        fragment_list : list of ObjectID fragment _ids
        rec_id: ObjectID (optional)
        '''
        
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        
        
        for f_id in traj_ids:
            f = self.dbr_v.find_one("_id", f_id)
            axs[0].scatter(f["timestamp"], f["x_position"], s=0.5, label=f_id)
            axs[1].scatter(f["timestamp"], f["y_position"], s=0.5, label=f_id)
            axs[2].scatter(f["x_position"], f["y_position"], s=0.5, label=f_id)

        axs[0].set_title("time v x")
        axs[1].set_title("time v y")
        axs[2].set_title("x v y")
            
        axs[0].legend()
        
    def fragment_length_dist(self):
        '''
        Get the distribution for the #fragments that matched to one trajectory
        '''
        if self.col2 is None:
            print("Collection 2 must be specified")
            return
        pipeline = [{'$project':{ 'count': { '$size':'$fragment_ids'} } }, 
                       { '$group' : {'_id':'$count', 'count':{'$sum':1} } },
                       { '$sort'  : {'count': -1 } } ]
        cur = self.col2.collection.aggregate(pipeline)
        pprint.pprint(list(cur))

    
    def evaluate_old(self):
        '''
        1. filtered fragments (not in reconciled fragment_ids)
        2. lengths of those fragments (should be <3) TODO: Some long documents are not matched
        3. 
        '''
        
        # find all unmatched fragments
        res = self.dbr_v.collection.aggregate([
               {
                  '$lookup':
                     {
                       'from': self.col2_name,
                       'localField': "_id",
                       'foreignField': "fragment_ids",
                        'pipeline': [
                            { '$project': { 'count': 1}  }, # dummy
                        ],
                       'as': "matched"
                     }
                },
                {
                 '$match': {
                     "matched": { '$eq': [] } # select all unmatched fragments
                   }
                }
            ] )
        self.res = list(res)
        print("{} out of {} fragments are not in reconciled ".format(len(self.res), self.col1.count()))
        
        # Get the length distribution of these unmatched fragments
        import math
        f_ids = [d['_id'] for d in self.res] # get all the ids
        pipeline = [{'$project':{ 'length': { '$size':'$timestamp'} } },
                    { '$match': { "_id": {'$in': f_ids } } },
                       { '$group' : {'_id':'$length', 'count':{'$sum':1}} },
                       { '$sort'  : {'count': -1 } } ]
        cur = self.dbr_v.collection.aggregate(pipeline)
        dict = {d["_id"]: math.log(d["count"]) for d in cur}
        plt.bar(dict.keys(), dict.values(), width = 2, color='g')
        plt.xlabel("Lengths of documents")
        plt.ylabel("Count (log-scale)")
        plt.title("Unmatched fragments length distribution in log")
        
        # pprint.pprint(list(cur))
        
    
        