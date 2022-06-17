import pandas as pd
import numpy as np
from cvxopt import matrix, solvers, sparse,spdiag,spmatrix
from bson.objectid import ObjectId
from i24_logger.log_writer import logger, catch_critical, log_warnings, log_errors

# TODO
# add try except and put errors/warnings to log
logger.set_name("reconciliation_module")
solvers.options['show_progress'] = False
dt = 1/30

# ==================== CVX optimization for 2d dynamics ==================
@catch_critical(errors = (Exception))
def combine_fragments(raw_collection, stitched_doc):
    '''
    stack fragments from stitched_doc to a single document
    fragment_ids should be sorted by last_timestamp (see PathCache data structures for details)
    :param raw_collection: a mongoDB collection object to query fragments from fragment_ids
    :param stitched_doc: an output document from stitcher
    fields that need to be extended: timestamp, x_position, y_positin, road_segment_id, flags
    fields that need to be removed: _id, 
    fields that are preserved (copied from any fragment): vehicle class
    fields that need to be re-assigned: first_timestamp, last_timestamp, starting_x, ending_x, length, width, height
    '''
    
    stacked = {}
    stacked["timestamp"] = []
    stacked["x_position"] = []
    stacked["y_position"] = []
    stacked["road_segment_id"] = []
    stacked["flags"] = []
    stacked["length"] = []
    stacked["width"] = []
    stacked["height"] = []
    
    # print("here")
    # print(stitched_doc["fragment_ids"])    # t0 = time.time()
    if isinstance(stitched_doc, list):
        fragment_ids = stitched_doc
    else:
        fragment_ids = stitched_doc["fragment_ids"]
    if isinstance(fragment_ids[0], str):
        fragment_ids = [ObjectId(_id) for _id in fragment_ids]

    all_fragment = raw_collection.find({"_id": {"$in": fragment_ids}}) # returns a cursor
    # print("in takes", time.time()-t0)

    # tt = time.time()
    for fragment in all_fragment:
        # print(fragment["first_timestamp"], fragment["last_timestamp"])
        # print(len(fragment["timestamp"]))
        stacked["timestamp"].extend(fragment["timestamp"])
        stacked["x_position"].extend(fragment["x_position"])
        stacked["y_position"].extend(fragment["y_position"])
        stacked["road_segment_id"].extend(fragment["road_segment_id"])
        stacked["flags"].extend(fragment["flags"])
        stacked["length"].extend(fragment["length"])
        stacked["width"].extend(fragment["width"])
        stacked["height"].extend(fragment["height"])
    
    # print("for loop: ", time.time()-tt)    
    # first fragment
    first_id = fragment_ids[0]
    first_fragment = raw_collection.find_one({"_id": first_id})
    stacked["starting_x"] = first_fragment["starting_x"]
    stacked["first_timestamp"] = first_fragment["first_timestamp"]
    
    # last fragment
    last_id = fragment_ids[-1]
    last_fragment = raw_collection.find_one({"_id": last_id})
    stacked["ending_x"] = last_fragment["ending_x"]
    stacked["last_timestamp"] = last_fragment["last_timestamp"]
    
    # take the median of dimensions
    stacked["length"] = np.median(stacked["length"])
    stacked["width"] = np.median(stacked["width"])
    stacked["height"] = np.median(stacked["height"])
    
    return stacked



@catch_critical(errors = (Exception))    
def resample(car):
    # resample timestamps to 30hz, leave nans for missing data
    '''
    resample the original time-series to uniformly sampled time series in 30Hz
    car: document
    '''

    # Select time series only
    try:
        time_series_field = ["timestamp", "x_position", "y_position"]
        data = {key: car[key] for key in time_series_field}
        
        # Read to dataframe and resample
        df = pd.DataFrame(data, columns=data.keys()) 
        index = pd.to_timedelta(df["timestamp"], unit='s')
        df = df.set_index(index)
        df = df.drop(columns = "timestamp")
        df = df.resample('0.033333333S').mean() # close to 30Hz
        df.index = df.index.values.astype('datetime64[ns]').astype('int64')*1e-9
    
        car['x_position'] = df['x_position'].values
        car['y_position'] = df['y_position'].values
        car['timestamp'] = df.index.values
    except Exception as e:
        logger.error(e)
    return car

  
@catch_critical(errors = (Exception))     
def _blocdiag(X, n):
    """
    makes diagonal blocs of X, for indices in [sub1,sub2]
    n indicates the total number of blocks (horizontally)
    """
    if not isinstance(X, spmatrix):
        X = sparse(X)
    a,b = X.size
    if n==b:
        return X
    else:
        mat = []
        for i in range(n-b+1):
            row = spmatrix([],[],[],(1,n))
            row[i:i+b]=matrix(X,(b,1))
            mat.append(row)
        return sparse(mat)


@catch_critical(errors = (Exception))
def _getQPMatrices(x, t, lam2, lam1, reg="l2"):
    '''
    turn ridge regression (reg=l2) 
    1/M||y-Hx||_2^2 + \lam2/N ||Dx||_2^2
    and elastic net regression (reg=l1)
    1/M||y-Hx-e||_2^2 + \lam2/N ||Dx||_2^2 + \lam1/M||e||_1
    to QP form
    min 1/2 z^T Q x + p^T z + r
    s.t. Gz <= h
    input:  x: data array with missing data
            t: array of timestamps (no missing)
    return: Q, p, H, (G, h if l1)
    TODO: uneven timestamps
    '''
    if reg == "l1" and lam1 is None:
        raise ValueError("lam1 must be specified when regularization is set to L1")
        
    # get data
    N = len(x)
    
    # non-missing entries
    idx = [i.item() for i in np.argwhere(~np.isnan(x)).flatten()]
    x = x[idx]
    M = len(x)
    
    # differentiation operator
    # D1 = _blocdiag(matrix([-1,1],(1,2), tc="d"), N) * (1/dt)
    # D2 = _blocdiag(matrix([1,-2,1],(1,3), tc="d"), N) * (1/dt**2)
    D3 = _blocdiag(matrix([-1,3,-3,1],(1,4), tc="d"), N) * (1/dt**3)
    
    if reg == "l2":
        DD = lam2 * D3.trans() * D3
        # sol: xhat = (I+delta D'D)^(-1)x
        I = spmatrix(1.0, range(N), range(N))
        H = I[idx,:]
        DD = lam2*D3.trans() * D3
        HH = H.trans() * H
        try:
            Q = 2*(HH/M+DD/N)
            p = -2*H.trans() * matrix(x)/M
        except ZeroDivisionError:
            logger.error("Zero division: M = {}, N = {}".format(M, N) )
        return Q, p, H, N, M
    else:
        DD = lam2 * D3.trans() * D3
        # define matices
        I = spmatrix(1.0, range(N), range(N))
        IM = spmatrix(1.0, range(M), range(M))
        O = spmatrix([], [], [], (N,N))
        OM = spmatrix([], [], [], (M,M))
        H = I[idx,:]
        HH = H.trans()*H

        Q = 2*sparse([[HH/M+DD/N,H/M,-H/M], # first column of Q
                    [H.trans()/M,IM/M, -H*H.trans()/M], 
                    [-H.trans()/M,-H*H.trans()/M,IM/M]]) 
        
        p = 1/M * sparse([-2*H.trans()*matrix(x), -2*matrix(x)+lam1, 2*matrix(x)+lam1])
        G = sparse([[H*O,H*O],[-IM,OM],[OM,-IM]])
        h = spmatrix([], [], [], (2*M,1))
        return Q, p, H, G, h, N,M

def _getQPMatrices_nan(x, t, lam2, lam1, reg="l2"):
    '''
    same with _getQPMatrices, but dealt with x vector is all nans
    Q = 2(lam2 D^T D)
    p = 0
    
    '''
    if reg == "l1" and lam1 is None:
        raise ValueError("lam1 must be specified when regularization is set to L1")
        
    # get data
    N = len(x)
    
    # non-missing entries
    idx = [i.item() for i in np.argwhere(~np.isnan(x)).flatten()]
    x = x[idx]
    M = len(x)
    assert M == 0
    
    # differentiation operator
    # D1 = _blocdiag(matrix([-1,1],(1,2), tc="d"), N) * (1/dt)
    # D2 = _blocdiag(matrix([1,-2,1],(1,3), tc="d"), N) * (1/dt**2)
    D3 = _blocdiag(matrix([-1,3,-3,1],(1,4), tc="d"), N) * (1/dt**3)
    DD = lam2 * D3.trans() * D3
    Q = 2*(DD/N)
    p = spmatrix([], [], [], (N,1))
    p = matrix(p, tc="d")
    return Q, p
    
    
def rectify_1d(car, lam2, axis):
    '''                        
    solve solve for ||y-x||_2^2 + \lam ||Dx||_2^2
    args: lam2
    axis: "x" or "y"
    '''  
    # get data and matrices
    x = car[axis + "_position"].values
    Q, p, H, N,M = _getQPMatrices(x, 0, lam2, None, reg="l2")
    
    sol=solvers.qp(P=Q, q=p)
    print(sol["status"])
    
    # extract result
    xhat = sol["x"][:N]
    return xhat

def rectify_1d_l1(car, lam2, lam1, axis):
    '''                        
    solve for ||y-Hx-e||_2^2 + \lam2 ||Dx||_2^2 + \lam1||e||_1
    convert to quadratic programming with linear inequality constraints
    handle sparse outliers in data
    rewrite l1 penalty to linear constraints https://math.stackexchange.com/questions/391796/how-to-expand-equation-inside-the-l2-norm
    :param args: (lam2, lam1)
    '''  
    x = car[axis + "_position"].values
    Q, p, H, G, h, N,M = _getQPMatrices(x, 0, lam2, lam1, reg="l1")
    sol=solvers.qp(P=Q, q=matrix(p) , G=G, h=matrix(h))
    
    # extract result
    xhat = sol["x"][:N]
    # u = sol["x"][N:N+M]
    # v = sol["x"][N+M:]
    print(sol["status"])
    
    return xhat

def rectify_2d(car, reg = "l2", **kwargs):
    '''
    rectify on x and y component independently
    batch method
    :param args: (lam2_x, lam2_y) if reg == "l2"
                (lam2_x, lam2_y, lam1_x, lam1_y) if reg == "l1"
    '''
    if reg == "l1" and "lam1_x" not in kwargs:
        raise ValueError("lam1 must be specified if regularization is set to l1.")
      
    lam2_x, lam2_y = kwargs["lam2_x"], kwargs["lam2_y"] # shared arguments
    
    if reg == "l2":
        xhat = rectify_1d(car, lam2_x, "x")
        yhat = rectify_1d(car, lam2_y, "y")
        
    elif reg == "l1": 
        lam1_x, lam1_y = kwargs["lam1_x"], kwargs["lam1_y"] # additional arguments for l1
        xhat = rectify_1d_l1(car, (lam2_x, lam1_x), "x")
        yhat = rectify_1d_l1(car, (lam2_y, lam1_y), "y")
        
    # write to document
    car['x_position'] = xhat 
    car['y_position'] = yhat   
    return car           





# =================== RECEDING HORIZON RECTIFICATION =========================

def receding_horizon_1d(car, lam2, PH, IH, axis="x"):
    '''
    rolling horizon version of rectify_1d
    car: dict
    args: (lam2, PH, IH)
        PH: prediction horizon
        IH: implementation horizon
    QP formulation with sparse matrix min ||y-x||_2^2 + \lam ||Dx||_2^2
    '''
    # TODO: compute matrices once

    # get data
    x = car[axis+"_position"]
    n_total = len(x)
    # Q, p, H, N,M = _getQPMatrices(x[:PH], 0, lam, reg="l2")
    # sol=solvers.qp(P=Q, q=p)
    
    # additional equality constraint: state continuity
    A = sparse([[spmatrix(1.0, range(4), range(4))], [spmatrix([], [], [], (4,PH-4))]])
    A = matrix(A, tc="d")
    
    # save final answers
    xfinal = matrix([])
    
    n_win = max(0,(n_total-PH+IH)//IH)
    last = False
    
    cs = 3
    for i in range(n_win+1):
        # print(i,'/',n_total, flush=True)
        if i == n_win: # last
            xx =x[i*IH:]
            last = True
        else:
            xx = x[i*IH: i*IH+PH]
        nn = len(xx)
        try:
            Q, p, H, N,M = _getQPMatrices(xx, 0, lam2, None, reg="l2")
        except ZeroDivisionError:
            # print("This particular moving window has no valid data, try longer PH")
            Q, p = _getQPMatrices_nan(xx, 0, lam2, None, reg="l2")
    
        try:
            A = sparse([[spmatrix(1.0, range(cs), range(cs))], [spmatrix([], [], [], (cs,nn-cs))]])
            A = matrix(A, tc="d")
            b = matrix(x_prev)
            sol=solvers.qp(P=Q, q=p, A = A, b=b) 
            
        except: # if x_prev exists - not first window
            sol=solvers.qp(P=Q, q=p)  
            
        xhat = sol["x"][:N]

        if last:
            xfinal = matrix([xfinal, xhat])         
        else:
            xfinal = matrix([xfinal, xhat[:IH]])
            
        # save for the next loop
        x_prev = xhat[IH:IH+cs]
    
    return xfinal


def receding_horizon_2d(car, lam2_x, lam2_y, PH, IH):
    '''
    car: stitched fragments from data_q
    TODO: parallelize x and y?
    '''
    # get data    
    xhat = receding_horizon_1d(car, lam2_x, PH, IH, "x")
    yhat = receding_horizon_1d(car, lam2_y, PH, IH, "y")
    
    car['x_position'] = xhat
    car['y_position'] = yhat
    
    # TODO: replace with schema validation in dbw before insert
    car["timestamp"] = list(car["timestamp"])
    car["x_position"] = list(car["x_position"])
    car["y_position"] = list(car["y_position"])
    
    

    return car


def receding_horizon_1d_l1(car, lam2, lam1, PH, IH, axis):
    '''
    rolling horizon version of rectify_1d_l1
    car: dict
    args: (lam1, lam2, PH, IH)
        PH: prediction horizon
        IH: implementation horizon
    '''
    # TODO: compute matrices once

    # get data
    x = car[axis+"_position"]
    n_total = len(x)
    
    # additional equality constraint: state continuity
    A = sparse([[spmatrix(1.0, range(4), range(4))], [spmatrix([], [], [], (4,PH-4))]])
    A = matrix(A, tc="d")
    
    # save final answers
    xfinal = matrix([])
    
    n_win = max(0,(n_total-PH+IH)//IH)
    last = False
    
    cs = 3
    for i in range(n_win+1):
        # print(i,'/',n_total, flush=True)
        if i == n_win: # last
            xx =x[i*IH:]
            last = True
        else:
            xx = x[i*IH: i*IH+PH]
        # nn = len(xx)
        Q, p, H, G, h, N,M = _getQPMatrices(xx, 0, lam2, lam1, reg="l1")
        
        
        try: # if x_prev exists - not first window
            A = sparse([[spmatrix(1.0, range(cs), range(cs))], [spmatrix([], [], [], (cs,N-cs + 2*M))]])
            A = matrix(A, tc="d")
            b = matrix(x_prev)
            sol=solvers.qp(P=Q, q=p, G=G, h=matrix(h), A = A, b=b)   
            
        except:
            sol=solvers.qp(P=Q, q=matrix(p) , G=G, h=matrix(h))    
        
        xhat = sol["x"][:N]

        if last:
            xfinal = matrix([xfinal, xhat])         
        else:
            xfinal = matrix([xfinal, xhat[:IH]])
            
        # save for the next loop
        x_prev = xhat[IH:IH+cs]
    
    return xfinal


def receding_horizon_2d_l1(car, lam2_x, lam2_y, lam1_x, lam1_y, PH, IH):
    '''
    car: stitched fragments from data_q
    TODO: parallelize x and y?
    '''
    xhat = receding_horizon_1d_l1(car, lam2_x, lam1_x, PH, IH, "x")
    yhat = receding_horizon_1d_l1(car, lam2_y, lam1_y, PH, IH, "y")
    
    car['x_position'] = xhat
    car['y_position'] = yhat

    # TODO: replace with schema validation in dbw before insert
    car["timestamp"] = list(car["timestamp"])
    car["x_position"] = list(car["x_position"])
    car["y_position"] = list(car["y_position"])
    return car




    