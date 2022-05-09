import pandas as pd
import numpy as np
import numpy.linalg as LA
from numpy import sin,cos
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
from cvxopt import matrix, solvers, sparse,spdiag,spmatrix
import time
import sys
from bson.objectid import ObjectId

# TODO
# add try except and put errors/warnings to log

solvers.options['show_progress'] = False
dt = 1/30

# ==================== CVX optimization for 2d dynamics ==================
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
    
    # stacked = stitched_doc
    # try:
    #   stacked.pop("_id")
    # except:
    #     pass
    # print(stitched_doc)
    # assert stitched_doc == 1
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
    stitched_doc = [ObjectId(_id) for _id in stitched_doc]
    all_fragment = raw_collection.find({"_id": {"$in": stitched_doc}}) # returns a cursor
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
    first_id = stitched_doc[0]
    first_fragment = raw_collection.find_one({"_id": first_id})
    stacked["starting_x"] = first_fragment["starting_x"]
    stacked["first_timestamp"] = first_fragment["first_timestamp"]
    
    # last fragment
    last_id = stitched_doc[-1]
    last_fragment = raw_collection.find_one({"_id": last_id})
    stacked["ending_x"] = last_fragment["ending_x"]
    stacked["last_timestamp"] = last_fragment["last_timestamp"]
    
    # take the median of dimensions
    stacked["length"] = np.median(stacked["length"])
    stacked["width"] = np.median(stacked["width"])
    stacked["height"] = np.median(stacked["height"])

    return stacked
    
def resample(car):
    # resample timestamps to 30hz, leave nans for missing data
    '''
    resample the original time-series to uniformly sampled time series in 30Hz
    car: document
    '''

    # Select time series only
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
    
    return car

       
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
        Q = 2*(HH/M+DD/N)
        p = -2*H.trans() * matrix(x)/M
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

    return car





# =============== need to be moved =================
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def generate_1d(initial_state, highest_order_dynamics, dt, order):
    '''
    generate vehicle states using 1-st order dynamics
    x(k+1) = x(k)+v(k)dt
    v(k+1) = v(k)+a(k)dt - if order==2
    a(k+1) = a(k)+j(k)dt - if order==3
    
    initial_state: list. [x0, v0, a0] 
    highest_order_dynamics: Nx1 array, acceleration, but only a[:-order] is used in dynamics
    dt: Nx1 array
    order: highest order of derivative. 2: acceleration. 3: jerk
    return: x,v,a
    '''
    # TODO: vectorize this function
    N = len(highest_order_dynamics)
    
    if order == 3:
        j = highest_order_dynamics
        a = np.zeros(N)
        x = np.zeros(N)
        v = np.zeros(N)
        x0,v0,a0 = initial_state
        a[0] = a0
        x[0] = x0
        v[0] = v0
        
        for k in range(0,N-1):
            a[k+1] = a[k] + j[k]*dt
            v[k+1] = v[k] + a[k]*dt
            x[k+1] = x[k] + v[k]*dt
        
    elif order == 2:
        j = np.nan
        a = highest_order_dynamics
        x = np.zeros(N)
        v = np.zeros(N)
        x0,v0 = initial_state
        x[0] = x0
        v[0] = v0
        
        for k in range(0,N-1):
            v[k+1] = v[k] + a[k]*dt
            x[k+1] = x[k] + v[k]*dt

    return x,v,a,j


def generate_2d(initial_state, highest_order_dynamics, theta, dt, order):
    '''
    generate vehicle states using 1-st order dynamics in 2D
    Simple steering dynamics:
    a(k+1) = a(k) + j(k)*dt, k=1,...,N-3
    v(k+1) = v(k) + a(k)*dt, k=1,...,N-2
    vx(k) = v(k) sin(theta(k)), k=1,...,N-1
    vy(k) = v(k) cos(theta(k)), k=1,...,N-1
    x(k+1) = x(k) + vx(k)dt, k=1,...,N-1
    y(k+1) = y(k) + vy(k)dt, k=1,...,N-1
    
    initial_state: list. [x0, y0, v0, a0] 
    highest_order_dynamics: Nx1 array
    theta: Nx1 array
    dt: float or Nx1 array
    order: highest order of derivative. 2: acceleration. 3: jerk
    return: x,y,theta,v,a,j
    '''
    # TODO: vectorize this function
    N = len(highest_order_dynamics)
    
    if order == 3:
        assert len(initial_state)==4
        j = highest_order_dynamics
        a = np.zeros(N)
        v = np.zeros(N)
        x = np.zeros(N)
        y = np.zeros(N)
        
        x0, y0, v0, a0 = initial_state
        a[0] = a0
        x[0] = x0
        y[0] = y0
        v[0] = v0
        
        for k in range(0,N-1):
            a[k+1] = a[k] + j[k]*dt
            v[k+1] = v[k] + a[k]*dt
            
        vx = v * np.cos(theta)
        vy = v * np.sin(theta)
        
        for k in range(0,N-1):
            x[k+1] = x[k] + vx[k]*dt
            y[k+1] = y[k] + vy[k]*dt
        
    elif order == 2:
        assert len(initial_state)==3
        j = np.nan
        a = highest_order_dynamics
        v = np.zeros(N)
        x = np.zeros(N)
        y = np.zeros(N)
        
        x0, y0, v0 = initial_state
        x[0] = x0
        y[0] = y0
        v[0] = v0
        
        for k in range(0,N-1):
            v[k+1] = v[k] + a[k]*dt
            
        vx = v * np.cos(theta)
        vy = v * np.sin(theta)
        
        for k in range(0,N-1):
            x[k+1] = x[k] + vx[k]*dt
            y[k+1] = y[k] + vy[k]*dt
            
    return  x,y,theta,v,a,j

def decompose_2d(car, write_to_df = False):
        '''
        the opposite of generate_2d
        given x,y,theta
        return vx,vy,ax,ay,jx,jy
        get velocity, acceleration, jerk from the GT box measurements
        '''
        try:
            x = (car["bbr_x"].values + car["bbl_x"].values)/2
            y = (car["bbr_y"].values + car["bbl_y"].values)/2
        except:
            x = car.x.values
            y = car.y.values
        vx = np.append(np.diff(x)/dt, np.nan)
        vy = np.append(np.diff(y)/dt, np.nan)
        
        ax = np.append(np.diff(vx)/dt, np.nan)
        ay = np.append(np.diff(vy)/dt, np.nan)
        
        jx = np.append(np.diff(ax)/dt, np.nan)
        jy = np.append(np.diff(ay)/dt, np.nan)
       
        if write_to_df:
            car.loc[:,"speed_x"] = vx
            car.loc[:,"speed_y"] = vy
            car.loc[:,"acceleration_x"] = ax
            car.loc[:,"acceleration_y"] = ay
            car.loc[:,"jerk_x"] = jx
            car.loc[:,"jerk_y"] = jy
            return car
        else:
            return vx,vy,ax,ay,jx,jy
        
def rectify_sequential(df, args):
    print("{} total trajectories, {} total measurements".format(df['ID'].nunique(), len(df)))
    start = time.time()
    df = df.groupby('ID').apply(rectify_single_car, args=args).reset_index(drop=True)
    end = time.time()
    print('total time rectify_sequential: ',end-start)
    return df

def generate_box(w,l, x, y, theta):
    '''
    generate 'bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y' from
    - x: Nx1 array of backcenter x
    - y: Nx1 array of backcenter y
    - theta: Nx1 array of angle relative to positive x direction (not steering)
    - w: width
    - l: length
    '''
    # compute positions
    xa = x + w/2*sin(theta)
    ya = y - w/2*cos(theta)
    xb = xa + l*cos(theta)
    yb = ya + l*sin(theta)
    xc = xb - w*sin(theta)
    yc = yb + w*cos(theta)
    xd = xa - w*sin(theta)
    yd = ya + w*cos(theta)

    Y = np.stack([xa,ya,xb,yb,xc,yc,xd,yd],axis=-1)
    return Y
def rectify_single_car(car, args):
    '''
    car: a document (dict)
    '''
    width = car['width']
    length = car['length']
    
    car = receding_horizon_2d(car, width, length, args)
    return car

def receding_horizon_1d_original(df, args, axis="x"):
    '''
    rolling horizon version of rectify_1d
    car: df
    args: (lam, axis, PH, IH)
        PH: prediction horizon
        IH: implementation horizon
    QP formulation with sparse matrix min ||y-x||_2^2 + \lam ||Dx||_2^2
    '''
    # get data
    lam, PH, IH = args
    
    x = df[axis].values
    n_total = len(x)          

    # Define constraints for the first PH
    idx = [i.item() for i in np.argwhere(~np.isnan(x[:PH])).flatten()]
    if len(idx) < 2: # not enough measurements
        print('not enough measurements in receding_horizon_1d')
        return
    xx = x[:PH]
    xx = xx[idx]

    # differentiation operator
    D1 = _blocdiag(matrix([-1,1],(1,2), tc="d"), PH) * (1/dt)
    D2 = _blocdiag(matrix([1,-2,1],(1,3), tc="d"), PH) * (1/dt**2)
    D3 = _blocdiag(matrix([-1,3,-3,1],(1,4), tc="d"), PH) * (1/dt**3)
    
    # sol: xhat = (I+delta D'D)^(-1)x
    I = spmatrix(1.0, range(PH), range(PH))
    H = I[idx,:]
    M = len(idx)

    DD = lam*D3.trans() * D3

    HH = H.trans() * H
    
    # QP formulation with sparse matrix min ||y-x||_2^2 + \lam ||Dx||_2^2
    Q = 2*(HH+DD)
    p = -2*H.trans() * matrix(xx)
    sol=solvers.qp(P=Q, q=p)
    
    # additional equality constraint: state continuity
    A = sparse([[spmatrix(1.0, range(4), range(4))], [spmatrix([], [], [], (4,PH-4))]])
    A = matrix(A, tc="d")
    
    # save final answers
    xfinal = matrix([])
    vfinal = matrix([])
    afinal = matrix([])
    jfinal = matrix([])
    
    n_win = max(0,(n_total-PH+IH)//IH)
    last = False
    
    cs = 3 # new constraint steps
    for i in range(n_win+1):
        # print(i,'/',n_total, flush=True)
        if i == n_win: # last
            xx =x[i*IH:]
            last = True
        else:
            xx = x[i*IH: i*IH+PH]
        nn = len(xx)
        idx = [i.item() for i in np.argwhere(~np.isnan(xx)).flatten()]
        xx = xx[idx]
        I = I[:nn, :nn]
        H = I[idx,:]
        D1 = D1[:nn-1 ,:nn]
        D2 = D2[:nn-2 ,:nn]
        D3 = D3[:nn-3 ,:nn]
        DD = lam*D3.trans() * D3
        HH = H.trans() * H

        Q = 2*(HH+DD)
        p = -2*H.trans() * matrix(xx)
        
        if i == 0:
            sol=solvers.qp(P=Q, q=p)          
        else: # if x_prev exists - not first window
            A = sparse([[spmatrix(1.0, range(cs), range(cs))], [spmatrix([], [], [], (cs,nn-cs))]])
            A = matrix(A, tc="d")
            b = matrix(x_prev)
            sol=solvers.qp(P=Q, q=p, A = A, b=b)          
            
        xhat = sol["x"]
        vhat = D1*xhat
        ahat = D2*xhat
        jhat = D3*xhat
        if last:
            xfinal = matrix([xfinal, xhat])
            vfinal = matrix([vfinal, vhat, np.nan])
            afinal = matrix([afinal, ahat, matrix([np.nan, np.nan])])
            jfinal = matrix([jfinal, jhat, matrix([np.nan, np.nan, np.nan])])
        else:
            xfinal = matrix([xfinal, xhat[:IH]])
            vfinal = matrix([vfinal, vhat[:IH]])
            afinal = matrix([afinal, ahat[:IH]])
            jfinal = matrix([jfinal, jhat[:IH]])

            
        # save for the next loop
        x_prev = xhat[IH:IH+cs]
    
    return xfinal, vfinal,  afinal, jfinal





# ===================== non-convex version =======================
def box_fitting(car, width, length):
    '''
    fit to measurements with the given width length
    output x,y -> best fit back center coordinates
    convert the problem into 2D point movement
    
    car: df that has raw boxes measuarement
    return car with best-fit x and y

    TODO: test with missing data
    TODO: consider direction
    '''
    # Decision variables 8N x 1
    print("in box_fitting")
    pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
    Y = np.array(car[pts])  
    N = len(Y)
    notNan = ~np.isnan(np.sum(Y,axis=-1))
    dir = car.direction.values[0]
    
    # Objective function - ||X-Xdata||_2^2
    def sos2(x,x_data):
        # x: 4 x 1, x_data: 8x1
        fbr_x, fbr_y, bbl_x, bbl_y = x
        x = np.array([bbl_x, fbr_y, fbr_x, fbr_y, fbr_x, bbl_y, bbl_x, bbl_y])
        return LA.norm(x-x_data,2)**2
    
    # only consider diagonal points
    A = np.array([[1,0,-1,0],
                  [0,1,0,-1]])
    b = np.array([length, -width]).T
    b = np.sign(dir) * b
    
    eq_cons = {'type': 'eq', 
            'fun' : lambda x: np.dot(A, x) - b}
    
    # Solve boxes are fixed-dimension rectangles, steering = 0
    x_opt = np.ones(N)*np.nan
    y_opt = np.ones(N)*np.nan
    Yre = np.ones(Y.shape) * np.nan
    
    for i, X_data in enumerate(Y):
        X = X_data [[2,3,6,7]] # fbr_x, fbr_y, bbl_x, bbl_y
        if ~np.isnan(np.sum(X)):
            res = minimize(sos2, X, (X_data), method='SLSQP',
                    constraints=[eq_cons], options={'disp': False})
            fbr_x, fbr_y, bbl_x, bbl_y = res.x
            x_opt[i] = bbl_x #(res.x[0]+res.x[6])/2
            y_opt[i] = (fbr_y + bbl_y)/2#(res.x[1]+res.x[7])/2
            Yre[i] = np.array([bbl_x, fbr_y, fbr_x, fbr_y, fbr_x, bbl_y, bbl_x, bbl_y])
       
    # newcar = car.copy()
    # newcar.loc[:,"x"] = x_opt
    # newcar.loc[:,"y"] = y_opt
    # newcar.loc[:,pts] = Yre
    car.loc[:,"x"] = x_opt
    car.loc[:,"y"] = y_opt
    car.loc[:,pts] = Yre
    return car


def obj_1d(X, x, order, N, dt, notNan, lam):
    """ The cost function for 1d
            X: decision variables X = [xhat, vhat, ahat]
                xhat: rectified positions (N x 1) 0:N
                vhat: N:2N-1
                ahat: 2N-1: 3N-3 rectified acceleration (N-2 x 1)
                jhat: 3N-3: 4N-6
            x: position measurements (N x 1), could have NaN
            N: int number of measurements
            dt: float delta T 
            notNan: bool array indicating timestamps that have measurements
            lam: given parameter
        Return: float
    """ 
    # unpack decision variables
    xhat = X[:N]
    offset1 = int((1+order-1)*(order-1)/2)

    highest_order_dynamics = X[order*N-offset1:] # to be regularized
    rescale = (30)**(order)
    # select valid measurements
    xhat = xhat[notNan]
    x = x[notNan]
    
    # min perturbation
    c1 = LA.norm(x-xhat,2)**2 * rescale /np.count_nonzero(notNan)
    c2 = LA.norm(highest_order_dynamics,2)**2 / len(highest_order_dynamics)

    cost = lam*c1 + (1-lam) * c2

    return cost

def const_1d(N, dt, order):
    """ The constraint representing linear dynamics
        N: number of timesteps for xhat, n=3N-3
        Return: matrix A (2N-3 x 3N-3), such that A dot X = [0]_(nx1)
        for scipy.optimize.LinearConstraint: lb<= A.dot(X) <= ub 'trust-constr'
    """ 
    offset = int((1+order)*order/2)
    n = (order+1)*N-offset
    m = order*N-offset
    A = np.zeros((m,n))
    
    for o in range(order):
        offset_pre = int((1+o)*o/2)
        offset_post = int((2+o)*(o+1)/2)
        start_row = o*N-offset_pre
        end_row = (o+1)*N-offset_post
        step = N-o
        for i in range(start_row, end_row):
            A[i][i+o] = -1
            A[i][i+o+1] = 1
            A[i][i+o+step] = -dt      
    
    return A
def loss(Yre, Y1, norm='l21'):
    '''
    different ways to compute the diff matrix
    '''
    notNan = ~np.isnan(np.sum(Y1,axis=-1))
    Y1 = Y1[notNan,:]
    Yre = Yre[notNan,:]
    diff = np.abs(Y1-Yre)
    N = len(diff)
    if N==0:
        return 0
    if norm=='l21':
        return np.nanmean(LA.norm(diff,axis=1))
    elif norm=='l2':
        return LA.norm(diff,'fro')/N
    elif norm=='xy': # weighted xy
        mae_x = np.abs(diff[:,[0,2,4,6]])
        mae_y = np.abs(diff[:,[1,3,5,7]]) 
        alpha = 0.3
        mae_xy = alpha*mae_x + (1-alpha)*mae_y
        return LA.norm(mae_xy,'fro')/N
    
def get_costs(Yre, Y1, x,y,v,a,theta, norm):
    '''
    for normalizing lambdas
    '''
    N = len(a)
    c1m, c2m, c3m, c4m, c5m = 1,1,1,1,1 # directly calculate 2-norm
    c1 = loss(Yre, Y1, norm)/c1m
    c2 = LA.norm(a,2)/N/30/c2m
    j = np.diff(a)
    c3 = LA.norm(j,2)/N/30/c3m
    st = sin(theta)
    c4 = LA.norm(st,2)/N/c4m
    o = np.diff(theta)
    c5 = LA.norm(o,2)/N/c5m
    return c1,c2,c3,c4,c5
    
def obj1(X, Y1,N,dt,notNan, lam1,lam2,lam3,lam4,lam5):
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
    
    Yre,x,y,a = generate(w,l,x0, y0, theta,v, outputall=True)
    Yre = Yre[notNan,:]
    # min perturbation
    c1 = lam1 * loss(Yre, Y1, 'l21')
    
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



    
def unpack1(res,N,dt):
    # extract results
    # unpack variables
                        
    x = np.zeros(N)            
    y = np.zeros(N) 
    
    # ver2 unpack
    v = res.x[:N]
    theta = res.x[N:2*N]
    x0,y0,w,l = res.x[2*N:]
    
    Yre, x, y, a = generate(w,l,x0, y0, theta,v, outputall=True)    
    return Yre, x,y,v,a,theta,w,l
                            

def rectify_single_camera(df, args):
    '''                        
    df: a single track in one camera view
    '''             
    lam1, lam2, lam3,lam4,lam5,niter = args
    
    timestamps = df.Timestamp.values
    dt = np.diff(timestamps)
    sign = df["direction"].iloc[0]
    
    # get bottom 4 points coordinates
    
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
            [(np.arccos(sign),np.arccos(sign)) for i in range(N)]+\
            [(-np.inf,np.inf),(0,np.inf),(1,4),(2,np.inf)]  
            # [(-np.pi/8+np.arccos(sign),np.pi/8+np.arccos(sign)) for i in range(N)]+\
    Y0 = generate(w0,l0,x0, y0, theta0,v0)
    diff = Y1-Y0[notNan,:]
    c1max = np.nanmean(LA.norm(diff,axis=1))
    c1max = max(c1max, 1e-4)

    # SOLVE FOR MAX C2-C5 BY SETTING LAM2-5 = 0
    lams = (100,0,0,0,0)
    minimizer_kwargs = {"method":"L-BFGS-B", "args":(Y1,N,dt,notNan,*lams),'bounds':bnds,'options':{'disp': False}}
    res = basinhopping(obj1, X0, minimizer_kwargs=minimizer_kwargs,niter=0)
    print('\n')
    print('Initilization: ',loss(Y0[notNan,:], Y1, norm='l2'))
    # extract results    
    Yre, x,y,v,a,theta,w,l = unpack1(res,N,dt)
    Yre = Yre[notNan,:]
    _,c2max,c3max,c4max,c5max = get_costs(Yre, Y1, x,y,v,a,theta,'l21')
    c2max,c3max,c4max,c5max = max(c2max, 1e-4), max(c3max, 1e-4), max(c4max, 1e-4), max(c5max, 1e-4)
    # SOLVE AGAIN - WITH NORMALIZED OBJECTIVES
    lams = (lam1/c1max,lam2/c2max,lam3/c3max,lam4/c4max,lam5/c5max)
    minimizer_kwargs = {"method":"L-BFGS-B", "args":(Y1,N,dt,notNan,*lams),'bounds':bnds,'options':{'disp': False}}
    res = basinhopping(obj1, X0, minimizer_kwargs=minimizer_kwargs,niter=niter)
    Yre, x,y,v,a,theta,w,l = unpack1(res,N,dt)
    print('Final: ',loss(Yre[notNan,:], Y1, norm='l2'))
    
    df.loc[:,pts] = Yre        
    df.loc[:,'acceleration'] = a
    df.loc[:,'speed'] = v    
    df.loc[:,'x'] = x        
    df.loc[:,'y'] = y        
    df.loc[:,'theta'] = theta
    df.loc[:,'width'] = w    
    df.loc[:,'length'] = l       
    
    return df
                            
                            
def applyParallel(dfGrouped, func, args=None):
    with Pool(cpu_count()) as p:
        if args is None:    
            ret_list = list(tqdm(p.imap(func, [group for name, group in dfGrouped]), total=len(dfGrouped.groups)))
        else:# if has extra arguments
            ret_list = list(tqdm(p.imap(partial(func, args=args), [group for name, group in dfGrouped]), total=len(dfGrouped.groups)))
    return pd.concat(ret_list)
                            
def rectify(df):            
    '''                        
    apply solving obj1 for each objects in the entire dataframe
    '''                        
    print('Rectifying...')    
    # filter out len<2        
    df = df.groupby("ID").filter(lambda x: len(x)>=2)
    tqdm.pandas()            
    # lams = (1,0.2,0.2,0.05,0.02) # lambdas
    lams = (1,0,0,0.1,0.1,0) # 1:data perturb 2: acceleration 3: jerk 4: theta 5: omega
    df = applyParallel(df.groupby("ID"), rectify_single_camera, args = lams).reset_index(drop=True)
    # df = df.groupby('ID').apply(rectify_single_camera, args=lams).reset_index(drop=True)
    return df                
                            
def rectify_receding_horizon(df):
    '''                        
    apply solving obj1 for each objects in the entire dataframe
    '''                        
    # filter out len<2        
    df = df.groupby("ID").filter(lambda x: len(x)>=2)
    tqdm.pandas()            
    # df = df.groupby("ID").progress_apply(receding_horizon_opt).reset_index(drop=True)
    return df                
                            
                            
def receding_horizon_opt(car):
    '''                        
    Y,timestamps,w,l,n,PH,IH
    re-write the batch optimization (opt1 and op2) into mini-batch optimization to save computational time
    n: number of frames, assuming 30 fps
    PH: prediction horizon    
    IH: implementation horizon
                            
    '''                        
    w,l = estimate_dimensions(car) # use some data to estimate vehicle dimensions
    # print('estimated w:',w,'l:',l)
                            
    # optimization parameters
    lam1 = 1 # modification of measurement    
    lam2 = 1 # acceleration 
    lam3 = 0 # jerk            
    lam4 = 50 # theta        
    lam5 = 1 # omega        
    PH = 200 # optimize over Prediction Horizon frames
    IH = 100 # implementation horizon
                            
    sign = car['direction'].iloc[0]
    timestamps = car.Timestamp.values
    pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
    Y = np.array(car[pts])    
    n = len(Y)                
                            
    Yre = np.empty((0,8))    
    a_arr = np.empty((0,0)) 
    x_arr = np.empty((0,0)) 
    y_arr = np.empty((0,0)) 
    v_arr = np.empty((0,0)) 
    theta_arr = np.empty((0,0))
                            
    for i in range(0,n-IH,IH):
        # print(i,'/',n, flush=True)
        Y1 = Y[i:min(i+PH,n),:]
        N = len(Y1)            
        notNan = ~np.isnan(np.sum(Y1,axis=-1))
        # if (i>0) and (np.count_nonzero(notNan)<4): # TODO: does not work if first PH has not enough measurements!
            # if not enough measurement for this PH, simply use the last round of answers
            # Yre = np.vstack([Yre,Yre1[:N if i+PH>=n else PH-IH,:]])
            # a_arr = np.append(a_arr,a[:N if i+PH>=n else PH-IH:])
            # x_arr = np.append(x_arr,x[:N if i+PH>=n else PH-IH:])
            # y_arr = np.append(y_arr,y[:N if i+PH>=n else PH-IH:])
            # v_arr = np.append(v_arr,v[:N if i+PH>=n else PH-IH:])
            # theta_arr = np.append(theta_arr,theta[:N if i+PH>=n else PH-IH:])
            # continue        
        Y1 = Y1[notNan,:]    
        ts = timestamps[i:min(i+PH,n)]
        dt = np.diff(ts)    
                            
        a0 = np.zeros((N))    
        try:                
            v0 = v_arr[-1]    
        except:                
            v0 =(Y1[-1,0]-Y1[0,0])/(ts[notNan][-1]-ts[notNan][0])
        try:                
            x0 = x_arr[-1]    
            y0 = y_arr[-1]    
        except:                
            x0 = (Y1[0,0]+Y1[0,6])/2
            y0 = (Y1[0,1]+Y1[0,7])/2
                            
        v0 = np.abs(v0)        
        theta0 = np.ones((N))*np.arccos(sign)
        X0 = np.concatenate((a0.T, theta0.T, \
                             [v0,x0,y0]),axis=-1)
        if sign>0:            
            bnds = [(-5,5) for ii in range(0,N)]+\
                [(-np.pi/8,np.pi/8) for ii in range(N)]+\
                [(0,40),(-np.inf,np.inf),(0,np.inf)]
        else:                
            bnds = [(-5,5) for ii in range(0,N)]+\
                [(-np.pi/8+np.pi,np.pi/8+np.pi) for ii in range(N)]+\
                [(0,40),(-np.inf,np.inf),(0,np.inf)]
        res = minimize(obj2, X0, (Y1,N,dt,notNan,w,l,lam1,lam2,lam3,lam4,lam5), method = 'L-BFGS-B',
                        bounds=bnds, options={'disp': False,'maxiter':100000})#
                            
        # extract results    
        Yre1, x,y,v,a,theta,omega = unpack2(res,N,dt,w,l)
        Yre = np.vstack([Yre,Yre1[:N if i+PH>=n else IH,:]])
        a_arr = np.append(a_arr,a[:N if i+PH>=n else IH])
        x_arr = np.append(x_arr,x[:N if i+PH>=n else IH])
        y_arr = np.append(y_arr,y[:N if i+PH>=n else IH])
        v_arr = np.append(v_arr,v[:N if i+PH>=n else IH])
        theta_arr = np.append(theta_arr,theta[:N if i+PH>=n else IH])
                            
    # write into df            
    car.loc[:,pts] = Yre    
    car.loc[:,'acceleration'] = a_arr
    car.loc[:,'speed'] = v_arr
    car.loc[:,'x'] = x_arr    
    car.loc[:,'y'] = y_arr    
    car.loc[:,'theta'] = theta_arr
    car.loc[:,'width'] = w    
    car.loc[:,'length'] = l 
                            
    return car                
    # return Yre,a_arr,x_arr,v_arr,theta_arr
                            
                            
def estimate_dimensions(car):
    # optimization parameters
    car = car[(car['camera']=='p1c3') | (car['camera']=='p1c4')]
    # TODO: what to do if car has no measurements?
                            
    lam1 = 1 # modification of measurement
    lam2 = 1 # acceleration 
    lam3 = 0 # jerk            
    lam4 = 50 # theta        
    lam5 = 1 # omega        
    ts = car.Timestamp.values
    Y1 = np.array(car[['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']])
    N = len(Y1)                
    notNan = ~np.isnan(np.sum(Y1,axis=-1))
    Y1 = Y1[notNan,:]        
    dt = np.diff(ts)        
                            
    a0 = np.zeros((N))        
                            
    sign = car['direction'].iloc[0]
    v0 = (Y1[-1,0]-Y1[0,0])/(ts[notNan][-1]-ts[notNan][0])
    v0 = np.abs(v0)            
    theta0 = np.ones((N))*np.arccos(sign)
    x0 = (Y1[0,0]+Y1[0,6])/2
    y0 = (Y1[0,1]+Y1[0,7])/2
    X0 = np.concatenate((a0.T, theta0.T, \
                         [v0,x0,y0,np.nanmean(np.abs(Y1[:,1]-Y1[:,7])),\
                          np.nanmean(np.abs(Y1[:,0]-Y1[:,2]))]),axis=-1)
    if sign>0:                
        bnds = [(-5,5) for ii in range(0,N)]+\
            [(-np.pi/8,np.pi/8) for ii in range(N)]+\
            [(0,40),(-np.inf,np.inf),(0,np.inf),(1,2.59),(2,np.inf)]
    else:                    
        bnds = [(-5,5) for ii in range(0,N)]+\
            [(-np.pi/8+np.pi,np.pi/8+np.pi) for ii in range(N)]+\
            [(0,40),(-np.inf,np.inf),(0,np.inf),(1,2.59),(2,np.inf)]
                            
    res = minimize(obj1, X0, (Y1,N,dt,notNan,lam1,lam2,lam3,lam4,lam5), method = 'L-BFGS-B',
                    bounds=bnds, options={'disp': False,'maxiter':100000})#
                            
    # extract results        
    Yre, x,y,v,a,theta,omega,w,l = unpack1(res,N,dt)
    return w,l                
                            
def generate(w,l, x0, y0, theta,v,outputall=False):
    '''
    constant velocity dynamics
    '''
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

def calculate_score(Y1,Yre):
    '''
    for one box (frame)
    '''
    diff = Y1-Yre
    score = np.nanmean(LA.norm(diff,axis=1))
    return score
    
def score_for_box(w,l,Y):
    '''
    find the min score of a box of fixed w,l, with respect to measurement Y
    Y: 1x8
    '''
    eq_cons = {'type': 'eq',
        'fun' : lambda x: np.array([
                                    (x[2]-x[0])**2-l**2,
                                    (x[1]-x[7])**2-w**2,
                                    (x[0]-x[6])**2,
                                    (x[2]-x[4])**2,
                                    (x[1]-x[3])**2,
                                    (x[5]-x[7])**2])}
    X0 = Y[0]
    # X0 += np.random.normal(0, 0.1, X0.shape)
    res = minimize(calculate_score, X0, (Y), method = 'SLSQP',constraints=[eq_cons],
                options={'disp': False})
    print(res.fun)
    # plot_track(Y.reshape((1,8)))
    # plot_track(res.x.reshape((1,8)))
    return res
    
    