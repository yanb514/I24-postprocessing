import pandas as pd
import numpy as np
from cvxopt import matrix, solvers, sparse,spdiag,spmatrix
from bson.objectid import ObjectId
from collections import defaultdict

from i24_logger.log_writer import logger, catch_critical, log_warnings, log_errors

# TODO
# add try except and put errors/warnings to log
logger.set_name("reconciliation_module")
solvers.options['show_progress'] = False
dt = 1/30


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
    
    stacked = defaultdict(list)

    if isinstance(stitched_doc, list):
        fragment_ids = stitched_doc
    else:
        fragment_ids = stitched_doc["fragment_ids"]
    if isinstance(fragment_ids[0], str):
        fragment_ids = [ObjectId(_id) for _id in fragment_ids]

    # logger.info("fragment_ids type: {}, {}".format(type(fragment_ids), fragment_ids))
    # logger.debug("first doc {}".format(raw_collection.find_one(fragment_ids[0]))) # this returns none
    
    stacked["fragment_ids"] = fragment_ids
    all_fragment = raw_collection.find({"_id": {"$in": fragment_ids}}) # returns a cursor

    for fragment in all_fragment:
        # logger.debug("fragment keys: {}".format(fragment.keys()))
        stacked["timestamp"].extend(fragment["timestamp"])
        stacked["x_position"].extend(fragment["x_position"])
        stacked["y_position"].extend(fragment["y_position"])
        stacked["road_segment_ids"].extend(fragment["road_segment_ids"])
        stacked["flags"].extend(fragment["flags"])
        stacked["length"].extend(fragment["length"])
        stacked["width"].extend(fragment["width"])
        stacked["height"].extend(fragment["height"])
        # stacked["detection_confidence"].extend(fragment["detection_confidence"])
        stacked["coarse_vehicle_class"].append(fragment["coarse_vehicle_class"])
        stacked["fine_vehicle_class"].append(fragment["fine_vehicle_class"])
        stacked["direction"].append(fragment["direction"])
        
        stacked["filter"].extend(fragment["filter"])
        
       
    # first fragment
    first_id = fragment_ids[0]
    # logger.debug("** first_id: {}, type: {}".format(first_id, type(first_id)), extra = None)
    # logger.debug("** timestamp: {}, collection size: {}".format(stacked["timestamp"], raw_collection.count_documents({})), extra = None)
    
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
    
    # Take the most frequent element of the list
    stacked["coarse_vehicle_class"] = max(set(stacked["coarse_vehicle_class"]), key = stacked["coarse_vehicle_class"].count)
    stacked["fine_vehicle_class"] = max(set(stacked["fine_vehicle_class"]), key = stacked["fine_vehicle_class"].count)
    stacked["direction"] = max(set(stacked["direction"]), key = stacked["direction"].count)
    
    # Apply filter
    if len(stacked["filter"]) == 0: # no good measurements
        stacked["post_flag"] = "low conf fragment"
    else:
        stacked["x_position"] = [stacked["x_position"][i] if stacked["filter"][i] == 1 else np.nan for i in range(len(stacked["filter"])) ]
        stacked["y_position"] = [stacked["y_position"][i] if stacked["filter"][i] == 1 else np.nan for i in range(len(stacked["filter"])) ]
    return stacked



@catch_critical(errors = (Exception))    
def resample(car):
    # resample timestamps to 30hz, leave nans for missing data
    '''
    resample the original time-series to uniformly sampled time series in 30Hz
    car: document
    leave empty slop as nan
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

    # resample to 25hz
    # df=df.groupby(df.index.floor('0.04S')).mean().resample('0.04S').asfreq()
    # df.index = df.index.values.astype('datetime64[ns]').astype('int64')*1e-9
    # df = df.interpolate(method='linear')


    # df=df.groupby(df.index.floor('0.04S')).mean().resample('0.04S').asfreq() # resample to 25Hz snaps to the closest integer
    car['x_position'] = df['x_position'].values
    car['y_position'] = df['y_position'].values
    car['timestamp'] = df.index.values
        
    return car






# ==================== CVX optimization for 2d dynamics ================== 
@catch_critical(errors = (Exception))    
def opt1(car, lam3_x, lam3_y):
    '''
    1/M||z-Hx||_2^2 + \lam3/N ||D3x||_2^2
    '''
    x = car["x_position"]
    y = car["y_position"]
    # x
    Q, p, H, N, M = _get_qp_opt1(x, lam3_x)
    sol=solvers.qp(P=Q, q=p)
    xhat = sol["x"][:N]
    
    # y
    Q, p, H, N, M = _get_qp_opt1(y, lam3_y)
    sol=solvers.qp(P=Q, q=p)
    yhat = sol["x"][:N]
    
    car["timestamp"] = list(car["timestamp"])
    car["x_position"] = list(xhat)
    car["y_position"] = list(yhat)
    
    # calculate residual
    xhat_re = np.reshape(xhat, -1) # (N,)
    yhat_re = np.reshape(yhat, -1) # (N,)
    # c1 = np.sqrt(np.nansum((x-xhat_re)**2)/M) # RMSE
    cx = np.nansum(np.abs(x-xhat_re))/M # MAE
    cy = np.nansum(np.abs(y-yhat_re))/M # MAE
    car["x_score"] = cx
    car["y_score"] = cy
    
    return car

def opt1_l1(car, lam3_x, lam3_y, lam1_x, lam1_y):
    '''
    1/M||z-Hx||_2^2 + \lam3/N ||D3x||_2^2 + \lam1/M ||e||_1
    '''
    x = car["x_position"]
    y = car["y_position"]
    
    # x
    Q, p, H, G, h, N,M = _get_qp_opt1_l1(x, lam3_x, lam1_x)
    sol=solvers.qp(P=Q, q=matrix(p) , G=G, h=matrix(h))
    xhat = sol["x"][:N]
    
    # y
    Q, p, H, G, h, N,M = _get_qp_opt1_l1(y, lam3_y, lam1_y)
    sol=solvers.qp(P=Q, q=matrix(p) , G=G, h=matrix(h))
    yhat = sol["x"][:N]
    
    car["timestamp"] = list(car["timestamp"])
    car["x_position"] = list(xhat)
    car["y_position"] = list(yhat)
    
    # calculate residual
    xhat_re = np.reshape(xhat, -1) # (N,)
    yhat_re = np.reshape(yhat, -1) # (N,)
    # c1 = np.sqrt(np.nansum((x-xhat_re)**2)/M) # RMSE
    cx = np.nansum(np.abs(x-xhat_re))/M # MAE
    cy = np.nansum(np.abs(y-yhat_re))/M # MAE
    car["x_score"] = cx
    car["y_score"] = cy
    
    return car

def opt2(car, lam2_x, lam2_y, lam3_x, lam3_y):
    '''
    1/M||z-Hx||_2^2 + \lam2/N ||D2x||_2^2 + \lam3/N ||D3x||_2^2
    '''
    x = car["x_position"]
    y = car["y_position"]
    # x
    Q, p, H, N, M = _get_qp_opt2(x, lam2_x, lam3_x)
    sol=solvers.qp(P=Q, q=p)
    xhat = sol["x"][:N]
    
    # y
    Q, p, H, N, M = _get_qp_opt2(y, lam2_y, lam3_y)
    sol=solvers.qp(P=Q, q=p)
    yhat = sol["x"][:N]
    
    car["timestamp"] = list(car["timestamp"])
    car["x_position"] = list(xhat)
    car["y_position"] = list(yhat)
    
    # calculate residual
    xhat_re = np.reshape(xhat, -1) # (N,)
    yhat_re = np.reshape(yhat, -1) # (N,)
    # c1 = np.sqrt(np.nansum((x-xhat_re)**2)/M) # RMSE
    cx = np.nansum(np.abs(x-xhat_re))/M # MAE
    cy = np.nansum(np.abs(y-yhat_re))/M # MAE
    car["x_score"] = cx
    car["y_score"] = cy
    
    return car

def opt2_l1(car, lam2_x, lam2_y, lam3_x, lam3_y, lam1_x, lam1_y):
    '''
    1/M||z-Hx||_2^2 + \lam2/N ||D2x||_2^2 + \lam3/N ||D3x||_2^2 + \lam1/M ||e||_1
    '''
    x = car["x_position"]
    y = car["y_position"]
    
    # x
    maxax = 99
    
    while maxax > 10:
        Q, p, H, G, h, N, M, D2 = _get_qp_opt2_l1(x, lam2_x, lam3_x, lam1_x)
        sol=solvers.qp(P=Q, q=matrix(p) , G=G, h=matrix(h))
        xhat = sol["x"][:N]
        ax = D2*xhat
        maxax = max(abs(ax))
        # print("ax: {:.2f}, {:.2f}".format(min(ax), max(ax)))
        lam2_x += 0.001
    
    # y
    Q, p, H, G, h, N, M, D2 = _get_qp_opt2_l1(y, lam2_y, lam3_y, lam1_y)
    sol=solvers.qp(P=Q, q=matrix(p) , G=G, h=matrix(h))
    yhat = sol["x"][:N]
    
    car["timestamp"] = list(car["timestamp"])
    car["x_position"] = list(xhat)
    car["y_position"] = list(yhat)
    
    # calculate residual
    xhat_re = np.reshape(xhat, -1) # (N,)
    yhat_re = np.reshape(yhat, -1) # (N,)
    # c1 = np.sqrt(np.nansum((x-xhat_re)**2)/M) # RMSE
    cx = np.nansum(np.abs(x-xhat_re))/M # MAE
    cy = np.nansum(np.abs(y-yhat_re))/M # MAE
    car["x_score"] = cx
    car["y_score"] = cy
    
    return car 


def _get_qp_opt1(x, lam3):
    '''
    rewrite opt1 to QP form:
    min 1/2 z^T Q x + p^T z + r
    s.t. Gz <= h
    input:  x: data array with missing data
    return: Q, p, H, (G, h if l1)
    '''
    # get data
    N = len(x)
    
    # non-missing entries
    idx = [i.item() for i in np.argwhere(~np.isnan(x)).flatten()]
    x = x[idx]
    M = len(x)
    
    if M == 0 or N <= 3:
        raise ZeroDivisionError
        
    # differentiation operator
    D3 = _blocdiag(matrix([-1,3,-3,1],(1,4), tc="d"), N) * (1/dt**3)
    
    DD = lam3 * D3.trans() * D3
    # sol: xhat = (I+delta D'D)^(-1)x
    I = spmatrix(1.0, range(N), range(N))
    H = I[idx,:]
    DD = lam3 * D3.trans() * D3
    HH = H.trans() * H

    Q = 2*(HH/M+DD/(N-3))
    p = -2*H.trans() * matrix(x)/M

    return Q, p, H, N, M


def _get_qp_opt1_l1(x, lam3, lam1):
    '''
    rewrite opt1_l1 to QP form:
    min 1/2 z^T Q x + p^T z + r
    s.t. Gz <= h
    '''
    # get data
    N = len(x)
    
    # non-missing entries
    idx = [i.item() for i in np.argwhere(~np.isnan(x)).flatten()]
    x = x[idx]
    M = len(x)
    
    if M == 0 or N-3 <= 0:
        raise ZeroDivisionError
    # differentiation operator
    # D1 = _blocdiag(matrix([-1,1],(1,2), tc="d"), N) * (1/dt)
    # D2 = _blocdiag(matrix([1,-2,1],(1,3), tc="d"), N) * (1/dt**2)
    D3 = _blocdiag(matrix([-1,3,-3,1],(1,4), tc="d"), N) * (1/dt**3)
    
    DD = lam3 * D3.trans() * D3
    # define matices
    I = spmatrix(1.0, range(N), range(N))
    IM = spmatrix(1.0, range(M), range(M))
    O = spmatrix([], [], [], (N,N))
    OM = spmatrix([], [], [], (M,M))
    H = I[idx,:]
    HH = H.trans()*H

    Q = 2*sparse([[HH/M+DD/(N-3),H/M,-H/M], # first column of Q
                [H.trans()/M,IM/M, -H*H.trans()/M], 
                [-H.trans()/M,-H*H.trans()/M,IM/M]]) 
    
    p = 1/M * sparse([-2*H.trans()*matrix(x), -2*matrix(x)+lam1, 2*matrix(x)+lam1])
    G = sparse([[H*O,H*O],[-IM,OM],[OM,-IM]])
    h = spmatrix([], [], [], (2*M,1))
    
    return Q, p, H, G, h, N, M



def _get_qp_opt2(x, lam2, lam3):
    '''
    rewrite opt2 to QP form:
    min 1/2 z^T Q x + p^T z + r   
    return: Q, p, H, M
    '''
    # get data
    N = len(x)
    
    # non-missing entries
    idx = [i.item() for i in np.argwhere(~np.isnan(x)).flatten()]
    x = x[idx]
    M = len(x)
    
    if M == 0 or N <= 3:
        raise ZeroDivisionError
        
    # differentiation operator
    D2 = _blocdiag(matrix([1,-2,1],(1,3), tc="d"), N) * (1/dt**2)
    D3 = _blocdiag(matrix([-1,3,-3,1],(1,4), tc="d"), N) * (1/dt**3)
    
    DD2 = lam2 * D2.trans() * D2
    DD3 = lam3 * D3.trans() * D3
    # sol: xhat = (I+delta D'D)^(-1)x
    I = spmatrix(1.0, range(N), range(N))
    H = I[idx,:]
    HH = H.trans() * H

    Q = 2*(HH/M +DD2/(N-2) + DD3/(N-3))
    p = -2*H.trans() * matrix(x)/M

    return Q, p, H, N, M


def _get_qp_opt2_l1(x, lam2, lam3, lam1):
    '''
    rewrite opt2_l1 to QP form:
    min 1/2 z^T Q x + p^T z + r
    s.t. Gz <= h
    '''
    # get data
    N = len(x)
    
    # non-missing entries
    idx = [i.item() for i in np.argwhere(~np.isnan(x)).flatten()]
    x = x[idx]
    M = len(x)
    
    if M == 0 or N-3 <= 0:
        raise ZeroDivisionError
    # differentiation operator
    D2 = _blocdiag(matrix([1,-2,1],(1,3), tc="d"), N) * (1/dt**2)
    D3 = _blocdiag(matrix([-1,3,-3,1],(1,4), tc="d"), N) * (1/dt**3)
    DD2 = lam2 * D2.trans() * D2
    DD3 = lam3 * D3.trans() * D3
    # define matices
    I = spmatrix(1.0, range(N), range(N))
    IM = spmatrix(1.0, range(M), range(M))
    O = spmatrix([], [], [], (N,N))
    OM = spmatrix([], [], [], (M,M))
    H = I[idx,:]
    HH = H.trans()*H

    Q = 2*sparse([[HH/M+DD2/(N-2)+DD3/(N-3),H/M,-H/M], # first column of Q
                [H.trans()/M,IM/M, -H*H.trans()/M], 
                [-H.trans()/M,-H*H.trans()/M,IM/M]]) 
    
    p = 1/M * sparse([-2*H.trans()*matrix(x), -2*matrix(x)+lam1, 2*matrix(x)+lam1])
    G = sparse([[H*O,H*O],[-IM,OM],[OM,-IM]])
    h = spmatrix([], [], [], (2*M,1))
    
    return Q, p, H, G, h, N, M, D2


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







    
if __name__ == '__main__': 
    print("not implemented")

        
    
    
    
    
    
    