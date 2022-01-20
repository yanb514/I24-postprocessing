import pandas as pd
import numpy as np
import numpy.linalg as LA
from numpy import sin,cos
from scipy.optimize import minimize, basinhopping, LinearConstraint,Bounds
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
import utils
import utils_vis as vis

# ==================== FOR debugging ONLY ===============
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
    dt: float or Nx1 array
    order: highest order of derivative. 2: acceleration. 3: jerk
    return: x,v,a
    '''
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
    # nvalid = np.count_nonzero(notNan)
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
    c2 = LA.norm(highest_order_dynamics,2)**2 / (N-order)

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
 
   
def rectify_1d(df, args, axis):
    '''                        
    solve for the optimization problem for both x and y component independently:
        minimize obj_1d(X, x, N, dt, notNan, lam)
        s.t. A = const_1d(X, dt), AX = 0  
    args: tuple (lam, order)
    axis: "x" or "y"
    lam_norm: automatically adjust lambda to scale perturbation and regularization
    '''  
    # get data
    lam, order = args
    dt = 1/30
    
    if axis == "x":
        x = df.x.values      
    else:
        x = df.y.values
    v = np.append(np.diff(x)/dt,np.nan)
    a = np.append(np.diff(v)/dt,np.nan)
    j = np.append(np.diff(a)/dt,np.nan)
    
    N = len(x)
    
    # sign = df["direction"].iloc[0]           
    notNan = ~np.isnan(x)

    # Define constraints
    A = const_1d(N, dt, order) 
    
    # Bounds constraints
    lb,ub = [],[]
    # if axis == "x":
    lb_list = [-np.inf, -np.inf, -np.inf, -np.inf] # x0, v0, a0, j0
    ub_list = [np.inf, np.inf,np.inf, np.inf]
    # else:
    #     lb_list = [0, -3, -1, -0.5] # y0, v0, a0, j0
    #     ub_list = [50, 3, 1, 0.5]
    for o in range(order+1):
        lb += [lb_list[o]]*(N-o)
        ub += [ub_list[o]]*(N-o)

    bounds = Bounds(lb,ub)
    
    # Initialize decision variables
    nans, ind = nan_helper(x)
    x[nans]= np.interp(ind(nans), ind(~nans), x[~nans])
    v[nans]= np.interp(ind(nans), ind(~nans), v[~nans])
    a[nans]= np.interp(ind(nans), ind(~nans), a[~nans])
    j[nans]= np.interp(ind(nans), ind(~nans), j[~nans])
    highest_order_dyn = j if order==3 else a
    
    xhat0,vhat0,ahat0,jhat0 = generate_1d([x[0], v[0],a[0]], highest_order_dyn, dt, order)
    if order == 3:
        X0 = np.concatenate((xhat0,vhat0[:-1],ahat0[:-2],jhat0[:-3]), axis=0)
    elif order == 2:
        X0 = np.concatenate((xhat0,vhat0[:-1],ahat0[:-2]), axis=0)
    
    # 2. SLSQP
    eq_cons = {'type': 'eq', 
            'fun' : lambda x: np.dot(A, x)}

    # res_lam0 = minimize(obj_1d, X0, (x, order,N, dt, notNan, 0), method='SLSQP',
    #             constraints=[eq_cons], options={'ftol': 1e-9, 'disp': False},
    #             bounds=bounds)
    # print(res_lam0.message)
    # res_lam1 = minimize(obj_1d, X0, (x, order,N, dt, notNan, 1), method='SLSQP',
    #             constraints=[eq_cons], options={'ftol': 1e-9, 'disp': False},
    #             bounds=bounds)
    # print(res_lam1.message)
    # c1min = obj_1d(res_lam1.x, x,order, N, dt, notNan, 1)
    # c2min = obj_1d(res_lam0.x, x,order, N, dt, notNan, 0)
    # c1max = obj_1d(res_lam0.x, x,order, N, dt, notNan, 1)
    # c2max = obj_1d(res_lam1.x, x,order, N, dt, notNan, 0)
    
    # if lam_norm:
    #     norm_args = (c1min, c2min, c1max, c2max)
    #     print("axis: ", axis, " ", norm_args)
    # else:
    #     norm_args = None
    
    # SOLVE AGAIN - WITH NORMALIZED OBJECTIVES
    res = minimize(obj_1d, X0, (x, order,N, dt, notNan, lam), method='SLSQP',
                constraints=[eq_cons], options={'ftol': 1e-9, 'disp': False},
                bounds=bounds)
    print(res.message)
    # final_cost1 = obj_1d(res.x, x,order, N, dt, notNan, 1)
    # final_cost2 = obj_1d(res.x, x,order, N, dt, notNan, 0)
    # print("Final cost: ", final_cost1,final_cost2)
    # print("Final constraint: ", LA.norm(np.dot(A, res.x),1))
    
    # extract results    
    xhat = res.x[:N]
    vhat = res.x[N:2*N-1]
    vhat = np.append(vhat,np.nan)
    ahat = res.x[2*N-1:3*N-3]
    ahat = np.append(ahat, np.ones(2)*np.nan)
    jhat = res.x[3*N-3:]
    jhat = np.append(jhat, np.ones(3)*np.nan)
    
    return xhat, vhat, ahat, jhat

def rectify_2d(df, w,l,args):
    '''
    rectify on x and y component independently
    '''
    lamx, lamy, order = args
    xhat, vxhat, axhat, jxhat = rectify_1d(df, (lamx,order), "x")
    yhat, vyhat, ayhat, jyhat = rectify_1d(df, (lamy,order), "y")
    
    # calculate the states
    
    vhat = np.sqrt(vxhat**2 + vyhat**2) # non-negative speed
    thetahat = np.arcsin(vyhat/vhat)
    ahat = np.append(np.diff(vhat)/(1/30),  np.ones(1)*np.nan)
    jhat = np.append(np.diff(ahat)/(1/30),  np.ones(1)*np.nan)
    
    # expand to full boxes measurements
    Y0 = generate_box(w,l, xhat, yhat, thetahat)
    
    # write to df
    df.loc[:,'x'] = xhat
    df.loc[:,'jerk'] = jhat
    df.loc[:,'acceleration'] = ahat
    df.loc[:,'speed'] = vhat   
    df.loc[:,'y'] = yhat   
    df.loc[:,'theta'] = thetahat
    
    # store auxiliary states for debugging
    df.loc[:,'speed_x'] = vxhat
    df.loc[:,'jerk_x'] = jxhat
    df.loc[:,'acceleration_x'] = axhat
    df.loc[:,'speed_y'] = vyhat
    df.loc[:,'jerk_y'] = jyhat
    df.loc[:,'acceleration_y'] = ayhat 
    
    pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
    df.loc[:, pts] = Y0
    
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

# ============================================
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
    pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
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
    
    