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

# ==================== FOR 1D DOT TOY EXAMPLE ONLY ===============
def generate_1d(x0, v0, a, dt):
    '''
    generate vehicle states using 1-st order dynamics
    x(k+1) = x(k)+v(k)dt
    v(k+1) = v(k)+a(k)dt
    
    x0, v0: initial position and velocity, in float
    a: Nx1 array, acceleration, but only a[:-2] is used in dynamics
    dt: float or Nx1 array
    
    return: x,v,a
    '''
    N = len(a)
    
    x = np.zeros(N)
    v = np.zeros(N)
    x[0] = x0
    v[0] = v0
    for k in range(0,N-2):
        v[k+1] = v[k] + a[k]*dt
        x[k+1] = x[k] + v[k]*dt
    x[-1] = x[-2] + v[-2]*dt 
    v[-1] = v[-2]
    
    return x,v,a

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

def obj_1d(X, x, N, dt, notNan, lam):
    """ The cost function for 1d
            X: decision variables X = [xhat, vhat, ahat]
                xhat: rectified positions (N x 1)
                ahat: rectified acceleration (N-2 x 1)
            x: position measurements (N x 1), could have NaN
            N: int number of measurements
            dt: float delta T 
            notNan: bool array indicating timestamps that have measurements
            lam: given parameter
        Return: float
    """ 
    nvalid = np.count_nonzero(notNan)
    # unpack decision variables
    xhat = X[:N]
    vhat = X[N:2*N]
    ahat = X[2*N:]
    
    # select valid measurements
    xhat = xhat[notNan]
    x = x[notNan]
    
    # min perturbation
    # c1 = loss(x, xhat, 'l2')
    c1 = LA.norm(x-xhat,2)**2
    # print("c1: ",c1)
    # regularize acceleration # not the real a or j, multiply a constant dt
    c2 = LA.norm(ahat,2)**2
    # print("c2: ",c2)
    cost = lam*c1 + (1-lam) * c2
    # print(cost)
    return c1

def const_1d(n, dt):
    """ The constraint representing linear dynamics
        n: number of decision variables
        Return: matrix A (nxn), such that A dot X = [0]_(nx1)
        for scipy.optimize.LinearConstraint: lb<= A.dot(X) <= ub 'trust-constr'
    """ 
    A = np.zeros((n,n))
    N = int(n/3) # number of total timestamps
    
    # xhat(1) = xhat(0)+vhat(0)*dt
    # A[1][0] = -1
    # A[1][1] = 1
    # A[1][N] = -dt
    
    for i in range(1, N): # xhat
        # xhat(k+1)-xhat(k)-vhat(k)*dt = 0
        A[i][i] = 1
        A[i][i-1] = -1
        A[i][i-1+N] = -dt
        
        # xhat(k+1)=xhat(k)+vhat(k)*dt = xhat(k)+vhat(k-1)*dt+ahat(k-1)dt^2
        # A[i][i] = 1
        # A[i][i-1] = -1
        # A[i][i-2+N] = -dt
        # A[i][i-3+N+N] = -dt**2
        
    for i in range(N+1, 2*N): # vhat 
        # vhat(k+1)-vhat(k)-ahat(k)*dt = 0
        A[i][i] = 1
        A[i][i-1] = -1
        A[i][i-1+N] = -dt
    
    
    # ahat[-1] = ahat[-3], ahat[-2] = ahat[-3]
    A[-1][-1] = 1
    A[-1][-2] = -1
    A[-2][-2] = 1
    A[-2][-3] = -1
    
    return A
 
   
def rectify_1d(df, args):
    '''                        
    solve for the optimization problem:
        minimize obj_1d(X, x, N, dt, notNan, lam)
        s.t. A = const_1d(X, dt), AX = 0  
    args: tuple (lam. niter)
    '''  
    # TODO: deal with signs
    # get data
    x = df.x.values
    v = df.speed.values
    a = df.acceleration.values
    N = len(x)
    n = 3*N
    lam,niter = args
    
    dt = 1/30
    # sign = df["direction"].iloc[0]           
    notNan = ~np.isnan(x)

    # Define constraints
    A = const_1d(n, dt)
    linear_constraint = LinearConstraint(A, np.zeros(n), np.zeros(n)) 
    # Bounds constraints
    lb = [-np.inf]*N + [0]*N + [-10]*N
    ub = [np.inf]*N + [50]*N + [10]*N
    bounds = Bounds(lb,ub)
    
    # Initialize decision variables
    nans, ind = nan_helper(a)
    a[nans]= np.interp(ind(nans), ind(~nans), a[~nans])
    xhat0,vhat0,ahat0 = generate_1d(x[0], v[0], a, dt)
    X0 = np.concatenate((xhat0,vhat0,ahat0), axis=0)

    initial_cost = obj_1d(X0, x, N, dt, notNan, lam)
    
    print("Initial cost: ", initial_cost)
    print("Initial constraint: ",  LA.norm(np.dot(A, X0),2))

    # 3 methods for constrained optimization
    # 1. trust-constr
    # res = minimize(obj_1d, X0, (x, N, dt, notNan, lam),method='trust-constr',
    #             constraints=[linear_constraint],
    #             options={'verbose': 1, 'maxiter': 50}, bounds=bounds)
    # minimizer_kwargs = {"method":"trust-constr", "args":(x, N, dt, notNan, lam),
    #                     "constraints":[linear_constraint],'bounds':bounds,'options':{'verbose': 0}}
    # res = basinhopping(obj_1d, X0, minimizer_kwargs=minimizer_kwargs,niter=niter)
    
    # 2. SLSQP
    eq_cons = {'type': 'eq',
            'fun' : lambda x: LA.norm(np.dot(A, x),2)**2}  
    # res = minimize(obj_1d, X0, (x, N, dt, notNan, lam), method='SLSQP',
    #            constraints=[eq_cons], options={'ftol': 1e-9, 'disp': True},
    #            bounds=bounds)
    minimizer_kwargs = {"method":"SLSQP", "args":(x, N, dt, notNan, lam),
                        "constraints": [eq_cons],'bounds':bounds,'options':{'disp': True}}
    res = basinhopping(obj_1d, X0, minimizer_kwargs=minimizer_kwargs,niter=niter)
    
    # 3. COBYLA
    
    final_cost = obj_1d(res.x, x, N, dt, notNan, lam)
    print("Final cost: ", final_cost)
    print("Final constraint: ",  LA.norm(np.dot(A, res.x),2))

    # extract results    
    xhat = res.x[:N]
    vhat = res.x[N:2*N]
    # vhat = np.append(vhat, vhat[-1])
    ahat = res.x[2*N:]
    # ahat = np.append(ahat, ahat[-2:])
    
    # write to df
    df.loc[:,'acceleration'] = ahat
    df.loc[:,'speed'] = vhat   
    df.loc[:,'x'] = xhat           
    
    return df

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
                            
def generate(w,l,x0, y0, theta,v,outputall=False):
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
    
    