import pandas as pd
import numpy as np
import numpy.linalg as LA
from numpy import sin,cos
# from scipy.optimize import minimize, basinhopping, LinearConstraint,Bounds
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
from cvxopt import matrix, solvers, sparse,spdiag,spmatrix
import cvxopt.lapack as cla
from scipy.interpolate import InterpolatedUnivariateSpline  

# import utils_vis as vis
solvers.options['show_progress'] = False
pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
dt = 1/30

# ==================== CVX optimization for 2d dynamics ==================
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
    

    
def _blocdiag(X, n):
    """
    makes diagonal blocs of X, for indices in [sub1,sub2[
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

  
def rectify_1d(df, args, axis):
    '''                        
    solve solve for ||y-x||_2^2 + \lam ||Dx||_2^2
    axis: "x" or "y"
    '''  
    # get data
    lam = args
    x = df[axis].values
    N = len(x)
          
    idx = [i.item() for i in np.argwhere(~np.isnan(x)).flatten()]
    x = x[idx]

    # differentiation operator
    D1 = _blocdiag(matrix([-1,1],(1,2), tc="d"), N) * (1/dt)
    D2 = _blocdiag(matrix([1,-2,1],(1,3), tc="d"), N) * (1/dt**2)
    D3 = _blocdiag(matrix([-1,3,-3,1],(1,4), tc="d"), N) * (1/dt**3)
    # D4 = _blocdiag(matrix([1,-4,6,-4,1],(1,5), tc="d"), N) * (1/dt**4)
    
    # sol: xhat = (I+delta D'D)^(-1)x
    I = spmatrix(1.0, range(N), range(N))
    H = I[idx,:]
    DD = lam*D3.trans() * D3
    HH = H.trans() * H
    
    # analytical solution
    # A = I + DD
    # xhat = +matrix(x)
    # cla.gesv(+matrix(A), xhat) # solve AX=B, change b in-place
    
    # QP formulation with sparse matrix min ||y-x||_2^2 + \lam ||Dx||_2^2
    Q = 2*(HH+DD)
    p = -2*H.trans() * matrix(x)
    sol=solvers.qp(P=Q, q=p)
    # print(sol["status"])
    
    # extract result
    xhat = sol["x"][:N]
    jhat = D3 * xhat
    ahat = D2 * xhat
    vhat = D1 * xhat
    return xhat, vhat, ahat, jhat

def rectify_1d_l1(df, args, axis):
    '''                        
    solve for ||y-x-e||_2^2 + \lam ||Dx||_2^2 + \delta||e||_1
    convert to quadratic programming with linear inequlity constraints
    handle sparse outliers in data
    '''  
    # get data
    lam, delta = args
    x = df[axis].values
    N = len(x)

    # impute missing data 
    idx = [i.item() for i in np.argwhere(~np.isnan(x)).flatten()]
    x = x[idx]
    M = len(x)
           
    # idx = np.argwhere(np.isnan(x)).flatten()
    # idx = [i.item() for i in idx]
    # idx1 = [i+N for i in idx]
    # idx2 = [i+2*N for i in idx]
    # t = df["Frame #"].values
    # spl = InterpolatedUnivariateSpline(t[notnan_idx],x[notnan_idx])
    # x = spl(t)
    
    # differentiation operator
    D1 = _blocdiag(matrix([-1,1],(1,2), tc="d"), N) * (1/dt)
    D2 = _blocdiag(matrix([1,-2,1],(1,3), tc="d"), N) * (1/dt**2)
    D3 = _blocdiag(matrix([-1,3,-3,1],(1,4), tc="d"), N) * (1/dt**3)
    DD = lam * D3.trans() * D3
    
    # define matices
    I = spmatrix(1.0, range(N), range(N))
    IM = spmatrix(1.0, range(M), range(M))
    O = spmatrix([], [], [], (N,N))
    OM = spmatrix([], [], [], (M,M))

    H = I[idx,:]
    HH = H.trans()*H
    Q = 2*sparse([[HH+DD,H*I,spmatrix([], [], [], (M,N))], 
                [I*H.trans(),IM,OM], 
                [spmatrix([], [], [], (N,M)),OM,OM]]) 
    
    p = sparse([-2*H.trans()*matrix(x), -2*matrix(x), matrix(delta, (M,1))])

    G = sparse([[H*O,H*O],[IM,-IM],[-IM,-IM]])
    h = spmatrix([], [], [], (2*M,1))
    sol=solvers.qp(P=Q, q=matrix(p) , G=G, h=matrix(h))
    
    # extract result
    xhat = sol["x"][:N]
    e = sol["x"][N:N+M]
    t = sol["x"][N+M:]
    print(sol["status"])

    jhat = D3 * xhat
    ahat = D2 * xhat
    vhat = D1 * xhat
    
    return xhat, vhat, ahat, jhat

def rectify_2d(df, w,l,args):
    '''
    rectify on x and y component independently
    '''
    try:
        lamx, lamy = args
    except: lamx, lamy, delta = args
    df.loc[:,'y'] = (df["bbr_y"].values + df["bbl_y"].values)/2
    df.loc[:,'x'] = (df["bbr_x"].values + df["bbl_x"].values)/2
    
    if len(args) == 2:
        xhat, vxhat, axhat, jxhat = rectify_1d(df, lamx, "x")
        yhat, vyhat, ayhat, jyhat = rectify_1d(df, lamy, "y")
    elif len(args) == 3:
        xhat, vxhat, axhat, jxhat = rectify_1d_l1(df, (lamx, delta), "x")
        yhat, vyhat, ayhat, jyhat = rectify_1d_l1(df, (lamy, delta), "y")
        
    # calculate the states
    vhat = np.sqrt(vxhat**2 + vyhat**2) # non-negative speed (N-1)
    thetahat = np.arctan2(vyhat,vxhat)
    thetahat[thetahat < 0] += 2*np.pi
    D1 = _blocdiag(matrix([-1,1],(1,2), tc="d"), len(xhat)) * (1/dt)
    ahat = D1[:-1,:-1] * matrix(vhat)
    # ahat = np.append(ahat, ahat[-1])
    
    # add constant jerk in the end
    jhat = D1[:-2,:-2] * matrix(ahat)
    # jhat =  np.append(jhat, 0)
    # expand to full boxes measurements
    Y0 = generate_box(w,l, np.reshape(xhat,-1), np.reshape(yhat,-1), np.reshape(np.append(thetahat, thetahat[-1]),-1))
    
    # write to df
    df.loc[:,'x'] = xhat
    df.loc[:,'jerk'] = matrix([jhat,matrix(3*[np.nan])])
    df.loc[:,'acceleration'] = matrix([ahat,matrix(2*[np.nan])])
    df.loc[:,'speed'] = matrix([matrix(vhat),matrix([np.nan])])   
    df.loc[:,'y'] = yhat   
    df.loc[:,'theta'] = matrix([matrix(thetahat),matrix([np.nan])])
    
    # store auxiliary states for debugging
    df.loc[:,'speed_x'] = matrix([vxhat,matrix([np.nan])])  
    df.loc[:,'jerk_x'] = matrix([jxhat,matrix(3*[np.nan])])
    df.loc[:,'acceleration_x'] = matrix([axhat,matrix(2*[np.nan])])
    df.loc[:,'speed_y'] = matrix([vyhat,matrix([np.nan])])  
    df.loc[:,'jerk_y'] = matrix([jyhat,matrix(3*[np.nan])])
    df.loc[:,'acceleration_y'] = matrix([ayhat,matrix(2*[np.nan])]) 
    
    # pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
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

def rectify_single_car(car, args):
    '''
    to replace rectify_single_camera
    1. get the medium width and length
    2. find the best-fit x and y without considering the dynamics
    3. run rectify_2d
    4. run generate_box to get full box rectified measurement
    '''
    width_array = np.abs(car["bbr_y"].values - car["bbl_y"].values)
    length_array = np.abs(car["bbr_x"].values - car["fbr_x"].values)
    
    width = np.nanmedian(width_array)
    length = np.nanmedian(length_array)
    
    # car = box_fitting(car, width, length) # modify x and y
    # car = utils.mark_outliers(car)
    # print(pts)
    # car.loc[car["Generation method"]=="outlier1",pts] = np.nan
    # print("outlier ratio: {}/{}".format( np.count_nonzero(~np.isnan(car.bbr_y.values)),len(car)))
    # car = decompose_2d(car, write_to_df=True)

    # short tracks
    # if np.count_nonzero(~np.isnan(car.bbr_y.values)) < 4:
    #     print("Short track ", car.ID.values[0])
    #     return car
    car = rectify_2d(car, width, length,args) # modify x, y, speed, acceleration, jerk, boxes coords
    return car

def rectify_sequential(df, args):
    
    df = df.groupby('ID').apply(rectify_single_car, args=args).reset_index(drop=True)
    return df

# =================== RECEDING HORIZON RECTIFICATION =========================
def receding_horizon_1d(df, args, axis="x"):
    '''
    rolling horizon version of rectify_1d
    car: df
    args: (lam, order, axis, PH, IH)
        PH: prediction horizon
        IH: implementation horizon
    
    '''
    # get data
    lam, order, PH, IH = args
    
    x = df[axis].values
    
    n_total = len(x)          
    notNan = ~np.isnan(x)

    # Define constraints
    A = const_1d(PH, dt, order) 
    P, q,b,G,h = constr_qp(np.ones(PH),lam,order,np.array([True]*PH))
    
    # additional equality constraint: state continuity
    A2 = np.zeros((4, A.shape[1]))
    A2[[0,1,2,3], [0,1,2,3]] = 1
    AA = np.vstack((A, A2))
    AA,G,h = matrix(AA, tc="d") ,matrix(G, tc="d"),matrix(h, tc="d")
    
    # save final answers
    xfinal = np.ones(n_total) * np.nan
    vfinal = np.ones(n_total) * np.nan
    afinal = np.ones(n_total) * np.nan
    jfinal = np.ones(n_total) * np.nan
    
    last = False
    for i in range(0,n_total-PH+IH,IH):
        print(i,'/',n_total, flush=True)
        if i+PH >= n_total: # last block
            last = True
            xx = x[i:]
            nn = len(xx)  
            notNan = ~np.isnan(xx)
            nan_idx = np.argwhere(~notNan)
        
            # redefine matrices
            A = const_1d(nn, dt, order) 
            PP, qq,b,G,h = constr_qp(xx,lam,order,notNan)
            
            # additional equality constraint: state continuity
            A2 = np.zeros((4, A.shape[1]))
            A2[[0,1,2,3], [0,1,2,3]] = 1
            AA = np.vstack((A, A2))
            AA = matrix(AA, tc="d") 
            
        else:
            xx = x[i: i+PH]
            nn = len(xx)  
            notNan = ~np.isnan(xx)
            nan_idx = np.argwhere(~notNan)
            # update matrices based on nans          
            PP,qq = P.copy(), q.copy()
            PP[nan_idx, :] = 0 # set the corresponding rows to 0
            qq[:PH]*=xx
            qq[nan_idx] = 0 
        
        if i==0:
            PP, qq, b, A = matrix(PP, tc="d"), matrix(qq, tc="d"), matrix(b, tc="d"), matrix(A, tc="d")
            sol=solvers.qp(P=PP, q=qq ,A=A, b=b)
            # print(sol["status"])
        else:
            # additional constraint AA X = bb
            bb = np.append(b, x_prev)
            PP, qq, bb = matrix(PP, tc="d"), matrix(qq, tc="d"), matrix(bb, tc="d")
            sol=solvers.qp(P=PP, q=qq ,A=AA, b=bb)
            # print(sol["status"])
            
        res = np.array(sol["x"])[:,0]
        # print(sol["status"])
         
        # unpack decision varibles
        xhat = res[:nn]
        vhat = res[nn:2*nn-1]
        ahat = res[2*nn-1:3*nn-3]
        jhat = res[3*nn-3:]
        
        if last:
            xfinal[i:i+nn] = xhat
            vfinal[i:i+nn-1] = vhat
            afinal[i:i+nn-2] = ahat
            jfinal[i:i+nn-3] = jhat
        else:
            xfinal[i:i+IH] = xhat[:IH]
            vfinal[i:i+IH] = vhat[:IH]
            afinal[i:i+IH] = ahat[:IH]
            jfinal[i:i+IH] = jhat[:IH]
            
        # save for the next loop
        x_prev = res[IH:IH+4]
        del PP, qq
    
    return xfinal, vfinal,  afinal, jfinal

def receding_horizon_2d(df, w,l,args):
    '''
    '''
    # get data
    lamx, lamy, order, PH, IH = args
    df.loc[:,'y'] = (df["bbr_y"].values + df["bbl_y"].values)/2
    df.loc[:,'x'] = (df["bbr_x"].values + df["bbl_x"].values)/2
    
    xhat,vxhat,axhat,jxhat = receding_horizon_1d(df, (lamx, order, PH, IH), "x")
    yhat,vyhat,ayhat,jyhat = receding_horizon_1d(df, (lamy, order, PH, IH), "y")
    
    # calculate the states
    vhat = np.sqrt(vxhat**2 + vyhat**2) # non-negative speed
    thetahat = np.arctan2(vyhat,vxhat)
    thetahat[thetahat < 0] += 2*np.pi
    ahat = np.diff(vhat)/dt
    ahat = np.append(ahat,  ahat[-1])
    
    # add constant jerk in the end
    jhat = np.diff(ahat)/dt
    jhat = np.append(jhat, 0)
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
    
    # pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
    df.loc[:, pts] = Y0
    
    
    return df













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
    
    