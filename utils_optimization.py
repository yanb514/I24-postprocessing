import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import math
import os
from numpy import arctan2,random,sin,cos,degrees, arcsin, radians,arccos
from scipy.optimize import minimize,NonlinearConstraint,leastsq,fmin_slsqp,least_squares
import numpy.linalg as LA
from utils import *

def obj(X, Y1,N,dt,notNan, lam1,lam2,lam3,lam4,lam5):
	"""The cost function
		X = [j,alpha,a0,v0,x0,y0,theta0,w,l]^T
	""" 
	# unpack variables
	j = X[:N]
	omega = X[N:2*N]
	a0,v0,x0,y0,theta0,w,l = X[2*N:]
	
	a = np.zeros(N)
	a[0] = a0
	for k in range(0,N-2):
		a[k+1] = a[k] + j[k]*dt[k]
	a[-1] = a[-2]
	
	theta = np.zeros(N)
	theta[0] = theta0
	for k in range(0,N-1):
		theta[k+1] = theta[k] + omega[k]*dt[k]
	
	v = np.zeros(N)
	v[0] = v0
	for k in range(0,N-2):
		v[k+1] = v[k] + a[k]*dt[k]
	v[-1]=v[-2]
	vx = v*cos(theta)
	vy = v*sin(theta)
	
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

	# min perturbation
	c1 = lam1*LA.norm(Y1-Yre[notNan,:],'fro')/np.count_nonzero(notNan)
	c2 = lam2*LA.norm(a,2)/np.count_nonzero(notNan)
	c3 = lam3*LA.norm(j,2)/np.count_nonzero(notNan)
	c4 = lam4*LA.norm(theta,2)/np.count_nonzero(notNan)
	c5 = lam5*LA.norm(omega,2)/np.count_nonzero(notNan)
	return c1+c2+c3+c4+c5
	
	
def unpack(res,N,dt):
	# extract results
	# unpack variables
	j = res.x[:N]
	omega = res.x[N:2*N]
	a0,v0,x0,y0,theta0,w,l = res.x[2*N:]
	
	a = np.zeros(N)
	a[0] = a0
	for k in range(0,N-2):
		a[k+1] = a[k] + j[k]*dt[k]
	a[-1] = a[-2]
	
	theta = np.zeros(N)
	theta[0] = theta0
	for k in range(0,N-1):
		theta[k+1] = theta[k] + omega[k]*dt[k]
	
	v = np.zeros(N)
	v[0] = v0
	for k in range(0,N-2):
		v[k+1] = v[k] + a[k]*dt[k]
	v[-1]=v[-2]
	vx = v*cos(theta)
	vy = v*sin(theta)

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
	return Yre, x,y,v,a,j,theta,omega,w,l
	
	
	
	
	
def obj1(X, Y1,N,dt,notNan, lam1,lam2,lam3,lam4,lam5):
	"""The cost function
		X = [a,theta,v0,x0,y0,w,l]^T
	""" 
	# unpack variables
	a = X[:N]
	theta = X[N:2*N]
	omega = np.diff(theta)/dt
	omega = np.append(omega,omega[-1])
	v0,x0,y0,w,l = X[2*N:]
	
	v = np.zeros(N)
	v[0] = v0
	for k in range(0,N-2):
		v[k+1] = v[k] + a[k]*dt[k]
	v[-1]=v[-2]
	vx = v*cos(theta)
	vy = v*sin(theta)
	
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

	# min perturbation
	c1 = lam1*LA.norm(Y1-Yre[notNan,:],'fro')/np.count_nonzero(notNan)
	c2 = lam2*LA.norm(a,2)/np.count_nonzero(notNan)
	# c3 = lam3*LA.norm(j,2)/np.count_nonzero(notNan)
	c4 = lam4*LA.norm(theta,2)/np.count_nonzero(notNan)
	c5 = lam5*LA.norm(omega,2)/np.count_nonzero(notNan)
	return c1+c2+c4
	
	
def unpack1(res,N,dt):
	# extract results
	# unpack variables

	a = res.x[:N]
	theta = res.x[N:2*N]
	omega = np.diff(theta)/dt
	omega = np.append(omega,omega[-1])
	v0,x0,y0,w,l = res.x[2*N:]
	
	v = np.zeros(N)
	v[0] = v0
	for k in range(0,N-2):
		v[k+1] = v[k] + a[k]*dt[k]
	v[-1]=v[-2]
	vx = v*cos(theta)
	vy = v*sin(theta)

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
	return Yre, x,y,v,a,theta,omega,w,l
	
	
	
	
def obj2(X, Y1,N,dt,notNan, w,l,lam1,lam2,lam3,lam4,lam5):
	"""The cost function given w, l
		X = [a,theta,v0,x0,y0,w,l]^T
	""" 
	# unpack variables
	a = X[:N]
	theta = X[N:2*N]
	omega = np.diff(theta)/dt
	omega = np.append(omega,omega[-1])
	v0,x0,y0 = X[2*N:]
	
	v = np.zeros(N)
	v[0] = v0
	for k in range(0,N-2):
		v[k+1] = v[k] + a[k]*dt[k]
	v[-1]=v[-2]
	vx = v*cos(theta)
	vy = v*sin(theta)
	
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

	# min perturbation
	c1 = lam1*LA.norm(Y1-Yre[notNan,:],'fro')/np.count_nonzero(notNan)
	c2 = lam2*LA.norm(a,2)/np.count_nonzero(notNan)
	# c3 = lam3*LA.norm(j,2)/np.count_nonzero(notNan)
	c4 = lam4*LA.norm(theta,2)/np.count_nonzero(notNan)
	c5 = lam5*LA.norm(omega,2)/np.count_nonzero(notNan)
	return c1+c2+c4
	
	
def unpack2(res,N,dt,w,l):
	# extract results
	# unpack variables

	a = res.x[:N]
	theta = res.x[N:2*N]
	omega = np.diff(theta)/dt
	omega = np.append(omega,omega[-1])
	v0,x0,y0 = res.x[2*N:]
	
	v = np.zeros(N)
	v[0] = v0
	for k in range(0,N-2):
		v[k+1] = v[k] + a[k]*dt[k]
	v[-1]=v[-2]
	vx = v*cos(theta)
	vy = v*sin(theta)

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
	return Yre, x,y,v,a,theta,omega
	
	
def estimate_dimensions(Y1, ts,lam1,lam2,lam3,lam4,lam5):
	N = len(Y1)
	notNan = ~np.isnan(np.sum(Y1,axis=-1))
	Y1 = Y1[notNan,:]
	dt = np.diff(ts)

	a0 = np.zeros((N))
	theta0 = np.zeros((N))
	v0 = (Y1[-1,0]-Y1[0,0])/(ts[notNan][-1]-ts[notNan][0])
	x0 = (Y1[0,0]+Y1[0,6])/2
	y0 = (Y1[0,1]+Y1[0,7])/2
	X0 = np.concatenate((a0.T, theta0.T, \
						 [v0,x0,y0,np.nanmean(np.abs(Y1[:,1]-Y1[:,7])),\
						  np.nanmean(np.abs(Y1[:,0]-Y1[:,2]))]),axis=-1)
	bnds = [(-5,5) for ii in range(0,N)]+\
		[(-np.pi/8,np.pi/8) for ii in range(N)]+\
		[(0,40),(-np.inf,np.inf),(0,np.inf),(1,4),(2,np.inf)]

	res = minimize(obj1, X0, (Y1,N,dt,notNan,lam1,lam2,lam3,lam4,lam5), method = 'L-BFGS-B',
					bounds=bnds, options={'disp': False,'maxiter':100000})#

	# extract results
	Yre, x,y,v,a,theta,omega,w,l = unpack1(res,N,dt)
	return w,l
	
def create_synth_data(n):
	timestamps =  np.linspace(0,n/30,n)
	theta = np.zeros(n)
	theta = np.random.normal(0, .02, theta.shape) + theta
	w = np.ones(n)*2 + np.random.normal(0, .2, n) 
	l = np.ones(n)*4 + np.random.normal(0, .4, n) 
	x = np.linspace(0,100,n)
	x = np.random.normal(0, .1, x.shape) + x
	y = np.ones(n)
	xa = x + w/2*sin(theta)
	ya = y - w/2*cos(theta)
	xb = xa + l*cos(theta)
	yb = ya + l*sin(theta)
	xc = xb - w*sin(theta)
	yc = yb + w*cos(theta)
	xd = xa - w*sin(theta)
	yd = ya + w*cos(theta)
	Y = np.stack([xa,ya,xb,yb,xc,yc,xd,yd],axis=-1)
	return timestamps,Y
	
def create_true_data(n):
	''' same as create_synth_data except no noise
	'''
	timestamps =  np.linspace(0,n/30,n)
	theta = np.zeros(n)
	w = np.ones(n)*2
	l = np.ones(n)*4
	x = np.linspace(0,100,n)
	y = np.ones(n)
	xa = x + w/2*sin(theta)
	ya = y - w/2*cos(theta)
	xb = xa + l*cos(theta)
	yb = ya + l*sin(theta)
	xc = xb - w*sin(theta)
	yc = yb + w*cos(theta)
	xd = xa - w*sin(theta)
	yd = ya + w*cos(theta)
	Y = np.stack([xa,ya,xb,yb,xc,yc,xd,yd],axis=-1)
	return timestamps,Y
	
def receding_horizon_opt(Y,timestamps,w,l,n,lam1,lam2,lam3,lam4,lam5,PH,IH):
	'''
	re-write the batch optimization (opt1 and op2) into mini-batch optimization to save computational time
	n: number of frames, assuming 30 fps
	PH: prediction horizon
	IH: implementation horizon
	
	'''
	Yre = np.empty((0,8))
	for i in range(0,n-IH,IH):
		Y1 = Y[i:min(i+PH,n),:]
		N = len(Y1)
		notNan = ~np.isnan(np.sum(Y1,axis=-1))
		Y1 = Y1[notNan,:]
		ts = timestamps[i:i+PH]
		dt = np.diff(ts)

		a0 = np.zeros((N))
		theta0 = np.zeros((N))
		v0 = (Y1[-1,0]-Y1[0,0])/(ts[notNan][-1]-ts[notNan][0])
		x0 = (Y1[0,0]+Y1[0,6])/2
		y0 = (Y1[0,1]+Y1[0,7])/2
		X0 = np.concatenate((a0.T, theta0.T, \
							 [v0,x0,y0]),axis=-1)
		bnds = [(-5,5) for ii in range(0,N)]+\
			[(-np.pi/8,np.pi/8) for ii in range(N)]+\
			[(0,40),(-np.inf,np.inf),(0,np.inf)]
		res = minimize(obj2, X0, (Y1,N,dt,notNan,w,l,lam1,lam2,lam3,lam4,lam5), method = 'L-BFGS-B',
						bounds=bnds, options={'disp': False,'maxiter':100000})#

		# extract results
		Yre1, x,y,v,a,theta,omega = unpack2(res,N,dt,w,l)
		Yre = np.vstack([Yre,Yre1[:N if i+PH>=n else IH,:]])
	return Yre
	
	