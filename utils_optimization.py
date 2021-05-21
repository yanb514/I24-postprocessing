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

def obj(X, Y1,N,dt,notNan, lam1,lam2,lam3,lam4):
	"""The cost function
		X = [a,alpha,v0,x0,y0,omega0,theta0,w,l]^T
	""" 
	# unpack variables
	a = X[:N]
	theta = X[N:2*N]
	v0,x0,y0,w,l = X[2*N:]
	
	# omega = np.zeros(N)
	# omega[0] = omega0
	# for k in range(0,N-2):
		# omega[k+1] = omega[k] + alpha[k]*dt[k]
	# omega[-1] = omega[-2]
	
	# theta = np.zeros(N)
	# theta[0] = theta0
	# for k in range(0,N-1):
		# theta[k+1] = theta[k] + omega[k]*dt[k]
	
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
	# print('Y1:',np.isnan(Y1).any()) # false, doesn't have nan
	# print(Yre[notNan,:]) # true
	# min perturbation
	c1 = lam1*LA.norm(Y1-Yre[notNan,:],2)
	c2 = lam2*LA.norm(a,2)
	# c3 = lam3*LA.norm(alpha,2)
	c4 = lam4*LA.norm(theta,2)

	# c3 = lam3*LA.norm(alpha,2)
	return c1+c2+c4
	
	
def unpack(res,N,dt):
	# extract results
	# unpack variables
	a = res.x[:N]
	theta = res.x[N:2*N]
	v0,x0,y0,w,l = res.x[2*N:]

	# omega = np.zeros(N)
	# omega[0] = omega0
	# for k in range(0,N-2):
		# omega[k+1] = omega[k] + alpha[k]*dt[k]
	# omega[-1] = omega[-2]
	
	# theta = np.zeros(N)
	# theta[0] = theta0
	# for k in range(0,N-1):
		# theta[k+1] = theta[k] + omega[k]*dt[k]
	
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
	return Yre, x,y,v,a,theta,w,l
	
	