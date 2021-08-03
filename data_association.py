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
from tqdm import tqdm
import multiprocessing
import itertools

def dist_score(B, B_data, DIST_MEAS='xy'):
	'''
	compute euclidean distance between B and B_data
	B: predicted bbox location
	B_data: measurement
	'''

	# average displacement RMSE of all points
	if DIST_MEAS == 'xy':
		return np.linalg.norm(B-B_data,2)

	# weighted x,y displacement, penalize y more heavily
	elif DIST_MEAS == 'xyw':
		return 0.2*np.linalg.norm(B[[0,2,4,6]]-B_data[[0,2,4,6]],2) + 0.8*np.linalg.norm(B[[1,3,5,7]]-B_data[[1,3,5,7]],2)
	
	else:
		return
	
# def IOU_score(D1,D2):
	# '''
	# calculate the intersection of union of two boxes defined by d1 and D2
	# D1: prediction box
	# D2: measurement box
	# https://stackoverflow.com/questions/57885406/get-the-coordinates-of-two-polygons-intersection-area-in-python
	# '''
	# print(type(D1), type(D2))
	# if np.isnan(D2).any():
		# return np.nan
	# p = Polygon([(D1[2*i],D1[2*i+1]) for i in range(int(len(D1)/2))])
	# q = Polygon([(D2[2*i],D2[2*i+1]) for i in range(int(len(D2)/2))])
	# if (p.intersects(q)):
		# intersection_area = p.intersection(q).area
		# union_area = p.union(q).area
		  # print(intersection_area, union_area)
		# return float(intersection_area/union_area)
	# else:
		# return 0
def IOU_score(car1, car2):
	'''
	calculate the intersection of union of two boxes defined by d1 and D2
	D1: prediction box
	D2: measurement box
	https://stackoverflow.com/questions/57885406/get-the-coordinates-of-two-polygons-intersection-area-in-python
	'''
	end = min(car1['Frame #'].iloc[-1],car2['Frame #'].iloc[-1])
	start = max(car1['Frame #'].iloc[0],car2['Frame #'].iloc[0])
	
	if end <= start: # if no overlaps in time
		return -1
	car1 = car1.loc[(car1['Frame #'] >= start) & (car1['Frame #'] <= end)]
	car2 = car2.loc[(car2['Frame #'] >= start) & (car2['Frame #'] <= end)]
	pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
	Y1 = np.array(car1[pts]) # N x 8
	Y2 = np.array(car2[pts])
	IOU = 0
	N = 0
	for j in range(min(len(Y1), len(Y2))):
		D1 = Y1[j,:]
		# try:
		D2 = Y2[j,:]
		# except:
			# print(Y2.shape)
			# print(j)
			# print(car1)
			# print(car2)
		if ~np.isnan(np.sum([D1,D2])):
			p = Polygon([(D1[2*i],D1[2*i+1]) for i in range(int(len(D1)/2))])
			q = Polygon([(D2[2*i],D2[2*i+1]) for i in range(int(len(D2)/2))])
			if (p.intersects(q)):
				N += 1
				intersection_area = p.intersection(q).area
				union_area = p.union(q).area
		#		  print(intersection_area, union_area)
				IOU += float(intersection_area/union_area)
			else:
				IOU += 0
	if N == 0:
		return -1
	return IOU / N
	
def predict_tracks(tracks):
	'''
	tracks: [dictionary]. Key: car_id, value: mx8 matrix with footprint positions
	if a track has only 1 frame, make the second frame nans
	otherwise do constant-velocity one-step-forward prediction
	'''
	x = []
	for car_id, track in tracks.items():
		if len(track)>1:  
			delta = (track[-1,:] - track[0,:])/(len(track)-1)
			x_pred = track[-1,:] + delta
			tracks[car_id] = np.vstack([track, x_pred])
			x.append(x_pred) # prediction next frame, dim=nx8
		else:
#			  x_pred = np.nan*np.empty((1,8)) # nan as place holder, to be interpolated
			# TODO: assume traveling 30m/s based on direction (y axis)
			x_pred = track[-1,:] # keep the last measurement
			tracks[car_id] = np.vstack([track, x_pred])
			x.append(track[-1,:]) # take the last row
#			  raise Exception('must have at least 2 frames to predict')
	return x, tracks
			
			
def stitch_objects(df):
	SCORE_THRESHOLD = 4 # TODO: to be tested
	xmin = 0
	xmax = 400
	ns = np.amin(np.array(df[['Frame #']])) # start frame
	nf = np.amax(np.array(df[['Frame #']])) # end frame
	tracks = dict() # a dictionary to store all current objects in view
	parent = {} # a dictionary to store all associated tracks
	groups = df.groupby('ID')
	gl = list(groups.groups)
	for g in gl:
		parent[g] = g
				
	for k in range(ns,nf):
		if (k%10==0):
			print("Frame : %4d" % (k), flush=True)
		# get all measurements from current frame
		frame = df.loc[(df['Frame #'] == k)] # TODO: use groupby frame to save time
		y = np.array(frame[['bbr_x','bbr_y','fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']])
		notnan = ~np.isnan(y).any(axis=1)
		y = y[notnan] # remove rows with missing values (dim = mx8)
		frame = frame.iloc[notnan,:]
		
		m_box = len(frame)
		n_car = len(tracks)
		
		if (n_car > 0): # delete track that are out of view
			for car_id in list(tracks.keys()):
				last_frame_x = tracks[car_id][-1,[0,2,4,6]]
				x1 = min(last_frame_x)
				x2 = max(last_frame_x)
				if (x1<xmin) or (x2>xmax):
	#				  print('--------------- deleting ',car_id)
					del tracks[car_id]
					n_car -= 1
		
		if (m_box == 0) & (n_car == 0): # simply advance to the next frame
	#		  print('[1] frame ',k,', no measurement and no tracks')
			continue
			
		elif (m_box == 0) & (n_car > 0): # if no measurements in current frame
	#		  print('[2] frame ',k,', no measurement, simply predict')
			# make predictions to all existing tracks
			x, tracks = predict_tracks(tracks)
			
		elif (m_box > 0) & (n_car == 0): # create new tracks (initialize?)
	#		  print('[3] frame ',k,', no tracks, initialize with first measurements')
			for index, row in frame.iterrows():
				new_id = row['ID']
				ym = np.array(row[['bbr_x','bbr_y','fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']])
				tracks[new_id] = np.reshape(ym, (1,-1))
		
		else: # if measurement boxes exist in current frame k and tracks is not empty
			# make prediction for each track for frame k
			x, tracks = predict_tracks(tracks)
			n_car = len(tracks)
			curr_id = list(tracks.keys()) # should be n id's 
			
			# calculate score matrix: for car out of scene, score = 99 for place holder
			score = np.ones([m_box,n_car])*99
			for m in range(m_box):
				for n in range(n_car):
					score[m,n] = dist_score(x[n],y[m],'xyw')
				
			# identify associated (m,n) pairs
	#		  print('m:',m_box,'total car:',curr_id, 'car in view:',len(tracks))
			bool_arr = score == score.min(axis=1)[:,None]
			score =	 bool_arr*score+np.invert(bool_arr)*99 # get the min of each row
			pairs = np.transpose(np.where(score<SCORE_THRESHOLD)) # pair if score is under threshold
	#		  print(pairs)
			
			# associate based on pairs!
			if len(pairs) > 0:
	#			  print('[4a] frame ',k, len(pairs),' pairs are associated')
				for m,n in pairs:
					new_id = curr_id[n]
					old_id = frame['ID'].iloc[m]
					tracks[new_id][-1,:] = y[m] # change the last row from x_pred to ym				  
					parent[old_id] = new_id
					
			# measurements that have no cars associated, create new
			if len(pairs) < m_box:
	#			  print('pairs:',len(pairs),'measuremnts:',m_box)
				m_unassociated = list(set(np.arange(m_box)) - set(pairs[:,0]))
	#			  print('[4b] frame ',k, len(m_unassociated),' measurements are not associated, create new')
				for m in m_unassociated:
					new_id = frame['ID'].iloc[m]
					tracks[new_id] = np.reshape(y[m], (1,-1))
					
	parent = compress(parent,gl)				
	
	return parent
	
	
def associate_cross_camera(df_original):
	'''
	get all the ID pairs that associated to the same car
	do this BEFORE preprocess_multi_camera
	'''
	df = df_original.copy() # TODO: make this a method
	
	camera_list = ['p1c1','p1c2','p1c3','p1c4','p1c5','p1c6']
	# camera_list = ['p1c5','p1c6'] # for debugging
	groups = df.groupby('ID')
	gl = list(groups.groups)
	
	# initialize tree
	parent = {}
	for g in gl:
		parent[g] = g
			
	df = df.groupby(['ID']).filter(lambda x: len(x['camera'].unique()) != len(camera_list)) # filter
	SCORE_THRESHOLD = 0 # IOU score
	
	for i in range(len(camera_list)-1):
		camera1, camera2 = camera_list[i:i+2]
		print('Associating ', camera1, camera2)
		df2 = df[(df['camera']==camera1) | (df['camera']==camera2)]
		df2 = df2.groupby(['ID']).filter(lambda x: len(x['camera'].unique()) < 2) # filter
		
		groups2 = df2.groupby('ID')
		gl2 = list(groups2.groups)
		
		# initialize tree
		# parent = {}
		# for g in gl:
			# parent[g] = g
			
		comb = itertools.combinations(gl2, 2)
		
		for c1,c2 in comb:
			car1 = groups2.get_group(c1)
			car2 = groups2.get_group(c2)
			if ((car1['Object class'].iloc[0]) == (car2['Object class'].iloc[0])) & ((car1['camera'].iloc[0])!=(car2['camera'].iloc[0])):
				score = IOU_score(car1,car2)
				if score > SCORE_THRESHOLD:
					# associate!
					parent[c2] = c1
			else:
				continue
				
		# path compression (part of union find): compress multiple ID's to the same object			
	parent = compress(parent, gl)
		# change ID to first appeared ones
		# df2['ID'] = df2['ID'].apply(lambda x: parent[x] if x in parent else x)
		
	return parent

# path compression
def find(parent, i):
	if parent[parent[i]] == i:
		parent[i] = i
	if (parent[i] != i):
		parent[i] = find(parent, parent[i])
	return parent[i]

def compress(parent, groupList):	
	for i in groupList:
		find(parent, i)
	return parent 
	
	