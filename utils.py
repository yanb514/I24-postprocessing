import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import arctan2,random,sin,cos,degrees, radians
from bs4 import BeautifulSoup
from IPython.display import IFrame
import gmplot 
import cv2
import csv
import sys
import re
import glob
from tqdm import tqdm
from utils_optimization import *
from data_association import *
from functools import partial
import time
import itertools
from itertools import combinations
from shapely.geometry import Polygon

# read data
def read_data(file_name, skiprows = 0, index_col = False):	 
#	  path = pathlib.Path().absolute().joinpath('tracking_outputs',file_name)
	df = pd.read_csv(file_name, skiprows = skiprows,error_bad_lines=False,index_col = index_col)
	df = df.rename(columns={"GPS lat of bbox bottom center": "lat", "GPS long of bbox bottom center": "lon", 'Object ID':'ID'})
	return df
	

def reorder_points(df):
	'''
		make sure points in the road-plane do not flip
	'''
	pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
	Y = np.array(df[pts]).astype("float")
	xsort = np.sort(Y[:,[0,2,4,6]])
	ysort = np.sort(Y[:,[1,3,5,7]])
	try:
		if df['direction'].values[0]== 1:
			Y = np.array([xsort[:,0],ysort[:,0],xsort[:,2],ysort[:,1],
			xsort[:,3],ysort[:,2],xsort[:,1],ysort[:,3]]).T
		else:
			Y = np.array([xsort[:,2],ysort[:,2],xsort[:,0],ysort[:,3],
			xsort[:,1],ysort[:,0],xsort[:,3],ysort[:,1]]).T
	except np.any(xsort<0) or np.any(ysort<0):
		print('Negative x or y coord, please redefine reference point A and B')
		sys.exit(1)

	df.loc[:,pts] = Y
	return df

def filter_width_length(df):
	'''
	filter out bbox if their width/length is 2 std-dev's away
	'''
	pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']

	Y = np.array(df[pts]).astype("float")
	
	# filter outlier based on width	
	w1 = np.abs(Y[:,3]-Y[:,5])
	w2 = np.abs(Y[:,1]-Y[:,7])
	m = np.nanmean([w1,w2])
	s = np.nanstd([w1,w2])
	# print(m,s)
	outliers =	np.logical_or(abs(w1 - m) > 2 * s,abs(w2 - m) > 2 * s)
	# print('width outlier:',np.count_nonzero(outliers))
	Y[outliers,:] = np.nan
	
	# filter outlier based on length
	l1 = np.abs(Y[:,0]-Y[:,2])
	l2 = np.abs(Y[:,4]-Y[:,6])
	m = np.nanmean([l1,l2])
	s = np.nanstd([l1,l2])
	# print(m,s)
	outliers =	np.logical_or(abs(l1 - m) > 2 * s,abs(l2 - m) > 2 * s)
	# print('length outlier:',np.count_nonzero(outliers))
	Y[outliers,:] = np.nan
	
	isnan = np.isnan(np.sum(Y,axis=-1))
	
	# write into df
	df.loc[:,pts] = Y
	return df
	
def filter_short_track(df):

	Y1 = df['bbrx']
	N = len(Y1) 
	notNans = np.count_nonzero(~np.isnan(df['bbrx']))

	if (notNans <= 3) or (N <= 3):
		# print('track too short: ', df['ID'].iloc[0])
		return False
	return True
	
def naive_filter_3D(df):
	# filter out direction==0
	df = df.groupby("ID").filter(lambda x: x['direction'].values[0] != 0)
	print('after direction=0 filter: ',len(df['ID'].unique()))
	# reorder points
	df = df.groupby("ID").apply(reorder_points).reset_index(drop=True)
	# filter out short tracks
	# df = df.groupby("ID").filter(filter_short_track)
	# print('after filtering short tracks: ',len(df['ID'].unique()))
	# filter out-of-bound length and width
	df = df.groupby("ID").apply(filter_width_length).reset_index(drop=True)
	print('filter width length:', len(df['ID'].unique()))
	return df

def findLongestSequence(car, k=0):
	'''
	keep the longest continuous frame sequence for each car
	# https://www.techiedelight.com/find-maximum-sequence-of-continuous-1s-can-formed-replacing-k-zeroes-ones/	
	'''
	A = np.diff(car['Frame #'].values)
	A[A!=1]=0
	
	left = 0		# represents the current window's starting index
	count = 0		# stores the total number of zeros in the current window
	window = 0		# stores the maximum number of continuous 1's found
					# so far (including `k` zeroes)
 
	leftIndex = 0	# stores the left index of maximum window found so far
 
	# maintain a window `[left…right]` containing at most `k` zeroes
	for right in range(len(A)):
 
		# if the current element is 0, increase the count of zeros in the
		# current window by 1
		if A[right] == 0:
			count = count + 1
 
		# the window becomes unstable if the total number of zeros in it becomes
		# more than `k`
		while count > k:
			# if we have found zero, decrement the number of zeros in the
			# current window by 1
			if A[left] == 0:
				count = count - 1
 
			# remove elements from the window's left side till the window
			# becomes stable again
			left = left + 1
 
		# when we reach here, window `[left…right]` contains at most
		# `k` zeroes, and we update max window size and leftmost index
		# of the window
		if right - left + 1 > window:
			window = right - left + 1
			leftIndex = left
 
	# print the maximum sequence of continuous 1's
#	  print("The longest sequence has length", window, "from index",
#		  leftIndex, "to", (leftIndex + window - 1))
	return car.iloc[leftIndex:leftIndex + window - 1,:]
	
def preprocess(file_path, tform_path, skip_row = 0):
	'''
	preprocess for one single camera data
	skip_row: number of rows to skip when reading csv files to dataframe
	'''
	print('Reading data...')
	df = read_data(file_path,skip_row)
	if (df.columns[0] != 'Frame #'):
		df = read_data(file_path,9)
	if 'Object ID' in df:
		df.rename(columns={"Object ID": "ID"})
	if 'frx' in df:
		df = df.rename(columns={"frx":"fbr_x", "fry":"fbr_y", "flx":"fbl_x", "fly":"fbl_y","brx":"bbr_x","bry":"bbr_y","blx":"bbl_x","bly":"bbl_y"})
	print('Total # cars before preprocessing:', len(df['ID'].unique()))
	camera_name = find_camera_name(file_path)
	print('Transform from image to road for {}...'.format(camera_name))
	
	df = img_to_road(df, tform_path, camera_name)
	# print('Deleting unrelavent columns...')
	# df = df.drop(columns=['BBox xmin','BBox ymin','BBox xmax','BBox ymax','vel_x','vel_y','lat','lon'])
	
	print('Interpret missing timestamps...')
	frames = [min(df['Frame #']),max(df['Frame #'])]
	times = [min(df['Timestamp']),max(df['Timestamp'])]
	if np.isnan(times).any(): # if no time is recorded
		print('No timestamps values')
		p = np.poly1d([1/30,0]) # frame 0 corresponds to time 0, fps=30
	else:
		z = np.polyfit(frames,times, 1)
		p = np.poly1d(z)
	df['Timestamp'] = p(df['Frame #'])
	
	print('Constrain x,y range by camera FOV')
	if 'camera' not in df:
		df['camera'] = find_camera_name(file_path)
	if len(df['camera'].unique())==1:
		xmin, xmax, ymin, ymax = get_camera_range(df['camera'][0])
	else:
		xmin, xmax, ymin, ymax = get_camera_range('all')
		
	print(xmin, xmax, ymin, ymax)
	pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
	df.loc[(df['bbr_x'] < xmin) | (df['bbr_x'] > xmax), pts] = np.nan # 
	df.loc[(df['bbr_y'] < ymin) | (df['bbr_y'] > ymax), pts] = np.nan # 

	print('Filtering out tailing place holders...')
	df = df.groupby('ID').apply(remove_tailing_place_holders).reset_index(drop=True)
	print('Get the longest continuous frame chuck...')
	df = df.groupby('ID').apply(findLongestSequence).reset_index(drop=True)
	print('Get x direction...')
	df = get_x_direction(df)
	print('Naive filter...')
	df = naive_filter_3D(df)
	
	return df

def remove_tailing_place_holders(car):
	notnan = ~np.isnan(np.sum(np.array(car[['bbr_x']]),axis=1))
	if np.count_nonzero(notnan)>0:
		start = np.where(notnan)[0][0]
		end = np.where(notnan)[0][-1]
		car = car.iloc[start:end+1]
	return car
	
def find_camera_name(file_path):
	camera_name_regex = re.compile(r'p(\d)*c(\d)*')
	camera_name = camera_name_regex.search(str(file_path))
	return camera_name.group()


 # for visualization
def insertapikey(fname):
	apikey = 'AIzaSyDBo88RY_39Evn87johzUvFw5x_Yg6cfkI'
#	  """put the google api key in a html file"""
	print('\n###############################')
	print('\n Beginning Key Insertion ...')

	def putkey(htmltxt, apikey, apistring=None):
#		  """put the apikey in the htmltxt and return soup"""
		if not apistring:
			apistring = "https://maps.googleapis.com/maps/api/js?key=%s&callback=initialize&libraries=visualization&sensor=true_or_false"
		soup = BeautifulSoup(htmltxt, 'html.parser')
		soup.script.decompose() #remove the existing script tag
		body = soup.body
		src = apistring % (apikey, )
		tscript = soup.new_tag("script", src=src) #, async="defer"
		body.insert(-1, tscript)
		return soup

	htmltxt = open(fname, 'r').read()
	# htmltxt = open(fname,'r+').read()
	soup = putkey(htmltxt, apikey)
	newtxt = soup.prettify()
	open(fname, 'w').write(newtxt)
	print('\nKey Insertion Completed!!')


def jupyter_display(gmplot_filename):
	google_api_key = 'AIzaSyDBo88RY_39Evn87johzUvFw5x_Yg6cfkI'
	
#	  """Hack to display a gmplot map in Jupyter"""
	with open(gmplot_filename, "r+b") as f:
		f_string = f.read()
		url_pattern = "https://maps.googleapis.com/maps/api/js?libraries=visualization&sensor=true_or_false"
		newstring = url_pattern + "&key=%s" % google_api_key
		f_string = f_string.replace(url_pattern.encode(), newstring.encode())
		f.write(f_string)
	return IFrame(gmplot_filename, width=900, height=600)

def draw_map_scatter(x,y):
	
	map_name = "test.html"
	gmap = gmplot.GoogleMapPlotter(x[0], y[0], 100) 

	gmap.scatter(x, y, s=.9, alpha=.8, c='red',marker = False)
	gmap.draw(map_name)
	
	insertapikey(map_name)
	return jupyter_display(map_name)
	
def draw_map(df, latcenter, loncenter, nO):
	
	map_name = "test.html"
	gmap = gmplot.GoogleMapPlotter(latcenter, loncenter, 100) 

	groups = df.groupby('ID')
	groupList = list(groups.groups)

	for i in groupList[:nO]:   
		group = groups.get_group(i)
		gmap.scatter(group.lat, group.lon, s=.5, alpha=.8, label=group.loc[group.index[0],'ID'],marker = False)
	gmap.draw(map_name)
	
	insertapikey(map_name)
	return jupyter_display(map_name)

# draw rectangles from 3D box on map
def draw_map_box(Y, nO, lats, lngs):
	
	map_name = "test.html"
	notNan = ~np.isnan(np.sum(Y,axis=-1))
	Y = Y[notNan,:]
	gmap = gmplot.GoogleMapPlotter(Y[0,0], Y[0,1], nO) 

	# get the bottom 4 points gps coords
	# Y = np.array(df[['bbrlat','bbrlon','fbrlat','fbrlon','fbllat','fbllon','bbllat','bbllon']])
	

	for i in range(len(Y)):
		coord = Y[i,:]
		coord = np.reshape(coord,(-1,2)).tolist()
		coord.append(coord[0]) #repeat the first point to create a 'closed loop'
		coord_tuple = [tuple(pt) for pt in coord]
		rectangle = zip(*coord_tuple) #create lists of x and y values
		gmap.polygon(*rectangle)	
	lats = lats[~np.isnan(lats)]
	lngs = lngs[~np.isnan(lngs)]
	gmap.scatter(lats, lngs, color='red', size=1, marker=True)
	gmap.scatter(Y[:,2], Y[:,3],color='red', size=0.1, marker=False)

	gmap.draw(map_name)

	insertapikey(map_name)
	return jupyter_display(map_name)
	
def get_x_direction(df):
	return df.groupby("ID").apply(ffill_direction).reset_index(drop=True)

def ffill_direction(df):
	bbrx = df['bbr_x'].values
	notnan = ~np.isnan(bbrx)
	bbrx = bbrx[notnan]
	
	if (len(bbrx)<=1):
		sign = 0
	else:
		sign = np.sign(bbrx[-1]-bbrx[0])
		
	# if all the y axis are below 18
	bbry = df['bbr_y'].values[notnan]
	if (bbry < 18).all():
		signy = 1
	elif (bbry > 18).all():
		signy = -1
	else:
		signy = 0
		
	if sign == signy:
		df = df.assign(direction = sign)
	else:
		df = df.assign(direction = 0)
	return df



def get_homography_matrix(camera_id, tform_path):
	'''
	camera_id: pxcx
	read from Derek's new transformation file
	'''
	# find and read the csv file corresponding to the camera_id
	tf_file = glob.glob(str(tform_path) + '/' + camera_id + "*.csv")
	tf_file = tf_file[0]
	tf = pd.read_csv(tf_file)
	M = np.array(tf.iloc[-3:,0:3], dtype=float)
	return M
	
def img_to_road(df,tform_path,camera_id,ds=1):
	'''
	ds: downsample rate
	'''
	M = get_homography_matrix(camera_id, tform_path)
	for pt in ['fbr','fbl','bbr','bbl']:
		img_pts = np.array(df[[pt+'x', pt+'y']]) # get pixel coords
		img_pts = img_pts/ds # downsample image to correctly correspond to M
		img_pts_1 = np.vstack((np.transpose(img_pts), np.ones((1,len(img_pts))))) # add ones to standardize
		road_pts_un = M.dot(img_pts_1) # convert to gps unnormalized
		road_pts_1 = road_pts_un / road_pts_un[-1,:][np.newaxis, :] # gps normalized s.t. last row is 1
		road_pts = np.transpose(road_pts_1[0:2,:])/3.281 # only use the first two rows, convert from ft to m
		df[[pt+'_x', pt+'_y']] = pd.DataFrame(road_pts, index=df.index)
	return df
	
def img_to_road_box(img_pts_4,tform_path,camera_id):
	'''
	the images are downsampled
	img_pts: N x 8
	'''
	M = get_homography_matrix(camera_id, tform_path)
	print(img_pts_4.shape)
	road_pts_4 = np.empty([len(img_pts_4),0])
	for i in range(4):
		img_pts = img_pts_4[:,2*i:2*i+1]
		print(img_pts.shape)
		img_pts = img_pts/2 # downsample image to correctly correspond to M
		img_pts_1 = np.vstack((np.transpose(img_pts), np.ones((1,len(img_pts))))) # add ones to standardize
		road_pts_un = M.dot(img_pts_1) # convert to gps unnormalized
		road_pts_1 = road_pts_un / road_pts_un[-1,:][np.newaxis, :] # gps normalized s.t. last row is 1
		road_pts = np.transpose(road_pts_1[0:2,:])/3.281 # only use the first two rows, convert from ft to m
		road_pts_4 = np.hstack([road_pts_4, road_pts])
	return road_pts
	
def get_xy_minmax(df):

	if isinstance(df, pd.DataFrame):
		Y = np.array(df[['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']])
	else:
		Y = df
	notNan = ~np.isnan(np.sum(Y,axis=-1))
	Yx = Y[:,[0,2,4,6]]
	# print(np.where(Yx[notNan,:]==Yx[notNan,:].min()))
	Yy = Y[:,[1,3,5,7]]
	return Yx[notNan,:].min(),Yx[notNan,:].max(),Yy[notNan,:].min(),Yy[notNan,:].max()
	
def extend_prediction(car, args):
	'''
	extend the dynamics of the vehicles that are still in view
	'''

	xmin, xmax, maxFrame = args
	dir = car['direction'].iloc[0]

	xlast = car['x'].iloc[-1]	
	xfirst = car['x'].iloc[0]
	
	if (dir == 1) & (xlast < xmax):
		car = forward_predict(car,xmin,xmax,'xmax',maxFrame)
	if (dir == -1) & (xlast > xmin):
		car = forward_predict(car,xmin,xmax,'xmin', maxFrame) # tested
	if (dir == 1) & (xfirst > xmin):
		car = backward_predict(car,xmin,xmax,'xmin')
	if (dir == -1) & (xfirst < xmax):
		car = backward_predict(car,xmin,xmax,'xmax') # tested, missing start point
	return car
		

def forward_predict(car,xmin,xmax,target, maxFrame):
	'''
	stops at maxFrame
	'''
	# lasts
	framelast = car['Frame #'].values[-1]
	if framelast >= maxFrame:
		return car
	ylast = car['y'].values[-1]
	xlast = car['x'].values[-1]
	vlast = car['speed'].values[-1]
	if vlast < 1:
		return car
	thetalast = car['theta'].values[-1]
	
	w = car['width'].values[-1]
	l = car['length'].values[-1]
	dir = car['direction'].values[-1]
	dt = 1/30
	x = []
	
	if target=='xmax':
		while xlast < xmax: 
			xlast = xlast + dir*vlast*dt
			x.append(xlast)
	else:
		while xlast > xmin: # tested
			xlast = xlast + dir*vlast*dt
			x.append(xlast)
	
	x = np.array(x)
	y = np.ones(x.shape) * ylast
	theta = np.ones(x.shape) * thetalast
	v = np.ones(x.shape) * vlast
	tlast = car['Timestamp'].values[-1]
	timestamps = np.linspace(tlast+dt, tlast+dt+dt*len(x), len(x), endpoint=False)

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
	
	frames = np.arange(framelast+1,framelast+1+len(x))
	pos_frames = frames<=maxFrame
	car_ext = {'Frame #': frames[pos_frames],
				'x':x[pos_frames],
				'y':y[pos_frames],
				'bbr_x': xa[pos_frames],
				'bbr_y': ya[pos_frames],
				'fbr_x': xb[pos_frames],
				'fbr_y': yb[pos_frames],
				'fbl_x': xc[pos_frames],
				'fbl_y': yc[pos_frames],
				'bbl_x': xd[pos_frames], 
				'bbl_y': yd[pos_frames],
				'speed': vlast,
				'theta': thetalast,
				'width': w,
				'length':l,
				'ID': car['ID'].values[-1],
				'direction': dir,
				'acceleration': 0,
				'Timestamp': timestamps[pos_frames],
				'Generation method': 'Extended'
				}
	car_ext = pd.DataFrame.from_dict(car_ext)
	return pd.concat([car, car_ext], sort=False, axis=0)

def backward_predict(car,xmin,xmax,target):
	'''
	backward predict up until frame 0
	'''
	# first
	framefirst = car['Frame #'].values[0]
	if framefirst <= 1:
		return car
	yfirst = car['y'].values[0]
	xfirst = car['x'].values[0]
	vfirst = car['speed'].values[0]
	if vfirst < 1:
		return car
	thetafirst = car['theta'].values[0]
	w = car['width'].values[-1]
	l = car['length'].values[-1]
	dt = 1/30
	dir = car['direction'].values[-1]
	x = []
	
	if target=='xmax': # dir=-1
		while xfirst < xmax: 
			xfirst = xfirst - dir*vfirst*dt
			x.insert(0,xfirst)
	else:
		while xfirst > xmin: 
			xfirst = xfirst - dir*vfirst*dt
			x.insert(0,xfirst)
	
	x = np.array(x)
	y = np.ones(x.shape) * yfirst
	theta = np.ones(x.shape) * thetafirst
	v = np.ones(x.shape) * vfirst
	tfirst = car['Timestamp'].values[0]
	timestamps = np.linspace(tfirst-dt-dt*len(x), tfirst-dt, len(x), endpoint=False)

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
	
	frames = np.arange(framefirst-len(x),framefirst)
	pos_frames = frames>=0
	# discard frame# < 0
	car_ext = {'Frame #': frames[pos_frames],
				'x':x[pos_frames],
				'y':y[pos_frames],
				'bbr_x': xa[pos_frames],
				'bbr_y': ya[pos_frames],
				'fbr_x': xb[pos_frames],
				'fbr_y': yb[pos_frames],
				'fbl_x': xc[pos_frames],
				'fbl_y': yc[pos_frames],
				'bbl_x': xd[pos_frames], 
				'bbl_y': yd[pos_frames],
				'speed': vfirst,
				'theta': thetafirst,
				'width': w,
				'length':l,
				'ID': car['ID'].values[0],
				'direction': dir,
				'acceleration': 0,
				'Timestamp': timestamps[pos_frames],
				'Generation method': 'Extended'
				}
	car_ext = pd.DataFrame.from_dict(car_ext)
	return pd.concat([car_ext, car], sort=False, axis=0)
	
def plot_track(D,length=15,width=1):
	fig, ax = plt.subplots(figsize=(length,width))

	for i in range(len(D)):
		coord = D[i,:]
		coord = np.reshape(coord,(-1,2)).tolist()
		coord.append(coord[0]) #repeat the first point to create a 'closed loop'
		xs, ys = zip(*coord) #lon, lat as x, y
		plt.plot(xs,ys,label='t=0' if i==0 else '',alpha=i/len(D),c='black')

		plt.scatter(D[i,2],D[i,3],color='black',alpha=i/len(D))
	ax = plt.gca()
	plt.xlabel('meter')
	plt.ylabel('meter')
	# plt.xlim([50,60])
	# plt.ylim([0,60])
	plt.legend()
	# ax.format_coord = lambda x,y: '%.6f, %.6f' % (x,y)
	plt.show() 
	return
	
def plot_track_df(df,length=15,width=1,show=True, ax=None, color='black'):
	D = np.array(df[['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']])
	if ax is None:
		fig, ax = plt.subplots(figsize=(length,width))
	
	for i in range(len(D)):
		coord = D[i,:]
		coord = np.reshape(coord,(-1,2)).tolist()
		coord.append(coord[0]) #repeat the first point to create a 'closed loop'
		xs, ys = zip(*coord) #lon, lat as x, y
		ax.plot(xs,ys,label='t=0' if i==0 else '',alpha=(i+1)/len(D),c=color)
		ax.scatter(D[i,2],D[i,3],color=color)#,alpha=i/len(D)
	ax.set_xlabel('meter')
	ax.set_ylabel('meter')
	if show:
		plt.show() 
		return
	else:
		return ax
	
from matplotlib import cm
def plot_track_df_camera(df,tform_path,length=15,width=1, camera='varies'):
	camera_list = ['p1c1','p1c2','p1c3','p1c4','p1c5','p1c6']
	color=cm.rainbow(np.linspace(0,1,len(camera_list)))
	camera_dict = dict(zip(camera_list,color))
	ID = df['ID'].iloc[0]
	fig, ax = plt.subplots(figsize=(length,width))
	if camera == 'varies':
		camera_group = df.groupby('camera')
		print('ID:',ID,'# frames:',len(df),'# cameras:',len(camera_group))
		for cameraID,cg in camera_group:
			Y = np.array(cg[['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']])
			c=camera_dict[cameraID]
			for i in range(len(Y)):
				coord = Y[i,:]
				coord = np.reshape(coord,(-1,2)).tolist()
				coord.append(coord[0]) #repeat the first point to create a 'closed loop'
				xs, ys = zip(*coord) #lon, lat as x, y	 
				plt.plot(xs,ys,c=c,label=cameraID if i == 0 else "")

			plt.scatter(Y[:,2],Y[:,3],color='black')
			ax = plt.gca()
			plt.xlabel('meter')
			plt.ylabel('meter')
			plt.legend()
			ax.format_coord = lambda x,y: '%.6f, %.6f' % (x,y) 
		
	else:
		c=camera_dict[camera]
		img_pts = np.array(df[['bbrx','bbry', 'fbrx','fbry','fblx','fbly','bblx', 'bbly']])
		Y = img_to_road_box(img_pts,tform_path,camera)
		for i in range(len(Y)):
			coord = Y[i,:]
			coord = np.reshape(coord,(-1,2)).tolist()
			coord.append(coord[0]) #repeat the first point to create a 'closed loop'
			xs, ys = zip(*coord) #lon, lat as x, y	 
			plt.plot(xs,ys,c=c,label=camera if i == 0 else "")

		plt.scatter(Y[:,2],Y[:,3],color='black')
		ax = plt.gca()
		plt.xlabel('meter')
		plt.ylabel('meter')
		ax.format_coord = lambda x,y: '%.6f, %.6f' % (x,y) 

		plt.legend()
		plt.show()
	
	return

def plot_track_compare(car,carre):
	ax = plot_track_df(car,show=False, color='red')
	plot_track_df(carre, show=True, ax=ax, color='blue')
	return
	
def overlap_score(car1, car2):
	'''
	apply after rectify, check the overlap between two cars
	'''
	end = min(car1['Frame #'].iloc[-1],car2['Frame #'].iloc[-1])
	start = max(car1['Frame #'].iloc[0],car2['Frame #'].iloc[0])
	
	if end <= start: # if no overlaps
		return 999
	car1 = car1.loc[(car1['Frame #'] >= start) & (car1['Frame #'] <= end)]
	car2 = car2.loc[(car2['Frame #'] >= start) & (car2['Frame #'] <= end)]

	pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
	Y1 = np.array(car1[pts])
	Y2 = np.array(car2[pts])
	return np.sum(np.linalg.norm(Y1-Y2,2, axis=1))/(len(Y1))
	


def get_id_rem(df,SCORE_THRESHOLD):
	'''
	get all the ID's to be removed due to overlapping
	'''
	groups = df.groupby('ID')
	groupList = list(groups.groups)
	nO = len(groupList)
	comb = itertools.combinations(groupList, 2)
	id_rem = [] # ID's to be removed

	for c1,c2 in comb:
		car1 = groups.get_group(c1)
		car2 = groups.get_group(c2)
	#	  score = overlap_score(car1, car2)
		score = IOU_score(car1,car2)
		IOU.append(score)
		if score > SCORE_THRESHOLD:
			# remove the shorter track
			if len(car1)>= len(car2):
				id_rem.append(c2)
			else:
				id_rem.append(c1)
	return id_rem



# delete repeated measurements per frame per object
del_repeat_meas = lambda x: x.head(1) if np.isnan(x['bbr_x'].values).all() else x[~np.isnan(x['bbr_x'].values)].head(1)

# x: df of measurements of same object ID at same frame, get average
def average_meas(x):
	mean = x.head(1)
	pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
	Y = np.array(x[pts])
	Y = np.nanmean(Y, axis=0)
	mean.loc[:,pts] = Y
	return mean
	
	
def del_repeat_meas_per_frame(framesnap):
	framesnap = framesnap.groupby('ID').apply(average_meas)
	return framesnap
	
def preprocess_multi_camera(data_path, tform_path):
	
	df = pd.DataFrame()
	for root,dirs,files in os.walk(str(data_path), topdown = True):
		for file in files:
			if file.endswith(".csv"):
				file_name = data_path.joinpath(file)
				camera_name = find_camera_name(file_name)
				print('*** Reading ',camera_name,'...')
				df1 = read_data(file_name,9)
				if 'Object ID' in df1:
					df1.rename(columns={"Object ID": "ID"})
				print(len(df1['ID'].unique()))
				print('Transform from image to road...')
				
				df1 = img_to_road(df1, tform_path, camera_name)
				print('Deleting unrelavent columns...')
				df1 = df1.drop(columns=['BBox xmin','BBox ymin','BBox xmax','BBox ymax','vel_x','vel_y','lat','lon'])
				df1 = df1.assign(camera=camera_name)
				df = pd.concat([df, df1])
		break
		
	# MUST SORT!!! OTHERWISE DIRECTION WILL BE WRONG
	df = df.sort_values(by=['Frame #','Timestamp']).reset_index(drop=True) 
	print('sorted.')
		
	print('Get x direction...')
	df = get_x_direction(df)
	print('Naive filter...')
	df = naive_filter_3D(df)
	
	print('Interpret missing timestamps...')
	frames = [min(df['Frame #']),max(df['Frame #'])]
	times = [min(df['Timestamp']),max(df['Timestamp'])]
	z = np.polyfit(frames,times, 1)
	p = np.poly1d(z)
	df['Timestamp'] = p(df['Frame #'])
	print('Get the longest continuous frame chuck...')
	df = df.groupby('ID').apply(findLongestSequence).reset_index(drop=True)
	return df
	
def preprocess_data_association(df):
	'''
	stitch objects based on their predicted trajectories
	associate objects based on obvious overlaps
	'''
	tqdm.pandas()
	
	# stitch based on prediction (weighted distance measure)
	print('Before DA: ', len(df['ID'].unique()), 'cars')
	parent = stitch_objects(df)
	df['ID'] = df['ID'].apply(lambda x: parent[x] if x in parent else x)
	print('After stitching: ', len(df['ID'].unique()), 'cars, checking for overlaps...')
	
	# associate based on overlaps (IOU measure)
	parent = associate_overlaps(df)
	df['ID'] = df['ID'].apply(lambda x: parent[x] if x in parent else x)  
	print('After assocating overlaps: ', len(df['ID'].unique()), 'cars')
	print('Get the longest continuous frame chunk...')
	df = df.groupby('ID').apply(findLongestSequence).reset_index(drop=True)
	df = applyParallel(df.groupby("Frame #"), del_repeat_meas_per_frame).reset_index(drop=True)
	# added the following noticed error in current version
	camera = find_camera_name(file)
	df = img_to_road(df, tform_path,camera)
	df = df.groupby("ID").apply(reorder_points).reset_index(drop=True)
	return df
	
def applyParallel(dfGrouped, func, args=None):
	with Pool(cpu_count()) as p:
		if args is None:
			ret_list = list(tqdm(p.imap(func, [group for name, group in dfGrouped]), total=len(dfGrouped.groups)))
		else:# if has extra arguments
			ret_list = list(tqdm(p.imap(partial(func, args=args), [group for name, group in dfGrouped]), total=len(dfGrouped.groups)))
	return pd.concat(ret_list)	
	
def get_camera_range(camera_id):
	'''
	return xmin, xmax, ymin, ymax (in meter)
	'''
	ymin = -5
	ymax = 45
	if camera_id=='p1c1':
		xmin = 0
		xmax = 130
	elif camera_id=='p1c2':
		xmin = 100
		xmax = 220
	elif camera_id=='p1c3':
		xmin = 180
		xmax = 260
	elif camera_id=='p1c4':
		xmin = 190
		xmax = 250
	elif camera_id=='p1c5':
		xmin = 210
		xmax = 300
	elif camera_id=='p1c6':
		xmin = 240
		xmax = 400
	elif camera_id=='p2c1':
		xmin = 182
		xmax = 234
	elif camera_id=='p2c2':
		xmin = 220
		xmax = 256
	elif camera_id=='p2c3':
		xmin = 840/3.281
		xmax = 960/3.281
	elif camera_id=='p2c4':
		xmin = 840/3.281
		xmax = 960/3.281
	elif camera_id=='p2c5':
		xmin = 920/3.281
		xmax = 1040/3.281
	elif camera_id=='p2c6':
		xmin = 970/3.281
		xmax = 1120/3.281
	elif camera_id=='p3c1':
		xmin = 1000/3.281
		xmax = 1250/3.281
	elif camera_id=='p3c2':
		xmin = 1200/3.281
		xmax = 1330/3.281
	elif camera_id=='p3c3':
		xmin = 1320/3.281
		xmax = 1440/3.281
	elif camera_id=='p3c4':
		xmin = 1320/3.281
		xmax = 1450/3.281
	elif camera_id=='p3c5':
		xmin = 1440/3.281
		xmax = 1720/3.281
	elif camera_id=='p3c6':
		xmin = 1680/3.281
		xmax = 1810/3.281
	elif camera_id=='all':
		xmin = 0
		xmax = 1810/3.281
	else:
		print('no camera ID in get_camera_range')
		return
	return xmin, xmax, ymin, ymax
	
def post_process(df):
	# print('remove overlapped cars...')
	# id_rem = get_id_rem(df, SCORE_THRESHOLD=0) # TODO: untested threshold
	# df = df.groupby(['ID']).filter(lambda x: (x['ID'].iloc[-1] not in id_rem))
	print('cap width at 2.59m...')
	df = df.groupby("ID").apply(width_filter).reset_index(drop=True)
	
	print('extending tracks to edges of the frame...')
	# del xmin, xmax
	# global xmin, xmax, maxFrame
	camera = df['camera'].iloc[0]
	xmin, xmax, ymin, ymax = get_camera_range(camera)
	maxFrame = max(df['Frame #'])
	print(xmin, xmax)
	args = (xmin, xmax, maxFrame)
	tqdm.pandas()
	# df = df.groupby('ID').apply(extend_prediction, args=args).reset_index(drop=True)
	df = applyParallel(df.groupby("ID"), extend_prediction, args=args).reset_index(drop=True)
	
	print('standardize format for plotter...')
	if ('lat' in df):
		df = df.drop(columns=['lat','lon'])
	# if ('frx' in df) and ('fbr_x' in df):
		# continue
	return df

def get_camera_x(x):
	x = x * 3.281 # convert to feet
	if x < 640:
		camera = 'p1c2'
	elif x < 770:
		camera = 'p1c3'
	elif x < 920:
		camera = 'p1c5'
	else:
		camera = 'p1c6'
	return camera
	
def road_to_img(df, tform_path):
	# TODO: to be tested
	if 'camera_post' not in df:
		df['camera_post'] = df[['x']].apply(lambda x: get_camera_x(x.item()), axis = 1)
	groups = df.groupby('camera_post')
	df_new = pd.DataFrame()
	for camera_id, group in groups:
		M = get_homography_matrix(camera_id, tform_path)
		Minv = np.linalg.inv(M)
		for pt in ['fbr','fbl','bbr','bbl']:
			road_pts = np.array(df[[pt+'_x', pt+'_y']]) * 3.281
			road_pts_1 = np.vstack((np.transpose(road_pts), np.ones((1,len(road_pts)))))
			img_pts_un = Minv.dot(road_pts_1)
			img_pts_1 = img_pts_un / img_pts_un[-1,:][np.newaxis, :]
			img_pts = np.transpose(img_pts_1[0:2,:])*2
			group[[pt+'x',pt+'y']] = pd.DataFrame(img_pts, index=df.index)
		df_new = pd.concat([df_new, group])
	return df_new
	
	
def width_filter(car):
# post processing only
# filter out width that's wider than 2.59m
# df is the df of each car
	
	w = car['width'].values[-1]
	l = car['length'].values[-1]
	notNan = np.count_nonzero(~np.isnan(np.sum(np.array(car[['bbr_x']]),axis=1)))
	if (w < 2.59) & (notNan == len(car)):
		return car
	theta = car['theta'].values
	dt=1/30
	
	
	if w > 2.59:
		w = 2.59
	# a = car['acceleration'].values
	# a = np.nan_to_num(a) # fill nan with zero
	v = car['speed'].values
	x = car['x'].values
	y = car['y'].values
	# dir = car['direction'].values[0]
	# for i in range(1,len(car)):
		# v[i] = v[i-1] + a[i-1] * dt
		# x[i] = x[i-1] + dir*v[i-1] * dt
		# y[i] = y[i-1]
	# compute new positions
	xa = x + w/2*sin(theta)
	ya = y - w/2*cos(theta)
	xb = xa + l*cos(theta)
	yb = ya + l*sin(theta)
	xc = xb - w*sin(theta)
	yc = yb + w*cos(theta)
	xd = xa - w*sin(theta)
	yd = ya + w*cos(theta)
	
	car['width'] = w
	car['x'] = x
	car['y'] = y
	car['bbr_x'] = xa
	car['bbr_y'] = ya
	car['fbr_x']= xb
	car['fbr_y']= yb
	car['fbl_x']= xc
	car['fbl_y']= yc
	car['bbl_x']= xd
	car['bbl_y']= yd
	car['speed']= v
	
	return car
	
def dashboard(cars):
	'''
	cars: list of dfs
	show acceleration/speed/theta/... of each car
	'''
	
	# acceleration
	# theta
	# omega

	fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(15,3))
	# x
	i = 0
	for car in cars:
		ax1.scatter(car['Frame #'].values, car['fbr_x'].values, label=i, s=1)
		i+=1
	ax1.legend()
	ax1.set_title('x (m)')
		
	# y positions
	i = 0
	for car in cars:
		ax2.scatter(car['Frame #'].values, car['fbr_y'].values, s=1)
		i+=1
	ax2.set_title('y (m)')
	
	# speed
	i = 0
	for car in cars:
		# notnan=~np.isnan(car.bbl_x.values)
		# ax3.scatter(car['Frame #'].values[notnan][:-1], -np.diff(car['bbl_x'].values[notnan])/(4/30))
		ax3.scatter(car['Frame #'].values, car['speed'].values, s=1)
		i+=1
	ax3.set_title('speed (m/s)')
	
	# acceleration
	i = 0
	for car in cars:
		ax4.scatter(car['Frame #'].values, car['acceleration'].values, s=1)
		i+=1
	ax4.set_title('acceleration (m/s2)')
	
	# theta
	i = 0
	for car in cars:
		ax5.scatter(car['Frame #'].values, car['theta'].values, s=1)
		i+=1
	ax5.set_title('theta (rad)')
	
	plt.show()
	return