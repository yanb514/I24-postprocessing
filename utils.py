import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import pathlib
# import math
# import os
from numpy import arctan2,random,sin,cos,degrees, radians
from bs4 import BeautifulSoup
from IPython.display import IFrame
import gmplot 
# from sklearn import linear_model
# from sklearn.metrics import r2_score
import cv2
import csv
# import warnings
import sys
import re
import glob
from tqdm import tqdm
from utils_optimization import *
# warnings.simplefilter ('default')

# global A,B
# A = [36.004654, -86.609976] # south west side, so that x, y coords obey counterclockwise
# B = [36.002114, -86.607129]


# read data
def read_data(file_name, skiprows = 0, index_col = False):	 
#	  path = pathlib.Path().absolute().joinpath('tracking_outputs',file_name)
	df = pd.read_csv(file_name, skiprows = skiprows,error_bad_lines=False,index_col = index_col)
	df = df.rename(columns={"GPS lat of bbox bottom center": "lat", "GPS long of bbox bottom center": "lon", 'Object ID':'ID'})
	# df = df.loc[df['Timestamp'] >= 0]
	return df

def read_new_data(file_name):
	df = pd.read_csv(file_name)
	return df


def p_frame_time(df):
# polyfit frame wrt timestamps
	z = np.polyfit(df['Frame #'].iloc[[0,-1]].values,df['Timestamp'].iloc[[0,-1]].values, 1)
	p = np.poly1d(z)
	return p
	

	
def haversine_distance(lat1, lon1, lat2, lon2):
	r = 6371
	phi1 = np.radians(lat1)
	phi2 = np.radians(lat2)
	delta_phi = np.radians(lat2 - lat1)
	delta_lambda = np.radians(lon2 - lon1)
	try:
		a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
	except RuntimeWarning:
		print('error here')
	res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
	return res * 1000 # km to m

def euclidean_distance(lat1, lon1, lat2, lon2):
# https://math.stackexchange.com/questions/29157/how-do-i-convert-the-distance-between-two-lat-long-points-into-feet-meters
	r = 6371000
	lat1,lon1 = np.radians([lat1, lon1])
	lat2 = np.radians(lat2)
	lon2 = np.radians(lon2)
	theta = (lat1+lat2)/2
	dx = r*cos(theta)*(lon2-lon1)
	dy = r*(lat2-lat1)
	d = np.sqrt(dx**2+dy**2)
	# d = r*np.sqrt((lat2-lat1)**2+(cos(theta)**2*(lon2-lon1)**2))
	return d,dx,dy
	



def reorder_points(df):
	'''
		make sure points in the road-plane do not flip
	'''
	pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
	Y = np.array(df[pts])
	Y = Y.astype("float")
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
	pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
	# pts_gps = ['bbrlat','bbrlon', 'fbrlat','fbrlon','fbllat','fbllon','bbllat', 'bbllon']
	
	Y = np.array(df[pts])
	# Ygps = np.array(df[pts_gps])
	Y = Y.astype("float")
	# filter outlier based on width	
	w1 = np.abs(Y[:,3]-Y[:,5])
	w2 = np.abs(Y[:,1]-Y[:,7])
	outliers = np.logical_or(w1>5, w2>5)
	# print('width outlier:',np.count_nonzero(outliers))
	Y[outliers,:] = np.nan
	
	# filter outlier based on length
	l1 = np.abs(Y[:,0]-Y[:,2])
	m1 = np.nanmean(l1)
	s1 = np.nanstd(l1)
	outliers =	abs(l1 - m1) > 2 * s1
	# print('length outlier:',np.count_nonzero(outliers))
	Y[outliers,:] = np.nan
	
	isnan = np.isnan(np.sum(Y,axis=-1))
	# Ygps[isnan,:] = np.nan
	
	# write into df
	df.loc[:,pts] = Y
	# df.loc[:,pts_gps] = Ygps
	return df
	
def filter_short_track(df):

	Y1 = df['bbrx']
	N = len(Y1) 
	notNans = np.count_nonzero(~np.isnan(df['bbrx']))

	if (notNans <= 3) or (N <= 3):
		print('track too short: ', df['ID'].iloc[0])
		return False
	return True
	
def naive_filter_3D(df):
	# filter out direction==0
	df = df.groupby("ID").filter(lambda x: x['direction'].values[0] != 0)
	print('after direction=0 filter: ',len(df['ID'].unique()))
	# reorder points
	df = df.groupby("ID").apply(reorder_points).reset_index(drop=True)
	# filter out short tracks
	df = df.groupby("ID").filter(filter_short_track)
	print('after filtering short tracks: ',len(df['ID'].unique()))
	# filter out-of-bound length and width
	# df = df.groupby("ID").apply(filter_width_length).reset_index(drop=True)
	# print('filter width length:', len(df['ID'].unique()))
	return df

def findLongestSequence(car, k=0):
    A = np.diff(car['Frame #'].values)
    A[A!=1]=0
    
    # https://www.techiedelight.com/find-maximum-sequence-of-continuous-1s-can-formed-replacing-k-zeroes-ones/
    left = 0        # represents the current window's starting index
    count = 0       # stores the total number of zeros in the current window
    window = 0      # stores the maximum number of continuous 1's found
                    # so far (including `k` zeroes)
 
    leftIndex = 0   # stores the left index of maximum window found so far
 
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
#     print("The longest sequence has length", window, "from index",
#         leftIndex, "to", (leftIndex + window - 1))
    return car.iloc[leftIndex:leftIndex + window - 1,:]
	
def preprocess(file_path, tform_path):
	print('Reading data...')
	df = read_data(file_path,0)
	print(len(df['ID'].unique()))
	print('Transform from image to road...')
	camera_name = find_camera_name(file_path)
	df = img_to_road(df, tform_path, camera_name)
	print('Deleting unrelavent columns...')
	df = df.drop(columns=['BBox xmin','BBox ymin','BBox xmax','BBox ymax','vel_x','vel_y','lat','lon'])
	
	print('Get x direction...')
	df = get_x_direction(df)
	print('Naive filter...')
	df = naive_filter_3D(df)
	print('Get the longest continuous frame chuck...')
	df = df.groupby('ID').apply(findLongestSequence).reset_index(drop=True)
	print('Interpret missing timestamps...')
	z = np.polyfit(df['Frame #'].iloc[[0,-1]].values,df['Timestamp'].iloc[[0,-1]].values, 1)
	p = np.poly1d(z)
	df['Timestamp'] = p(df['Frame #'])
	return df

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
		df = df.assign(direction=0)
	else:
		sign = np.sign(bbrx[-1]-bbrx[0])
		df = df.assign(direction = sign)
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
	
def img_to_road(df,tform_path,camera_id):
	'''
	the images are downsampled
	'''
	M = get_homography_matrix(camera_id, tform_path)
	for pt in ['fbr','fbl','bbr','bbl']:
		img_pts = np.array(df[[pt+'x', pt+'y']]) # get pixel coords
		img_pts = img_pts/2 # downsample image to correctly correspond to M
		img_pts_1 = np.vstack((np.transpose(img_pts), np.ones((1,len(img_pts))))) # add ones to standardize
		road_pts_un = M.dot(img_pts_1) # convert to gps unnormalized
		road_pts_1 = road_pts_un / road_pts_un[-1,:][np.newaxis, :] # gps normalized s.t. last row is 1
		road_pts = np.transpose(road_pts_1[0:2,:])/3.281 # only use the first two rows, convert from ft to m
		df[[pt+'_x', pt+'_y']] = pd.DataFrame(road_pts, index=df.index)
	return df
	

	
def get_xy_minmax(df):
# for plotting
	Y = np.array(df[['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']])
	notNan = ~np.isnan(np.sum(Y,axis=-1))
	Yx = Y[:,[0,2,4,6]]
	# print(np.where(Yx[notNan,:]==Yx[notNan,:].min()))
	Yy = Y[:,[1,3,5,7]]
	return Yx[notNan,:].min(),Yx[notNan,:].max(),Yy[notNan,:].min(),Yy[notNan,:].max()
	
def extend_prediction(car, args):
	'''
	extend the dynamics of the vehicles that are still in view
	'''
	# print(args)
	# xmin = args[0]
	# xmax = args[1]
	# print(kwargs.itmes())
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
				'Timestamp': timestamps[pos_frames]
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
				'Timestamp': timestamps[pos_frames]
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
	
def plot_track_df(df,length=15,width=1):
	D = np.array(df[['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']])
	fig, ax = plt.subplots(figsize=(length,width))

	for i in range(len(D)):
		coord = D[i,:]
		coord = np.reshape(coord,(-1,2)).tolist()
		coord.append(coord[0]) #repeat the first point to create a 'closed loop'
		xs, ys = zip(*coord) #lon, lat as x, y
		plt.plot(xs,ys,label='t=0' if i==0 else '',alpha=i/len(D),c='black')

		plt.scatter(D[i,2],D[i,3],color='black')#,alpha=i/len(D)
	ax = plt.gca()
	plt.xlabel('meter')
	plt.ylabel('meter')
	# plt.xlim([50,60])
	# plt.ylim([0,60])
	# plt.legend()
	# ax.format_coord = lambda x,y: '%.6f, %.6f' % (x,y)
	plt.show() 
	return
	
def plot_3D_csv(sequence,label_file,framerate = 30):	
	colors = (np.random.rand(100,3)*255)
	downsample = 2
	
	outfile = label_file.split("_track_outputs_3D.csv")[0] + "_3D.mp4"
	out = cv2.VideoWriter(outfile,cv2.VideoWriter_fourcc(*'mp4v'), framerate, (3840,2160))
	cap= cv2.VideoCapture(sequence)
	
	frame_labels = {}
	with open(label_file,"r") as f:
		read = csv.reader(f)
		HEADERS = True
		for row in read:
			
			if not HEADERS:
				frame_idx = int(row[0])
				
				if frame_idx not in frame_labels.keys():
					frame_labels[frame_idx] = [row]
				else:
					frame_labels[frame_idx].append(row)
					
			if HEADERS and len(row) > 0:
				if row[0][0:5] == "Frame":
					HEADERS = False # all header lines have been read at this point
	
	ret,frame = cap.read()
	frame_idx = 0
	while ret:
		
		print("\rWriting frame {}".format(frame_idx),end = '\r', flush = True)	
		
		# get all boxes for current frame
		try:
			cur_frame_labels = frame_labels[frame_idx]
		except:
			cur_frame_labels = []
			
		for row in cur_frame_labels:
			obj_idx = int(row[2])
			obj_class = row[3]
			label = "{} {}".format(obj_class,obj_idx)
			color = colors[obj_idx%100]
			color = (0,0,255)
			bbox2d = np.array(row[4:8]).astype(float)  
			if len(row) == 46: # has 3D bbox, rectified data
				try:
					bbox = np.array(row[13:29]).astype(float).astype(int).reshape(8,2) #* downsample
				except:
					# if there was a previous box for this object, use it instead
					try:
						NOMATCH = True
						prev = 1
						while prev < 4 and NOMATCH:
							for row in frame_labels[frame_idx -prev]: 
								if int(row[2]) == obj_idx:
									bbox = np.array(row[13:29]).astype(float).astype(int).reshape(8,2) * downsample
									x_offset = (bbox2d[2] - bbox2d[0])/2.0 - (float(row[4]) + float(row[6]))/2.0 
									y_offset = (bbox2d[3] - bbox2d[1])/2.0 - (float(row[5]) + float(row[7]))/2.0 
									shift = np.zeros([8,2])
									shift[:,0] += x_offset
									shift[:,1] += y_offset
									bbox += shift
									NOMATCH = False
									break
								else:
									prev += 1
					except:
						bbox = []
				
				# get rid of bboxes that lie outside of a bbox factor x larger than 2d bbox
				factor = 3
				for point in bbox:
					if point[0] < bbox2d[0] - (bbox2d[2] - bbox2d[0])/factor:
						bbox = []
						break
					if point[1] < bbox2d[1] - (bbox2d[3] - bbox2d[1])/factor:
						bbox = []
						break
					if point[0] > bbox2d[2] + (bbox2d[2] - bbox2d[0])/factor:
						bbox = []
						break
					if point[1] > bbox2d[3] + (bbox2d[3] - bbox2d[1])/factor:
						bbox = []
						break
					
				
				frame = plot_3D_ordered(frame, bbox,color = color)
			
			
			color = (255,0,0)
			# frame = cv2.rectangle(frame,(int(bbox2d[0]),int(bbox2d[1])),(int(bbox2d[2]),int(bbox2d[3])),color,2)
			frame = cv2.putText(frame,"{}".format(label),(int(bbox2d[0]),int(bbox2d[1] - 10)),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),3)
			frame = cv2.putText(frame,"{}".format(label),(int(bbox2d[0]),int(bbox2d[1] - 10)),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),1)
		
		# lastly, add frame number in top left
		frame_label = "{}: frame {}".format(sequence.split("/")[-1],frame_idx)
		frame = cv2.putText(frame,frame_label,(0,50),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)

		out.write(frame)
		
		frame_show = cv2.resize(frame.copy(),(1920,1080))
		
		cv2.imshow("frame",frame_show)
		key = cv2.waitKey(1)
		if key == ord('q'):
			cv2.destroyAllWindows()
			break
			
		# get next frame
		ret,frame = cap.read()
		frame_idx += 1

		if frame_idx > 2050:
			break
		
	cap.release()
	out.release()
	
	print("Finished writing {}".format(outfile))
	
def plot_3D_ordered(frame,box,color = None,label = None):
	"""
	Plots 3D points as boxes, drawing only line segments that point towards vanishing points
	"""
	if len(box) == 0:
		return frame
	
	DRAW = [[0,1,1,0,1,0,0,0], #bfl
			[0,0,0,1,0,1,0,0], #bfr
			[0,0,0,1,0,0,1,1], #bbl
			[0,0,0,0,0,0,1,1], #bbr
			[0,0,0,0,0,1,1,0], #tfl
			[0,0,0,0,0,0,0,1], #tfr
			[0,0,0,0,0,0,0,1], #tbl
			[0,0,0,0,0,0,0,0]] #tbr
	
	DRAW_BASE = [[0,1,1,1,0,0,0,0], #bfl
			[0,0,1,1,0,0,0,0], #bfr
			[0,0,0,1,0,0,0,0], #bbl
			[0,0,0,0,0,0,0,0], #bbr
			[0,0,0,0,0,0,0,0], #tfl
			[0,0,0,0,0,0,0,0], #tfr
			[0,0,0,0,0,0,0,0], #tbl
			[0,0,0,0,0,0,0,0]] #tbr
	
	if color is None:
		color = (100,255,100)
		
	for a in range(len(box)):
		ab = box[a]
		for b in range(a,len(box)):
			bb = box[b]
			if DRAW_BASE[a][b] == 1:
				frame = cv2.line(frame,(int(ab[0]),int(ab[1])),(int(bb[0]),int(bb[1])),color,2)
			# if DRAW_BASE[a][b] == 1:
			#	  frame = cv2.line(frame,(int(ab[0]),int(ab[1])),(int(bb[0]),int(bb[1])),color,2)
	
	size = 4
	# color = (0,0,255)
	color = (0,0,0)
	frame = cv2.circle(frame,(int(box[0][0]),int(box[0][1])),size,color,-1)
	# color = (0,100,255)
	frame = cv2.circle(frame,(int(box[1][0]),int(box[1][1])),size,color,-1)
	# color = (0,175,255)
	frame = cv2.circle(frame,(int(box[2][0]),int(box[2][1])),size,color,-1)
	# color = (0,255,255)
	frame = cv2.circle(frame,(int(box[3][0]),int(box[3][1])),size,color,-1)
	# color = (255,0,0)
	# frame = cv2.circle(frame,(int(box[4][0]),int(box[4][1])),size,color,-1)
	# color = (255,100,0)
	# frame = cv2.circle(frame,(int(box[5][0]),int(box[5][1])),size,color,-1)
	# color = (255,175,0)
	# frame = cv2.circle(frame,(int(box[6][0]),int(box[6][1])),size,color,-1)
	# color = (255,255,0)
	# frame = cv2.circle(frame,(int(box[7][0]),int(box[7][1])),size,color,-1)
	
	if label is not None:
		left = min([point[0] for point in box])
		top	 = min([point[1] for point in box])
		frame = cv2.putText(frame,"{}".format(label),(int(left),int(top - 10)),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),3)
		frame = cv2.putText(frame,"{}".format(label),(int(left),int(top - 10)),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
	return frame	
	
# delete repeated measurmeents per frame per object
del_repeat_meas = lambda x: x.head(1) if np.isnan(x['bbr_x'].values).all() else x[~np.isnan(x['bbr_x'].values)].head(1)

def del_repeat_meas_per_frame(framesnap):
    framesnap = framesnap.groupby('ID').apply(del_repeat_meas)
    return framesnap
	
def preprocess_multi_camera(df):
	tqdm.pandas()
	# df_new = df.groupby('Frame #').progress_apply(del_repeat_meas_per_frame).reset_index(drop=True)
	df_new = applyParallel(df.groupby("Frame #"), del_repeat_meas_per_frame).reset_index(drop=True)
	return df_new

def post_process(df):
	print('cap width at 2.59m...')
	df = df.groupby("ID").apply(width_filter).reset_index(drop=True)
	print('extending tracks to edges of the frame...')
	# del xmin, xmax
	# global xmin, xmax, maxFrame
	xmin, xmax, ymin, ymax = get_xy_minmax(df)
	maxFrame = max(df['Frame #'])
	print(xmin, xmax)
	if xmin<0:
		xmin=0
	if xmax>600:
		xmax = 600
	args = (xmin, xmax, maxFrame)
	tqdm.pandas()
	# df = df.groupby('ID').apply(extend_prediction, args=args).reset_index(drop=True)
	df = applyParallel(df.groupby("ID"), extend_prediction, args=args).reset_index(drop=True)
	return df
	
def width_filter(car):
# post processing only
# filter out width that's wider than 2.59m
# df is the df of each car
	
	w = car['width'].values[-1]
	l = car['length'].values[-1]
	notNan = np.count_nonzero(~np.isnan(np.sum(np.array(car[['bbr_x']]),axis=1)))
	theta = car['theta'].values
	dt=1/30
	
	if (w < 2.59) & (notNan == len(car)):
		return car
	else:
		if w > 2.59:
			w = 2.59
		a = car['acceleration'].values
		a = np.nan_to_num(a) # fill nan with zero
		v = car['speed'].values
		x = car['x'].values
		y = car['y'].values
		dir = car['direction'].values[0]
		for i in range(1,len(car)):
			v[i] = v[i-1] + a[i-1] * dt
			x[i] = x[i-1] + dir*v[i-1] * dt
			y[i] = y[i-1]
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
	
	
	