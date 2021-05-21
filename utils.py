import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import math
import os
from numpy import arctan2,random,sin,cos,degrees, arcsin, radians,arccos
from bs4 import BeautifulSoup
from IPython.display import IFrame
import gmplot 
from sklearn import linear_model
from sklearn.metrics import r2_score
import cv2


# read data
def read_data(file_name, skiprows = 0):	 
#	  path = pathlib.Path().absolute().joinpath('tracking_outputs',file_name)
	df = pd.read_csv(file_name, skiprows = skiprows)
	df = df.rename(columns={"GPS lat of bbox bottom center": "lat", "GPS long of bbox bottom center": "lon", 'Object ID':'ID'})
	# df = df.loc[df['Timestamp'] >= 0]
	return df

def read_new_data(file_name):
	df = pd.read_csv(file_name)
	return df
	
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
	
def haversine_distance(lat1, lon1, lat2, lon2):
	r = 6371
	phi1 = np.radians(lat1)
	phi2 = np.radians(lat2)
	delta_phi = np.radians(lat2 - lat1)
	delta_lambda = np.radians(lon2 - lon1)
	a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) *   np.sin(delta_lambda / 2)**2
	res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
	return res * 1000 # km to m

# path compression
def find(parent, i):
	if (parent[i] != i):
		parent[i] = find(parent, parent[i])
	return parent[i]

def compress(parent, groupList):	
	for i in groupList:
		find(parent, i)
	return parent 


# calculate average speed of an object
def calc_velocity(df):
	if (len(df)<=1):
		return # do nothing
#	  lat_dist = df.loc[df.index[-1],'lat']-df.loc[df.index[0],'lat']
#	  lon_dist = df.loc[df.index[-1],'lon']-df.loc[df.index[0],'lon']
#	  timestep = df.loc[df.index[-1],'Timestamp'] - df.loc[group.index[0],'Timestamp']	
	lat_dist = df.lat.values[-1] - df.lat.values[0]
	lon_dist = df.lon.values[-1] - df.lon.values[0]
	timestep = df.Timestamp.values[-1] - df.Timestamp.values[0]
	df['lat_vel'] = lat_dist/timestep
	df['lon_vel'] = lon_dist/timestep
	return df

# calculate average speed of an object (in m/s)
def calc_velocity_mps(df):
	if (len(df)<=1):
		return # do nothing 
	distance = haversine_distance(df.lat.values[0], df.lon.values[0],df.lat.values[-1],df.lon.values[-1])
	timestep = df.Timestamp.values[-1] - df.Timestamp.values[0]
	df['mps'] = distance/timestep
	return df
	
def calc_accel(positions, timestamps):
	dx = np.gradient(positions)
	dt = np.gradient(timestamps)
	v = dx/dt
	a = np.gradient(v)/dt
	return a
	
def calc_velx(positions, timestamps):
	dx = np.gradient(positions)
	dt = np.gradient(timestamps)
	return dx/dt

	
def calc_vel(Y, timestamps):
	cx = (Y[:,0]+Y[:,6])/2
	cy = (Y[:,1]+Y[:,7])/2
	vx = calc_velx(cx, timestamps)
	vy = calc_velx(cy, timestamps)
	v = np.sqrt(vx**2+vy**2)
	return vx,vy,v

def calc_positions(cx,cy,theta,w,l):
	# compute positions
	xa = cx + w/2*sin(theta)
	ya = cy - w/2*cos(theta)
	xb = xa + l*cos(theta)
	yb = ya + l*sin(theta)
	xc = xb - w*sin(theta)
	yc = yb + w*cos(theta)
	xd = xa - w*sin(theta)
	yd = ya + w*cos(theta)
	Yre = np.stack([xa,ya,xb,yb,xc,yc,xd,yd],axis=-1) 
	return Yre
	
def calc_theta(Y,timestamps):
	vx,vy,v = calc_vel(Y,timestamps)
	# theta0 = np.arccos(vx/v)
	# return theta0
	# to get negative angles
	return np.arctan(vy/vx)
	
def calc_steering(Y,timestamps):
# approximate because l is not the distance between axis
	theta = calc_theta(Y,timestamps)
	thetadot
	tan_phi = thetadot*v/l
	return arctan(tan_phi)
# calculate the distance traveled
def calc_distance(df, filename):
	startpts, endpts = get_lane_info(filename)
	if 'lane' not in df:
		df = assign_lane(df, startpts, endpts)
	distance = []
	for i in range(len(df)):
		start = startpts[df.lane.values[i]]
		distance.append(haversine_distance(df.lat.values[i], df.lon.values[i], start[0], start[1]))
	df['distance'] = distance
	return df

# metafile for the start and end points of each lane for each file
def get_lane_info(filename):
	if 'p2c4' in str(filename):
		startpts = np.array([[36.00348, -86.60806],
					 [36.00346, -86.60810],
					 [36.003441, -86.60813],
					 [36.003415, -86.60818],
					 [36.00282, -86.60768],
					 [36.00279, -86.60774]
					])

		endpts = np.array([[36.00295, -86.60749],
					 [36.00293, -86.60754],
					 [36.00291, -86.607575],
					 [36.002885, -86.6076],
					 [36.0033, -86.6082],
					 [36.00323, -86.6082]
					])
	elif 'p3c6' in str(filename):
		startpts = np.array([[36.001777, -86.606115],
					 [36.001765, -86.606154],
					 [36.001751, -86.606196],
					 [36.001738, -86.606235],
					 [36.000145, -86.604334],
					 [36.000121, -86.604366],
					 [36.000105, -86.604397],
					 [36.000084, -86.604429]
					])

		endpts = np.array([[36.000354, -86.604256],
					 [36.000319, -86.604287],
					 [36.000256, -86.604268],
					 [36.000224, -86.604283],
					 [36.001669, -86.606354],
					 [36.001666, -86.606400],
					 [36.001661, -86.606452],
					 [36.001646, -86.606495]
					])
		return startpts, endpts
	else:
		print('lane info not provided') #TODO
		return

# calculate average traveling direction
def calc_direction(df):
	if (len(df)<=1):
		return # do nothing
#	  lat_dist = df.loc[df.index[-1],'lat']-df.loc[df.index[0],'lat']
#	  lon_dist = df.loc[df.index[-1],'lon']-df.loc[df.index[0],'lon']
	lat_dist = df.lat.values[-1] - df.lat.values[0]
	lon_dist = df.lon.values[-1] - df.lon.values[0]
	df['direction'] = lon_dist/lat_dist
	return df

def calc_bearing(df):
	# https://towardsdatascience.com/calculating-the-bearing-between-two-geospatial-coordinates-66203f57e4b4
	alat = df.lat.values[0]
	blat = df.lat.values[-1]
	alon = df.lon.values[0]
	blon = df.lon.values[-1]

	dl = blon - alon
	X = cos(blat) * sin(dl)
	Y = cos(alat) * sin(blat) - sin(alat) * cos(blat) * cos(dl)

	df['bearing'] = degrees(arctan2(X,Y))
	return df

def get_bearing_bounds(df):
	# only two major directions. Select those two bearing ranges and extract their bounds (4 bin edges based on histogram)
	hist, bin_edges = np.histogram(df.bearing.values)
	top_two = hist.argsort()[-2:][::-1]
	top_two.sort()
	return [bin_edges[top_two[0]], bin_edges[top_two[0]+1], bin_edges[top_two[1]], bin_edges[top_two[1] + 1]]

def calc_dynamics_all(df, filename):
	groups = df.groupby('ID')
	groupList = list(groups.groups)
	df_new = pd.DataFrame()
	for key, group in groups:
		if (len(group) > 1):
			group = calc_bearing(group)
			group = calc_velocity_mps(group)
			group = calc_distance(group, filename)
			df_new = pd.concat([df_new, group])
	return df_new


def naive_filter(df):
	# select based on proper bearings
	b = get_bearing_bounds(df)
	df = df.loc[(df['bearing']>=b[0]) & (df['bearing']<=b[1]) | (df['bearing']>=b[2]) & (df['bearing']<=b[3])]

	groups = df.groupby('ID')
	df_new = pd.DataFrame()
	for key,group in groups:
		if (len(group) > 1):
			df_new = pd.concat([df_new, group])
	return df_new

def predict_n_steps(n, group, dt):
#	  lat_v_avg = np.mean(group.lat_vel.values)
#	  lon_v_avg = np.mean(group.lon_vel.values)
	lat_v_avg = group.lat_vel.values[0]
	lon_v_avg = group.lon_vel.values[0]
	last = group.loc[group.index[-1]]
	lat_pred = [last.lat]
	lon_pred = [last.lon]
	time = [last.Timestamp]
	
	for i in range(n):
		lat_pred.append(lat_pred[i] + dt*lat_v_avg)
		lon_pred.append(lon_pred[i] + dt*lon_v_avg)
		time.append(time[-1] + dt)
	return np.append(group.Timestamp.values, time[1:]), np.append(group.lat.values, lat_pred[1:]), np.append(group.lon.values, lon_pred[1:])

def predict_distance(n,group,dt):
	last = group.loc[group.index[-1]]
	distance = [last.distance]
	# mps = group.mps.values[0]
	mps = 30
	time = [last.Timestamp]
	for i in range(n):
		distance.append(distance[i] + dt*mps)
		time.append(time[-1] + dt)
	return time[1:], distance[1:]
	# return np.append(group.Timestamp.values, time[1:]), np.append(group.distance, distance[1:])


def overlap(group1, group2, n, dt):
	# check of the next n steps of group1 trajectry will overlap with any of group2's trajectory
	time, lat_pred, lon_pred = predict_n_steps(n, group1, dt)
	subgroup2 = group2.loc[(group2['Timestamp'] >= time[0]-dt/2) & (group2['Timestamp'] <= time[-1]+dt/2)]

	if len(subgroup2)<=1:
		return 999
	lat_rs = np.interp(subgroup2.Timestamp.values, time, lat_pred)
	lon_rs = np.interp(subgroup2.Timestamp.values, time, lon_pred)
	dist = []
	for i in range(len(subgroup2)):
		dist.append(haversine_distance(lat_rs[i], lon_rs[i], subgroup2.lat.values[i], subgroup2.lon.values[i]))
	dist = np.sum(np.absolute(dist))/len(subgroup2)
	return dist

def overlap_distance(group1, group2, n, dt):
	# check of the next n steps of group1 trajectry will overlap with any of group2's trajectory
	# return the MSE
	time, distance = predict_distance(n, group1, dt)
	subgroup2 = group2.loc[(group2['Timestamp'] >= time[0]-dt/2) & (group2['Timestamp'] <= time[-1]+dt/2)]

	if len(subgroup2)<=1:
		return 999
	dist_rs = np.interp(subgroup2.Timestamp.values, time, distance)
	# print('group1: {:-1} group2:{:-1}'.format(group1.ID.values[0], group2.ID.values[0]))
	# print(dist_rs)
	
	error = 0
	for i in range(len(subgroup2)):
		error += abs(subgroup2.distance.values[i]-dist_rs[i])
	mae = error/len(subgroup2)
	# print(mae)
	return mae


# calculate velocity and direction information
def calc_velocity_direction(df):
	groups = df.groupby('ID')
	groupList = list(groups.groups)
	df_new = pd.DataFrame()
	for key, group in groups:
		if (len(group) > 1):
			calc_velocity(group)
			calc_direction(group)
			df_new = pd.concat([df_new, group])
	return df_new


# stitch connected objects together
# make some location prediction of an object based on its past trajectory
# check if the predicted trajectory overlaps with the measurement of another object at the same time frame 
# if overlaps, combine the two objects

def find_parent(dfall, tm, tp, thresh):
	groups = dfall.groupby('ID')
	groupList = list(groups.groups)
	
	# initialize all the objects to disjoint sets: parent=itself (a dictoinary)
	# parent stores the first-appeared object ID that one object is connect with
	# e.g., parent[obj2] = obj1 means that obj2 is connected with obj1, and thus should be combined
	parent = {}
	for g in groupList:
		parent[g] = g

	# updated = 0
	for i in range(len(groupList)-1):
		a = groupList[i]
		ga = groups.get_group(a)
	#	  if parent[a] == a: # if this object is not connected with others
		neighbors = find_neighbors_lr(dfall, ga, tm, tp)
			
		for b in neighbors:
			gb = groups.get_group(b)
			# if (a==84):
			#	  print(b)
			#	  print(overlap_lr(ga,gb))
			#	  predb = lr.predict(gb.Timestamp.values.reshape(-1,1))
			#	  plt.plot(gb.Timestamp.values, predb)
			#	  plt.scatter(gb.Timestamp.values, gb.distance.values, label=str(b))
			#	  # print(np.sum(np.absolute(predb-gb.distance.values)/len(predb)))
			#	  print(np.absolute(predb.reshape(-1,1)-gb.distance.values.reshape(-1,1)))
			#	  plt.legend(fontsize = 10)
			if (overlap_lr(ga,gb) <= thresh):
			# if (overlap_distance(ga,gb, n, dt) <= 6):
				parent[b] = a
				# updated = updated + 1
		
		# plt.show()
	# path compression: the parent of any object should be the ID that appeared first 
	parent = compress(parent, groupList)
	# print('Modified the ID of {:-1} / {:-1} objects'.format(updated, len(dfall['ID'].unique())))	
	return parent


# find the neighbors (in the same time range and travel the same direction)
def find_neighbors(dfall, df, n, dt):
	time, distance = predict_distance(n, df, dt)
	subgroup = dfall.loc[(dfall['Timestamp'] >= time[0]-dt/2) & (dfall['Timestamp'] <= time[-1]+dt/2) & (dfall['bearing'] >= df['bearing'].values[0] - 30) & (dfall['bearing'] <= df['bearing'].values[0] + 30) &(dfall['lane'] <= 2) &(dfall['lane'] >= max(0,df['lane'].values[0] - 1))]	 
	return subgroup["ID"].unique()


def fit_lr(df):
	lr = linear_model.LinearRegression()
	X = df.Timestamp.values.reshape(-1,1)
	y = df.distance.values.reshape(-1,1)
	lr.fit(X, y)
	return lr

def overlap_lr(group1, group2): 
	lr = fit_lr(group1)
	pred_dist = lr.predict(group2.Timestamp.values.reshape(-1,1))
	# MAE fit
	mae = np.sum(np.absolute(pred_dist.reshape(-1,1)-group2.distance.values.reshape(-1,1)))/len(pred_dist)
	print(mae)
	return mae

	# R-squared
	# r2 = r2_score(group2.distance.values,pred_dist)
	# return r2

# find the neighbors (in the same time range and travel the same direction)
def find_neighbors_lr(dfall, df, tm, tp):
	st = df.Timestamp.values[0] + tm
	et = df.Timestamp.values[-1]+tp # predict (et-st) sec into the future
	subgroup = dfall.loc[(dfall['Timestamp'] > st) & (dfall['Timestamp'] < et) & (dfall['bearing'] >= df['bearing'].values[0] - 30) & (dfall['bearing'] <= df['bearing'].values[0] + 30) &(dfall['lane'] <= 2) &(dfall['lane'] >= max(0,df['lane'].values[0] - 1))]	  
	return subgroup["ID"].unique()

# change objects'ID to be the same with their parents
def assignID(df, parent):
	
	groups = df.groupby('ID')
	groupList = list(groups.groups)
	
	new_df = pd.DataFrame()

	for g in groupList:
		p = parent[g]
		group = groups.get_group(g)
		if (g != p): 
			par = groups.get_group(p)
			group = group.assign(ID=par.loc[par.index[0],'ID'])
		new_df = pd.concat([new_df, group])
		
	return new_df

def stitch(file_name, n, dt):
	
	print('Reading '+str(file_name))
	df = read_data(file_name)

	# calculate and add velocity and direction information
	print('Calculating velocity and bearing ...')
	df = calc_dynamics_all(df)
	
	print('Naive filtering')
	df = naive_filter(df)

	print('Finding parent ...')
	parent = find_parent(df, n, dt)
	
	print('Assigning IDs ...')
	new_df = assignID(df, parent)
	
	print('Original algorithm counts {:-1} unique cars'.format(len(df['ID'].unique().tolist())))
	print('After stitching counts {:-1} unique cars'.format(len(new_df['ID'].unique().tolist())))
	
	return new_df



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
def draw_map_box(Y, latcenter, loncenter, nO, lats, lngs):
	
	map_name = "test.html"
	gmap = gmplot.GoogleMapPlotter(latcenter, loncenter, nO) 

	# get the bottom 4 points gps coords
	# Y = np.array(df[['bbrlat','bbrlon','fbrlat','fbrlon','fbllat','fbllon','bbllat','bbllon']])
	for i in range(len(Y)):
		coord = Y[i,:]
		coord = np.reshape(coord,(-1,2)).tolist()
		coord.append(coord[0]) #repeat the first point to create a 'closed loop'
		coord_tuple = [tuple(pt) for pt in coord]
		rectangle = zip(*coord_tuple) #create lists of x and y values
		gmap.polygon(*rectangle)	
	
	gmap.scatter(lats, lngs, color='red', size=1, marker=True)
	gmap.scatter(Y[:,2], Y[:,3],color='red', size=0.1, marker=False)
	gmap.draw(map_name)

	insertapikey(map_name)
	return jupyter_display(map_name)



def lineseg_dists(p, a, b):
	"""Cartesian distance from point to line segment

	Edited to support arguments as series, from:
	https://stackoverflow.com/a/54442561/11208892

	Args:
		- p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
		- a: np.array of shape (x, 2), start points
		- b: np.array of shape (x, 2), end points
	"""
	# normalized tangent vectors
	d_ba = b - a
	d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1])
						   .reshape(-1, 1)))

	# signed parallel distance components
	# rowwise dot products of 2D vectors
	s = np.multiply(a - p, d).sum(axis=1)
	t = np.multiply(p - b, d).sum(axis=1)

	# clamped parallel distance
	h = np.maximum.reduce([s, t, np.zeros(len(s))])

	# perpendicular distance component
	# rowwise cross products of 2D vectors	
	d_pa = p - a
	c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

	return np.hypot(h, c)

# calculate the distance from p3 to a line defined by p1 and p2
def pt_to_line_dist(p1,p2,p3):
	d = np.abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1))
	return d

def pt_to_line_dist_gps(lat1, lon1, lat2, lon2, lat3, lon3):
	# distance from point (lat3, lon3) to a line defined by p1 and p2
	# use cross track distance
	# R = 6371*1000 # in meter6378137
	# omega_13 = haversine_distance(lat1, lon1, lat3, lon3)/R 
	# theta_13 = bearing(lat1, lon1, lat3, lon3)
	# theta_12 = bearing(lat1, lon1, lat2, lon2)
	# min_distance = arcsin(sin(omega_13)*sin(theta_13-theta_12))*R
	
	# use trigonometry
	toA = haversine_distance(lat1, lon1, lat3, lon3)
	toB = haversine_distance(lat2, lon2, lat3, lon3)
	AB = haversine_distance(lat1, lon1, lat2, lon2)
	s = (toA+toB+AB)/2
	area = (s*(s-toA)*(s-toB)*(s-AB)) ** 0.5
	min_distance = area*2/AB
	return min_distance
	
def bearing(lat1, lon1, lat2, lon2):
# https://www.movable-type.co.uk/scripts/latlong.html
	dl = lon1 - lon2
	y = sin(dl) * cos(lat2)
	x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dl)
	return arctan2(y,x)
	 
def calc_y(df, A, B):
	# calculate the distance from each point to line AB
	# pt in lat and lng, calc the distance from pt to line AB
	lat1, lon1 = A
	lat2, lon2 = B
	for pt in ['fbr','fbl','bbr','bbl']:
		pt_lats = np.array(df[[pt+'lat']])
		pt_lons = np.array(df[[pt+'lon']])
		toAB = pt_to_line_dist_gps(lat1, lon1, lat2, lon2, pt_lats, pt_lons)
		df[pt+'_y'] = toAB
	return df
	
def get_x_direction(df):
	groups = df.groupby('ID')
	groupList = list(groups.groups)
	df_new = pd.DataFrame()
	for ID, group in groups:
		bbrx = group['bbr_x'].values
		if (len(bbrx[~np.isnan(bbrx)])<1):
			group = group.assign(direction='0')
			continue
		if (bbrx[~np.isnan(bbrx)][-1]-bbrx[~np.isnan(bbrx)][0]>0):
			# group['direction'] = '+'
			group = group.assign(direction='+')
		elif (bbrx[~np.isnan(bbrx)][-1]-bbrx[~np.isnan(bbrx)][0]<0):
			# group['direction'] = '-'
			group = group.assign(direction='-')
		else:
			group = group.assign(direction='0')
		df_new = pd.concat([df_new, group])
	return df_new
	
def calc_xy(df, A, B):
# TODO: not assume flat earth, using cross track distance
# use trigonometry 
	df = calc_y(df, A, B)
	lat1, lon1 = A
	for pt in ['fbr','fbl','bbr','bbl']:
		pt_lats = np.array(df[[pt+'lat']])
		pt_lons = np.array(df[[pt+'lon']])
		toA = haversine_distance(lat1, lon1, pt_lats, pt_lons)
		toAB = np.array(df[[pt+'_y']])
		df[pt+'_x'] = np.sqrt(toA**2-toAB**2)
	return df
	
	# use cross-track and along-track distance
	# R = 6371*1000 # in meter6378137
	# lat1, lon1 = A
	# lat2, lon2 = B
	#convert to n-vector https://en.wikipedia.org/wiki/N-vector
	# nA = np.array([cos(radians(lat1))*cos(radians(lon1)), cos(radians(lat1))*sin(radians(lon1)), sin(radians(lat1))]).T
	# nB = np.array([cos(radians(lat2))*cos(radians(lon2)), cos(radians(lat2))*sin(radians(lon2)), sin(radians(lat2))]).T
	# print(nA.shape)
	# c = np.cross(nA, nB)
	# c = c/np.linalg.norm(c)
	
	# theta_12 = bearing(lat1, lon1, lat2, lon2)
	# for pt in ['fbr','fbl','bbr','bbl']:
		# pt_lats = np.array(df[[pt+'lat']])
		# pt_lons = np.array(df[[pt+'lon']])
		##cross-track distance (y) - this one results in too small distance
		# omega_13 = haversine_distance(lat1, lon1, pt_lats, pt_lons)/R 
		# theta_13 = bearing(lat1, lon1, pt_lats, pt_lons)
		# cross_track = arcsin(sin(omega_13)*sin(theta_13-theta_12))*R
		##along-track distance (x)
		# along_track = np.arccos(cos(omega_13)/cos(cross_track/R))*R
		# df[pt+'_y'] = np.absolute(cross_track)
		# df[pt+'_x'] = along_track
	# return df
	
def gps_to_road(Ygps,A,B):
	# use cross-track and along-track distance
	R = 6371*1000 # in meter6378137
	lat1, lon1 = A
	lat2, lon2 = B
	# convert to n-vector https://en.wikipedia.org/wiki/N-vector
	
	theta_12 = bearing(lat1, lon1, lat2, lon2)
	for i in range(int(Ygps.shape[1]/2)):
		pt_lats = Y[:,2*i]
		pt_lons = Y[:,2*i+1]
		#cross-track distance (y) - this one results in too small distance
		omega_13 = haversine_distance(lat1, lon1, pt_lats, pt_lons)/R 
		theta_13 = bearing(lat1, lon1, pt_lats, pt_lons)
		cross_track = arcsin(sin(omega_13)*sin(theta_13-theta_12))*R
		#along-track distance (x)
		along_track = np.arccos(cos(omega_13)/cos(cross_track/R))*R
		# df[pt+'_y'] = np.absolute(cross_track)
		# df[pt+'_x'] = along_track
		Y[:,2*i] = along_track
		Y[:,2*i+1] = np.absolute(cross_track)
	return Y
	
def road_to_gps(Y, A, B):
# TODO: make this bidirectional
# https://stackoverflow.com/questions/7222382/get-lat-long-given-current-point-distance-and-bearing
	R = 6371
	lat1, lon1 = A
	lat2, lon2 = B
	Ygps = np.zeros(Y.shape)
	gamma_ab = bearing(lat1,lon1,lat2,lon2)
	lat1 = np.radians(lat1)
	lon1 = np.radians(lon1)
	for i in range(int(Y.shape[1]/2)):
		xs = Y[:,2*i]/1000
		ys = Y[:,2*i+1]/1000
		AC = np.sqrt(xs**2+ys**2)
		CAB = np.absolute(np.arctan(ys/xs))
		gamma_ac = gamma_ab-CAB
		lat3 = arcsin(sin(lat1)*cos(AC/R) + cos(lat1)*sin(AC/R)*cos(gamma_ac))
		lon3 = lon1 + arctan2(sin(gamma_ac)*sin(AC/R)*cos(lat1), cos(AC/R)-sin(lat1)*sin(lat3))
		Ygps[:,2*i] = degrees(lat3)
		Ygps[:,2*i+1] = degrees(lon3)
	return Ygps
		
def assign_lane(df, startpts, endpts):
	pts = np.array(df[['lat','lon']])
	laneID = []
	for i in range(pts.shape[0]):
		dists = lineseg_dists(pts[i], startpts, endpts)
		laneID.append(np.argmin(dists))
	df['lane'] = laneID
	return df


def calc_homography_matrix(camera_id, file_name):
	c = pd.read_csv(file_name)
	camera = c.loc[c['Camera']==camera_id]
	gps_pts = camera[['GPS Lat','GPS Long']].to_numpy(dtype ='float32')
	xy_pts = camera[['Camera X','Camera Y']].to_numpy(dtype ='float32')
	# transform from pixel coords to gps coords
	M = cv2.getPerspectiveTransform(xy_pts,gps_pts)
	return M

def calc_rr_coords(df, camera_id, file_name):
	# vectorized
	M = calc_homography_matrix(camera_id,file_name)
	for pt in ['fbr','fbl','bbr','bbl']:
		ps = np.array(df[[pt+'x', pt+'y']])
		ps1 = np.vstack((np.transpose(ps), np.ones((1,len(ps)))))
		pds = M.dot(ps1)
		pds = pds / pds[-1,:][np.newaxis, :]
		ptgps = np.transpose(pds[0:2,:])
		df = pd.concat([df, pd.DataFrame(ptgps,columns=[pt+'lat', pt+'lon'])], axis=1)
	return df

def calc_timestamp(df, fps):
	df['Timestamp'] = df['Frame #']/fps
	return df