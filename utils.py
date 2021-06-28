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
import csv
import warnings
import sys
warnings.simplefilter ('default')

global A,B
A = [36.004654, -86.609976] # south west side, so that x, y coords obey counterclockwise
B = [36.002114, -86.607129]

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

def find_camera_id(file_name):
	start = file_name.find('record_')+7
	end = file_name.find('_000', start)
	return file_name[start:end]
	
def nan_helper(y):
	n = len(y)
	nans = np.isnan(y)
	x = lambda z: z.nonzero()[0]
	z = np.polyfit(x(~nans), y[~nans], 1)
	p = np.poly1d(z)
	y[nans]=p(x(nans))
	return y
	
def nan_helper_orig(y):
	return np.isnan(y), lambda z: z.nonzero()[0]

	
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
	
# path compression
def find(parent, i):
	if (parent[i] != i):
		parent[i] = find(parent, parent[i])
	return parent[i]

def compress(parent, groupList):	
	for i in groupList:
		find(parent, i)
	return parent 


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
# TODO: finish this
	theta = calc_theta(Y,timestamps)
	thetadot
	tan_phi = thetadot*v/l
	return arctan(tan_phi)

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
	pts_gps = ['bbrlat','bbrlon', 'fbrlat','fbrlon','fbllat','fbllon','bbllat', 'bbllon']
	
	Y = np.array(df[pts])
	Ygps = np.array(df[pts_gps])
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
	Ygps[isnan,:] = np.nan
	
	# write into df
	df.loc[:,pts] = Y
	df.loc[:,pts_gps] = Ygps
	return df
	
def naive_filter_3D(df):

	# filter out direction==0
	df = df.groupby("ID").filter(lambda x: x['direction'].values[0] != 0)
	# reorder points
	df = df.groupby("ID").apply(reorder_points).reset_index(drop=True)
	# filter out-of-bound length and width
	df = df.groupby("ID").apply(filter_width_length).reset_index(drop=True)
	
	return df
	
		


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



def pt_to_line_dist_gps(lat1, lon1, lat2, lon2, lat3, lon3):
	# distance from point (lat3, lon3) to a line defined by p1 and p2
	toA,_,_ = euclidean_distance(lat1, lon1, lat3, lon3)
	toB,_,_ = euclidean_distance(lat2, lon2, lat3, lon3)
	AB,_,_ = euclidean_distance(lat1, lon1, lat2, lon2)
	s = (toA+toB+AB)/2
	area = (s*(s-toA)*(s-toB)*(s-AB)) ** 0.5
	min_distance = area*2/AB
	return min_distance
	
def bearing(lat1, lon1, lat2, lon2):
# TODO: check north bound direction
	AB,dx,dy = euclidean_distance(lat1, lon1, lat2, lon2)
	return np.pi-np.arctan2(np.abs(dx),np.abs(dy))
	
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


	
def gps_to_road_df(df):
# TODO: consider traffic in the other direction
# use trigonometry 
	lat1, lon1 = A
	lat2, lon2 = B
	Y_gps =	 np.array(df[['bbrlat','bbrlon','fbrlat','fbrlon','fbllat','fbllon','bbllat','bbllon']])
	Y = gps_to_road(Y_gps)
	# write Y to df
	i = 0
	for pt in ['bbr','fbr','fbl','bbl']:
		df[pt+'_x'] = Y[:,2*i]
		df[pt+'_y'] = Y[:,2*i+1]
		i = i+1
	return df
	
def gps_to_road(Ygps):
	# use equal-rectangle approximation
	R = 6371*1000 # in meter6378137
	lat1, lon1 = A
	lat2, lon2 = B
	AB,_,_ = euclidean_distance(lat1,lon1,lat2,lon2)
	# convert to n-vector https://en.wikipedia.org/wiki/N-vector
	Y = np.empty(Ygps.shape)
	
	# use euclidean_distance
	for i in range(int(Ygps.shape[1]/2)):
		pt_lats = Ygps[:,2*i]
		pt_lons = Ygps[:,2*i+1]
		AC,_,_ = euclidean_distance(lat1,lon1,pt_lats,pt_lons)
		# cross-track: toAB
		toAB = pt_to_line_dist_gps(lat1, lon1, lat2, lon2, pt_lats, pt_lons)
		#along-track distance (x)
		along_track = np.sqrt(AC**2-toAB**2)
		Y[:,2*i] = along_track
		Y[:,2*i+1] = toAB
	return Y
	
def road_to_gps(Y, A, B):
# TODO: make this bidirectional
# https://stackoverflow.com/questions/7222382/get-lat-long-given-current-point-distance-and-bearing
	R = 6371000
	lat1, lon1 = A
	lat2, lon2 = B
	Ygps = np.zeros(Y.shape)
	gamma_ab = bearing(lat1,lon1,lat2,lon2)
	gamma_dc = gamma_ab - np.pi/2
	lat1 = np.radians(lat1)
	lon1 = np.radians(lon1)
	for i in range(int(Y.shape[1]/2)):
		xs = Y[:,2*i]
		ys = Y[:,2*i+1]
		latD, lonD = destination_given_distance_bearing(lat1, lon1, xs, gamma_ab)
		latC, lonC = destination_given_distance_bearing(latD, lonD, ys, gamma_dc)
		Ygps[:,2*i] = degrees(latC)
		Ygps[:,2*i+1] = degrees(lonC)
	return Ygps
	
	
def destination_given_distance_bearing(lat1, lon1, d, bearing):
	'''
	find the destination lat and lng given distance and bearing from the start point
	https://www.movable-type.co.uk/scripts/latlong.html
	lat1, lon1: start point gps coordinates
	d: distance from the start point
	bearing: bearing from the start point
	'''
	R = 6371000
	lat2 = arcsin(sin(lat1)*cos(d/R)+cos(lat1)*sin(d/R)*cos(bearing))
	lon2 = lon1 + arctan2(sin(bearing)*sin(d/R)*cos(lat1), cos(d/R)-sin(lat1)*sin(lat2))
	return lat2, lon2

def calc_homography_matrix(camera_id, file_name):
	c = pd.read_csv(file_name)
	camera = c.loc[c['Camera'].str.lower()==camera_id.lower()]

	gps_pts = camera[['GPS Lat','GPS Long']].to_numpy(dtype ='float32')
	xy_pts = camera[['Camera X','Camera Y']].to_numpy(dtype ='float32')
	# transform from pixel coords to gps coords
	M = cv2.getPerspectiveTransform(xy_pts,gps_pts)
	return M

def img_to_gps(df, camera_id, file_name):
	# vectorized
	M = calc_homography_matrix(camera_id,file_name)
	for pt in ['fbr','fbl','bbr','bbl']:
		ps = np.array(df[[pt+'x', pt+'y']]) # get pixel coords
		ps1 = np.vstack((np.transpose(ps), np.ones((1,len(ps))))) # add ones to standardize
		pds = M.dot(ps1) # convert to gps unnormalized
		pds = pds / pds[-1,:][np.newaxis, :] # gps normalized s.t. last row is 1
		ptgps = np.transpose(pds[0:2,:]) # only use the first two rows
		df = pd.concat([df, pd.DataFrame(ptgps,columns=[pt+'lat', pt+'lon'])], axis=1)
	return df

def gps_to_img(df, camera_id, file_name):
	# vectorized
	M = calc_homography_matrix(camera_id, file_name)
	Minv = np.linalg.inv(M)
	for pt in ['fbr','fbl','bbr','bbl']:
		ptgps = np.array(df[[pt+'lat', pt+'lon']]) 
		pds = np.vstack((np.transpose(ptgps), np.ones((1,len(ptgps)))))
		pds = Minv.dot(pds)
		ps1 = pds / pds[-1,:][np.newaxis, :]
		ps = np.transpose(ps1[0:2,:])
		df.loc[:,[pt+'x',pt+'y']] = ps
	return df
	
def get_xy_minmax(df):
# for plotting
	Y = np.array(df[['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']])
	notNan = ~np.isnan(np.sum(Y,axis=-1))
	Yx = Y[:,[0,2,4,6]]
	Yy = Y[:,[1,3,5,7]]
	return Yx[notNan,:].min(),Yx[notNan,:].max(),Yy[notNan,:].min(),Yy[notNan,:].max()
	
def plot_track(D,length,width):
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
	plt.legend()
	ax.format_coord = lambda x,y: '%.6f, %.6f' % (x,y)
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
		