# unused functions
def gps_to_road_df(df, A, B):
# TODO: not assume flat earth, using cross track distance
# TODO: consider traffic in the other direction

	
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
	AB = euclidean_distance(lat1,lon1,lat2,lon2)
	# convert to n-vector https://en.wikipedia.org/wiki/N-vector
	Y = np.empty(Ygps.shape)
	theta_12 = bearing(lat1, lon1, lat2, lon2)
	for i in range(int(Ygps.shape[1]/2)):
		pt_lats = Ygps[:,2*i]
		pt_lons = Ygps[:,2*i+1]
		#cross-track distance (y) - this one results in too small distance
		omega_13 = haversine_distance(lat1, lon1, pt_lats, pt_lons)/R 
		theta_13 = bearing(lat1, lon1, pt_lats, pt_lons)
		cross_track = arcsin(sin(omega_13)*sin(theta_13-theta_12))*R
		#along-track distance (x)
		along_track = np.arccos(cos(omega_13)/cos(cross_track/R))*R
		Y[:,2*i] = along_track
		Y[:,2*i+1] = np.absolute(cross_track)
	return Y
	
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

def assign_lane(df, startpts, endpts):
	pts = np.array(df[['lat','lon']])
	laneID = []
	for i in range(pts.shape[0]):
		dists = lineseg_dists(pts[i], startpts, endpts)
		laneID.append(np.argmin(dists))
	df['lane'] = laneID
	return df

def naive_filter_3D(df):
	groups = df.groupby('ID')

	# filter out direction==0
	df = groups.filter(lambda x: x['direction'].values[0] != 0)
	new_df = pd.DataFrame()
	pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
	pts_gps = ['bbrlat','bbrlon', 'fbrlat','fbrlon','fbllat','fbllon','bbllat', 'bbllon']
	
	for ID, g in groups:
		if (len(g)<1):
			print('length less than 1')
			continue
		Y = np.array(g[pts])
		Ygps = np.array(g[pts_gps])
		Y = Y.astype("float")
		xsort = np.sort(Y[:,[0,2,4,6]])
		ysort = np.sort(Y[:,[1,3,5,7]])
		try:
			if g['direction'].values[0]== '+':
				for i in range(len(Y)):
					Y[i,:] = [xsort[i,0],ysort[i,0],xsort[i,2],ysort[i,1],
					xsort[i,3],ysort[i,2],xsort[i,1],ysort[i,3]]

			if g['direction'].values[0]== '-':
				for i in range(len(Y)):
					Y[i,:] = [xsort[i,2],ysort[i,2],xsort[i,0],ysort[i,3],
					xsort[i,1],ysort[i,0],xsort[i,3],ysort[i,1]]
		
		except np.any(xsort<0) or np.any(ysort<0):
			print('Negative x or y coord, please redefine reference point A and B')
			sys.exit(1)
		
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
		
		for i in range(len(pts)):
			# g[pts[i]]=Y[:,i]
			# g[pts_gps[i]]=Ygps[:,i]
			g.loc[:,pts[i]] = Y[:,i]
			g.loc[:,pts_gps[i]] = Ygps[:,i]
		new_df = pd.concat([new_df, g])
	return new_df
