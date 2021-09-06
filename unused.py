# unused functions

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
		# along-track distance (x)
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
	
def create_synth_data(n):
	timestamps =  np.linspace(0,n/30,n)
	theta = np.zeros(n)
	theta = np.random.normal(0, .02, theta.shape) + theta
	w = np.ones(n)*2 + np.random.normal(0, .2, n) 
	l = np.ones(n)*4 + np.random.normal(0, .4, n) 
	x = np.linspace(0,n,n) # assume 30fps and 30m/s, then 1 frame = 1m
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


def obj(X, Y1,N,dt,notNan, lam1,lam2,lam3,lam4,lam5):
	"""The cost function
		X = [j,alpha,a0,v0,x0,y0,theta0,w,l]^T
		penalize omega, a, jerk, theta and correction
		slow and not so accurate
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
	
	
	v = np.array(v)
	theta = np.ones(v.shape) * thetalast
	# v = np.ones(x.shape) * vlast
	tlast = car['Timestamp'].values[-1]
	timestamps = np.linspace(tlast+dt, tlast+dt+dt*len(v), len(v), endpoint=False)

	# compute positions
	Yre,x,y,a = generate(w,l,xlast+dt*vlast*cos(theta[0]),ylast+dt*vlast*sin(theta[0]),theta,v,outputall=True)
	
	frames = np.arange(framelast+1,framelast+1+len(x))
	pos_frames = frames<=maxFrame
	pts = ['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
	car_ext = {'Frame #': frames[pos_frames],
				'x':x[pos_frames],
				'y':y[pos_frames],
				'bbr_x': 0,
				'bbr_y': 0,
				'fbr_x': 0,
				'fbr_y': 0,
				'fbl_x': 0,
				'fbl_y': 0,
				'bbl_x': 0,
				'bbl_y': 0,
				'speed': v,
				'theta': theta,
				'width': w,
				'length':l,
				'ID': car['ID'].values[-1],
				'direction': dir,
				'acceleration': a,
				'Timestamp': timestamps[pos_frames],
				'Generation method': 'Extended'
				}
	car_ext = pd.DataFrame.from_dict(car_ext)
	car_ext[pts] = Yre[pos_frames,:]
	return pd.concat([car, car_ext], sort=False, axis=0)	
	