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