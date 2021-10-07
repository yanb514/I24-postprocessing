# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:30:57 2021

@author: wangy79
"""
from bs4 import BeautifulSoup
from IPython.display import IFrame
import numpy as np
import gmplot 
import matplotlib.pyplot as plt
import utils

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
	
def plot_track(D,length=15,width=1):
	fig, ax = plt.subplots(figsize=(length,width))

	for i in range(len(D)):
		coord = D[i,:]
		coord = np.reshape(coord,(-1,2)).tolist()
		coord.append(coord[0]) #repeat the first point to create a 'closed loop'
		xs, ys = zip(*coord) #lon, lat as x, y
		plt.plot(xs,ys,label=('t=0' if i==0 else ''),c='black')#alpha=i/len(D)
		plt.scatter(D[i,2],D[i,3],color='black')
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
		ax.plot(xs,ys,label='t=0' if i==0 else '',c=color)
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
		Y = utils.img_to_road_box(img_pts,tform_path,camera)
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
	ax = plot_track_df(carre, show=False, ax=ax, color='blue')
	return