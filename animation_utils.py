from matplotlib import cm
import numpy as np
import math
	
# animation!!
# Pacakge Imports
from utils import *
import importlib
import utils
importlib.reload(utils)
import os.path
from os import path
import pandas as pd
import mplcursors
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from datetime import datetime
import cv2
import os
import multiprocessing
from functools import partial
from itertools import repeat
import random

def getCarColor(speed, maxSpeed, carID) :
	# based on speed
	# if(carID == 316120) : return 'black'
	# elif(carID == 344120) : return 'red'
	# elif(carID == 399120) : return 'white'
	
	# p1c1
	# if carID==3986: return 'blue'
	# elif carID==4024: return 'black'
	# elif carID==4152: return 'white'
	# elif carID==4498: return 'lightsteelblue'
	# elif carID==4834: return 'ghostwhite'
	# elif carID==5030: return 'white'
	# elif carID==5298: return 'black'
	# elif carID==5485: return 'black'
	# elif carID==5638: return 'white'
	# elif carID==5704: return 'ghostwhite'
	# elif carID==5828: return 'greenyellow'

	# p1c2
	# if carID==1244: return 'blue'
	# elif carID==1279: return 'black'
	# elif carID==1333: return 'white'
	# elif carID==1375: return 'lightsteelblue'
	# elif carID==1442: return 'ghostwhite'
	# elif carID==1536: return 'white'
	# elif carID==1578: return 'black'
	# elif carID==1608: return 'black'
	# elif carID==1674: return 'white'
	# elif carID==1714: return 'ghostwhite'
	# elif carID==1782: return 'greenyellow'
	
	# p1c3
	# if carID==769: return 'blue'
	# elif carID==800: return 'black'
	# elif carID==865: return 'white'
	# elif carID==894: return 'lightsteelblue'
	# elif carID==913: return 'ghostwhite'
	# elif carID==960: return 'white'
	# elif carID==1034: return 'black'
	# elif carID==1052: return 'black'
	# elif carID==1095: return 'white'
	# elif carID==1125: return 'ghostwhite'
	# elif carID==1215: return 'greenyellow'
	
	# p1c4
	# if carID==535: return 'blue'
	# elif carID==549: return 'black'
	# elif carID==589: return 'white'
	# elif carID==622: return 'lightsteelblue'
	# elif carID==635: return 'ghostwhite'
	# elif carID==656: return 'white'
	# elif carID==706: return 'black'
	# elif carID==713: return 'black'
	# elif carID==721: return 'white'
	# elif carID==725: return 'ghostwhite'
	# elif carID==759: return 'greenyellow'
		
	# p1c5
	# if carID==1054: return 'blue'
	# elif carID==1081: return 'black'
	# elif carID==1142: return 'white'
	# elif carID==1246: return 'lightsteelblue'
	# elif carID==1293: return 'ghostwhite'
	# elif carID==1365: return 'white'
	# # elif carID==: return 'black'
	# elif carID==1485: return 'black'
	# elif carID==1515: return 'white'
	# elif carID==1559: return 'ghostwhite'
	# elif carID==1594: return 'greenyellow'
	
	# # p1c6
	# if carID==1803: return 'blue'
	# elif carID==1874: return 'black'
	# elif carID==2008: return 'white'
	# elif carID==2212: return 'lightsteelblue'
	# elif carID==2438: return 'ghostwhite'
	# elif carID==2537: return 'white'
	# elif carID==2629: return 'black'
	# elif carID==2718: return 'black'
	# elif carID==2827: return 'white'
	# # elif carID==5704: return 'ghostwhite'
	# elif carID==2958: return 'greenyellow'
	
	# p1 all
	if carID==9999: return 'blue'
	elif carID==9998: return 'black'
	elif carID==9997: return 'white'
	elif carID==9996: return 'lightsteelblue'
	elif carID==9995: return 'ghostwhite'
	elif carID==9994: return 'white'
	elif carID==9993: return 'black'
	elif carID==9992: return 'black'
	elif carID==9991: return 'white'
	elif carID==9990: return 'ghostwhite'
	elif carID==9989: return 'greenyellow'
		
	
	else :
		return 'orange'
		# coolwarm = cm.get_cmap('coolwarm_r')
		# if speed > 34 :
			# return coolwarm(0.999)
		# else :
			# normVal = speed / 34.0
			# return coolwarm(normVal)
			
def get_cmap(n, name='hsv'):
	'''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
	RGB color; the keyword argument name must be a standard mpl colormap name.'''
	return plt.cm.get_cmap(name, n)
	
	
def restructCoord(frameSnap) :
	for i in range(len(frameSnap)) :
		if frameSnap[i,9] == 1 :  # If car is going left to right
			# Transform the coordinates so that bbr_x and so on are in sync
			# with cars going from right to left
			
			temp = frameSnap[i,0]
			frameSnap[i,0] = frameSnap[i,4]
			frameSnap[i,4] = temp
			
			temp = frameSnap[i,1]
			frameSnap[i,1] = frameSnap[i,5]
			frameSnap[i,5] = temp
			
			temp = frameSnap[i,2]
			frameSnap[i,2] = frameSnap[i,6]
			frameSnap[i,6] = temp
			
			temp = frameSnap[i,3]
			frameSnap[i,3] = frameSnap[i,7]
			frameSnap[i,7] = temp
			
		# Loop to change to feet
		for j in range(0,8) :
			frameSnap[i,j] *= 3.28084
			
		if math.isnan(frameSnap[i,11]) : frameSnap[i,11] = 0
			
def fillBetweenX(xs) :
	# Minor misalignments between the coordinates causes the fill function
	# to fill color in random spaces. Fixing the numbers to be exact.
	temp = list(xs)
	temp[1] = temp[2]
	temp[3] = temp[0]
	newxs = tuple(temp)
	
	return newxs

def fillBetweenY(ys) :
	temp = list(ys)
	temp[1] = temp[0]
	temp[2] = temp[3]
	newys = tuple(temp)
	
	return newys

def generate_image_per_frame(i,df, dim, skip_frame, color_dic, image_folder):
	xmin, xmax, ymin, ymax = dim
	ymin = 0
	ymax = 118
	img = plt.imread("highway_p1c3.jpg")
	if 'speed' not in df:
		df['speed'] = 30
	maxSpeed = np.amax(np.array(df[['speed']]))		   # Find the maximum speed of cars
	
	if (i%skip_frame==0):
		# Plot dimension setup
		fig, ax = plt.subplots(figsize=(9,6))
		ax.imshow(img, extent=[xmin,xmax,ymin, ymax])
		plt.xlim(xmin, xmax)
		plt.ylim(ymin, ymax)
		plt.xlabel('feet')
		plt.ylabel('feet')
		# extract the ID & road coordinates of the bottom 4 points of all vehicles at frame # i
		frameSnap = df.loc[(df['Frame #'] == i)]
		try:
			frame_time = frameSnap.Timestamp.iloc[0]
		except:# when a frameSnap is empty
			frame_time = 1000000000
		frameSnap = np.array(frameSnap[['bbr_x','bbr_y','fbr_x','fbr_y','fbl_x','fbl_y','bbl_x','bbl_y',
										'ID','direction','Timestamp', 'speed','Frame #']])
		restructCoord(frameSnap)
		# Looping thru every car in the frame
		for j in range(len(frameSnap)):
			carID = frameSnap[j,8]
			frameID = frameSnap[j,12]
			carSpeed = frameSnap[j,11]
			coord = frameSnap[j,0:8]	 # Road Coordinates of the Car
			coord = np.reshape(coord,(-1,2)).tolist()
			coord.append(coord[0])
			xs, ys = zip(*coord)
			xcoord = frameSnap[j,2]
			ycoord = frameSnap[j,3]
			# Displaying information above the car
			if xcoord < xmax and xcoord > xmin and ycoord < ymax :
				plt.text(xcoord, ycoord, str(int(carID)), fontsize=8)
#				  plt.text(xcoord, ycoord, str(int(carSpeed * 2.2369)) + ' mph', fontsize=8)	
			# Setting up car color
			# oneCarColor = getCarColor(carSpeed, maxSpeed, carID)
			oneCarColor = color_dic[carID] # random color
			# Plotting the car
			newxs = fillBetweenX(xs)
			newys = fillBetweenY(ys)
			ax.plot(newxs, newys, c = oneCarColor)
			ax.fill(newxs, newys, color = oneCarColor)
		try:
			# plt.title(datetime.fromtimestamp(frame_time).strftime("%H:%M:%S"), pad=20)
			plt.title('Frame '+str(int(frameID)), pad=20)
		except:
			pass
		fig.savefig(image_folder + '/' + format(i,"04d") + '.jpg', dpi=80)
		plt.close(fig)
	return
			
def generate_frames(df, dim, skip_frame, image_folder):
	
	# Divide all the data into frame numbers(1 ~ 2000). Then save each frame snapshot as a .jpg file
	# within a separate folder to later create an animation.
	# initialize car color 
	color_dic = {}
	groups = df.groupby('ID')
	
	nc = 10
	cmap = get_cmap(nc)
	
	for carID, group in groups:
		color_dic[carID] = cmap(random.randint(0,nc))
		
		
	maxFrameNum = int(max(df['Frame #']))	 # Find the maximum number of frame
	# maxFrameNum = 600
	# if maxFrameNum > 2100:
		# maxFrameNum = 2100
	minFrameNum = int(min(df['Frame #']))	 # Find the maximum number of frame
	
	print('Frame: ', minFrameNum, maxFrameNum)

	with multiprocessing.Pool() as pool:
		pool.map(partial(generate_image_per_frame, df=df, dim = dim, skip_frame = skip_frame, color_dic = color_dic, image_folder = image_folder), range(minFrameNum,maxFrameNum))
	return
	
def write_video(image_folder, video_name, fps):
	images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
	images.sort()
	frame = cv2.imread(os.path.join(image_folder, images[1]))
	height, width, layers = frame.shape
	video = cv2.VideoWriter(video_name, 0, fps, (width,height))
	for image in images:
		video.write(cv2.imread(os.path.join(image_folder, image)))
	cv2.destroyAllWindows()
	video.release()
	return
	
	