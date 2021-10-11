
	
import csv
import numpy as np
import multiprocessing as mp
import cv2
import queue
import time
import argparse
from utils import *

def plot_vehicle_csv(
		sequence,
		csv_file,
		frame_rate = 10.0,
		show_2d = False,
		show_3d = False,
		show_LMCS = False,
		show_rectified = False,
		save = False,
		ds = False
		):
		
	
	class_colors = [
			(0,255,0),
			(255,0,0),
			(0,0,255),
			(255,255,0),
			(255,0,255),
			(0,255,255),
			(255,100,0),
			(255,50,0),
			(0,255,150),
			(0,255,100),
			(0,255,50)]
	
	classes = { "sedan":0,
					"midsize":1,
					"van":2,
					"pickup":3,
					"semi":4,
					"truck (other)":5,
					"truck": 5,
					"motorcycle":6,
					"trailer":7,
					0:"sedan",
					1:"midsize",
					2:"van",
					3:"pickup",
					4:"semi",
					5:"truck (other)",
					6:"motorcycle",
					7:"trailer",
					}
	
	# get the camera id - only these boxes will be parsed from the data file
	# camera_id = sequence.split("/")[-1].split("_")[0]
	camera_id = find_camera_name(sequence)
	if camera_id[0] != "p":
		camera_id = sequence.split("/")[-1].split("_")[1]
		if camera_id[0] != "p":
			print("Check sequence naming, cannot find camera id")
			return
	relevant_camera = camera_id
	
	# load LMCS -> im space homography matrix
	# transform_path = "./annotate/tform/{}_im_lmcs_transform_points.csv".format(relevant_camera)
	transform_path = "../tform/{}_im_lmcs_transform_points.csv".format(relevant_camera)
	# get transform from RWS -> ImS
	keep = []
	with open(transform_path,"r") as f:
		read = csv.reader(f)
		FIRST = True
		for row in read:
			if FIRST:
				FIRST = False
				continue
					
			if "im space"  in row[0]:
				break
			
			keep.append(row)
	pts = np.stack([[float(item) for item in row] for row in keep])
	im_pts = pts[:,:2] # * 2.0
	lmcs_pts = pts[:,2:]
	H,_ = cv2.findHomography(lmcs_pts,im_pts)
	print(H)
	
	# store each item by frame_idx
	all_frame_data = {}
	
	# store estimated heights and classes per object idx
	obj_heights = {}
	obj_cls = {}
	

	# loop through CSV file	   
	frame_labels = {}
	with open(csv_file,"r") as f:
		read = csv.reader(f)
		HEADERS = True
		for row in read:
			
			#get rid of first row
			if HEADERS:
				if len(row) > 0 and row[0] == "Frame #":
					HEADERS = False
				continue
				
			camera = row[36]
			# if camera != relevant_camera:
			#	  continue
			
			frame_idx = int(row[0])
			if frame_idx<1800:
				continue
			id = int(row[2])
			cls = row[3]
		
			if cls != '' and id not in obj_cls.keys():
				obj_cls[id] = cls
		
			interp = True if row[10] == "interp 3d" else False
		
			# a. store 2D bbox
			if row[4] != '' and camera == relevant_camera:
				bbox_2d = np.array(row[4:8]).astype(float)
			else:
				bbox_2d = np.zeros(4)
				
			# b. store 3D bbox if it exists
			if row[11] != '' and camera == relevant_camera: # there was an initial 3D bbox prediction
				if id not in obj_heights.keys():
					try:
						est_height = (float(row[15]) + float(row[17]) + float(row[19]) + float(row[21])) - (float(row[7]) + float(row[9]) + float(row[11]) + float(row[13])) 
						obj_heights[id] = est_height

					except:
						pass
				
				try:
					bbox_3d = np.array(row[11:27]).astype(float).reshape(8,2)
				except:
					bbox_3d = np.zeros([8,2])

			else:
				bbox_3d = np.zeros([8,2])
				
			# c. store projected footprint coords if they exist
			if row[27] != '':
				footprint = np.array(row[27:35]).astype(float).reshape(4,2) *3.281
				
				# reproject these coords with inverse homography
				
				lmcs_bbox = np.array([footprint[:,0],footprint[:,1],[1,1,1,1]])#.transpose()
				
				# H = H+np.random.normal(0,0.000001,(3,3))
				out = np.matmul(H,lmcs_bbox) # H might be ill-conditioned
				
				im_footprint = np.zeros([2,4])
				im_footprint[0,:] = out[0,:] / out[2,:]
				im_footprint[1,:] = out[1,:] / out[2,:]
				im_footprint = im_footprint.transpose()
			
			else:
				im_footprint = np.zeros([4,2])
			
			# d. store reprojected LMCS state as 3D box points if state exists
			if row[39] != '':
				# add image space box points
				x_center = float(row[39]) * 3.281
				y_center = float(row[40]) * 3.281
				theta	 = float(row[41])
				width	 = float(row[42])	 * 3.281
				length	 = float(row[43])	* 3.281
				
				fx = x_center + length * np.cos(theta) 
				ly = y_center + width/2 * np.cos(theta)
				bx = x_center
				ry = y_center - width/2 * np.cos(theta)	   
				
				lmcs_bbox = np.array([[fx,fx,bx,bx],[ry,ly,ry,ly],[1,1,1,1]])#.transpose()
				
				out = np.matmul(H,lmcs_bbox)
				re_footprint = np.zeros([2,4])
				re_footprint[0,:] = out[0,:] / out[2,:]
				re_footprint[1,:] = out[1,:] / out[2,:]
				re_footprint = re_footprint.transpose()
				
				im_top = re_footprint.copy()
				try:
					est_height = obj_heights[id]
				except KeyError:
					est_height = 50
					
				im_top[:,1] -= est_height
				rectified_bbox_3d = np.array(list(re_footprint.reshape(-1)) + list(im_top.reshape(-1)) )
				rectified_bbox_3d = rectified_bbox_3d.reshape(8,2)
			
			else:
				rectified_bbox_3d = np.zeros([8,2])
					
					
				
			if frame_idx in all_frame_data.keys():
				all_frame_data[frame_idx].append([id,bbox_2d,bbox_3d,im_footprint,rectified_bbox_3d,interp])
			else:
				all_frame_data[frame_idx] = [[id,bbox_2d,bbox_3d,im_footprint,rectified_bbox_3d,interp]]
								
			
	# All data gathered from CSV
	if save:	  
		outfile = "camera_{}_track_outputs_3D.mp4".format(camera)
		if ds:
			out = cv2.VideoWriter(outfile,cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (1920,1080))
		else:
			out = cv2.VideoWriter(outfile,cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (3840,2160))
	
	cap= cv2.VideoCapture(sequence)
	cap.set(1,2250)
	ret,frame = cap.read()
	
	frame_idx = 2250
	
	while ret:		  
			
		if frame_idx in all_frame_data.keys():
			for box in all_frame_data[frame_idx]:
				
				# plot each box
				id				= box[0]
				bbox_2d			= box[1]
				bbox_3d			= box[2]
				bbox_lmcs		= box[3]
				bbox_rectified	= box[4]
				interp			= box[5]
				try:
					cls = obj_cls[id]
				except:
					cls = "sedan"
				
				
				
				DRAW = [[0,1,1,0,1,0,0,0], #bfl
				[0,0,0,1,0,1,0,0], #bfr
				[0,0,0,1,0,0,1,1], #bbl
				[0,0,0,0,0,0,1,1], #bbr
				[0,0,0,0,0,1,1,0], #tfl
				[0,0,0,0,0,0,0,1], #tfr
				[0,0,0,0,0,0,0,1], #tbl
				[0,0,0,0,0,0,0,0]] #tbr
		
				DRAW_BASE = [[0,1,1,1], #bfl
							 [0,0,1,1], #bfr
							 [0,0,0,1], #bbl
							 [0,0,0,0]] #bbr
						
	
				color = class_colors[classes[cls]]
		
		
				color = (0,0,255)
				
				if show_2d:
					frame = cv2.rectangle(frame,(int(bbox_2d[0]),int(bbox_2d[1])),(int(bbox_2d[2]),int(bbox_2d[3])),color,1)
					
				color = (0,255,255)
				if interp:
					color = (0,100,255)
				if show_3d:
					for a in range(len(bbox_3d)):
						ab = bbox_3d[a]
						for b in range(a,len(bbox_3d)):
							bb = bbox_3d[b]
							if DRAW[a][b] == 1:
								frame = cv2.line(frame,(int(ab[0]),int(ab[1])),(int(bb[0]),int(bb[1])),color,1)
			   
				color = (0,255,0)			  
				if show_LMCS:
					for a in range(len(bbox_lmcs)):
						ab = bbox_lmcs[a]
						for b in range(len(bbox_lmcs)):
							bb = bbox_lmcs[b]	
							if DRAW_BASE[a][b] == 1 or DRAW_BASE[b][a] == 1:
								frame = cv2.line(frame,(int(ab[0]),int(ab[1])),(int(bb[0]),int(bb[1])),color,1)
				
				color = (255,0,0)
				if show_rectified:
					for a in range(len(bbox_rectified)):
						ab = bbox_rectified[a]
						for b in range(a,len(bbox_rectified)):
							bb = bbox_rectified[b]
							if DRAW[a][b] == 1:
								frame = cv2.line(frame,(int(ab[0]),int(ab[1])),(int(bb[0]),int(bb[1])),color,2)
								
				
				label = "{} {}".format(cls,id)
				left = bbox_2d[0]
				top	 = bbox_2d[1]
				frame = cv2.putText(frame,"{}".format(label),(int(left),int(top - 10)),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),3)
				frame = cv2.putText(frame,"{}".format(label),(int(left),int(top - 10)),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)

		  
		#frame = cv2.resize(frame,(1920,1080))
		y_offset = 50
		if show_2d:
			frame = cv2.putText(frame,"2D bbox",(10,y_offset),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),5)
			frame = cv2.putText(frame,"2D bbox",(10,y_offset),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),3)
			y_offset += 30
			
		if show_3d:
			frame = cv2.putText(frame,"Auto 3D bbox",(10,y_offset),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),5)
			frame = cv2.putText(frame,"Auto 3D bbox",(10,y_offset),cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),3)
			y_offset += 30
			frame = cv2.putText(frame,"Interpolated 3D",(10,y_offset),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),5)
			frame = cv2.putText(frame,"Interpolated 3D",(10,y_offset),cv2.FONT_HERSHEY_PLAIN,2,(0,100,255),3)
			y_offset += 30
			
		if show_LMCS:
			frame = cv2.putText(frame,"3D footprint double projected",(10,y_offset),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),5)
			frame = cv2.putText(frame,"3D footprint double projected",(10,y_offset),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),3)
			y_offset += 30
		if show_rectified:
			frame = cv2.putText(frame,"Rectified 3D bbox",(10,y_offset),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),5)	 
			frame = cv2.putText(frame,"Rectified 3D bbox",(10,y_offset),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)	 
			
		if ds:
			frame = cv2.resize(frame,(1920,1080))
	
		cv2.imshow("frame",frame)
		
		
		if frame_rate == 0:
			key = cv2.waitKey(0)
		else:
			key = cv2.waitKey(int(1000/float(frame_rate)))
		if key == ord('q') or frame_idx > 3600:
			break
		
		if save:
			out.write(frame)
		
		# get next frame
		ret,frame = cap.read()
		frame_idx += 1	   
		
		
		
	cv2.destroyAllWindows()
	cap.release()
	
	if save:
		out.release()
	

if __name__ == "__main__":
    
     #add argparse block here so we can optinally run from command line
     try:

        parser = argparse.ArgumentParser()
        parser.add_argument("video_path",help = "path to video sequence")
        parser.add_argument("csv_path",help = "path to csv label file")
        
        parser.add_argument("-fps",type = float, help = "Speed of video playback (0 pauses on each frame)", default = 10)
        parser.add_argument("--show_2d",action = "store_true")
        parser.add_argument("--show_3d",action = "store_true")
        parser.add_argument("--show_lmcs",action = "store_true")
        parser.add_argument("--show_rectified",action = "store_true")
        parser.add_argument("--save",action = "store_true")
        parser.add_argument("--ds",action = "store_true")


        args = parser.parse_args()
        sequence = args.video_path
        csv_file = args.csv_path
        show_2d = args.show_2d
        show_3d = args.show_3d
        show_LMCS = args.show_lmcs
        show_rectified = args.show_rectified
        save = args.save
        frame_rate = args.fps
        ds = args.ds

        
     except:
         print("No path specified, using default paths and settings instead")
         show_2d = False
         show_3d = True
         show_LMCS = True
         show_rectified = False
         save = False
         frame_rate = 30
         ds = False
        
         camera_name = "p1c4"
         sequence_idx = 0
         csv_file = r"E:\I24-postprocess\June_5min\rectified\{}_{}.csv".format(camera_name,sequence_idx)  
         # csv_file = r"E:\I24-postprocess\June_5min\Automatic 3D (uncorrected)\{}_{}_track_outputs_3D.csv".format(camera_name,sequence_idx)
         sequence = r"E:\I24-postprocess\June_5min\Raw Video\{}_{}.mp4".format(camera_name,sequence_idx)
         
         BOI = [1244,1279,1333,1375,1437,1536,1578,1608,1669,1714,1782]
    
     plot_vehicle_csv(sequence,csv_file,frame_rate = frame_rate,show_2d = show_2d,show_3d = show_3d,show_LMCS = show_LMCS,show_rectified = show_rectified, save = save,ds=ds)