import csv
import numpy as np
import multiprocessing as mp
import cv2
import queue
import time
import utils


def plot_rectified_objects(sequence,csv_file,frame_rate = 15):
	
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
	camera_id = utils.find_camera_name(sequence)
	if camera_id[0] != "p":
		camera_id = sequence.split("/")[-1].split("_")[1]
		if camera_id[0] != "p":
			print("Check sequence naming, cannot find camera id")
			return
	relevant_camera = camera_id
	
	# store each item by frame_idx
	all_frame_data = {}
	
	# store estimated heights per object idx
	obj_heights = {}
	obj_cls = {}
	
	frame_labels = {}
	with open(csv_file,"r") as f:
		read = csv.reader(f)
		HEADERS = True
		for row in read:
			
			#get rid of first row
			if HEADERS:
				HEADERS = False
				continue
				
			camera = row[39]
			if camera != relevant_camera:
				continue
			
			frame_idx = int(row[1])
			id = int(row[3])
			cls = row[4]
		
			if cls != '' and id not in obj_cls.keys():
				obj_cls[id] = cls
		
			if row[7] != '': # there was an initial 3D bbox prediction
				if id not in obj_heights.keys():
					try:
						est_height = (float(row[15]) + float(row[17]) + float(row[19]) + float(row[21])) - (float(row[7]) + float(row[9]) + float(row[11]) + float(row[13])) 
						obj_heights[id] = est_height

					except:
						pass
					
			bbox_re = np.array(row[40:48]).astype(float).reshape(4,2)
			# if (bbox_re < 0).any():
				# print('bbox out of range: ', frame_idx, id, len(bbox_re))
				# bbox_re = []
			if frame_idx in all_frame_data.keys():
				all_frame_data[frame_idx].append([id,bbox_re,camera])
			else:
				all_frame_data[frame_idx] = [[id,bbox_re,camera]]
			
	# All data gathered from CSV
			
	# outfile = label_file.split("_track_outputs_3D.csv")[0] + "_3D.mp4"
	# out = cv2.VideoWriter(outfile,cv2.VideoWriter_fourcc(*'mp4v'), framerate, (3840,2160))
	
	cap= cv2.VideoCapture(sequence)
	ret,frame = cap.read()
	frame_idx = 0
	
	while ret:		  
		
		if frame_idx in all_frame_data.keys():
			for box in all_frame_data[frame_idx]:
				
				# plot each box
				id = box[0]
				camera = box[2]
				bbox = box[1]
				bbox_top = bbox.copy()
				
				try:
					cls = obj_cls[id]
				except:
					cls = "sedan"
				
				# generate top points from footprint and height estimate
				if False and id in obj_heights.keys():	  
					bbox_top[:,1] += obj_heights[id]
				else:
					bbox_top[:,1] += 0
				
				bbox = np.concatenate((bbox,bbox_top),axis = 0)
				
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
	
				color = class_colors[classes[cls]]
		
				for a in range(len(bbox)):
					ab = bbox[a]
					for b in range(a,len(bbox)):
						bb = bbox[b]
						if DRAW[a][b] == 1:
							frame = cv2.line(frame,(int(ab[0]),int(ab[1])),(int(bb[0]),int(bb[1])),color,1)
						if DRAW_BASE[a][b] == 1:
							frame = cv2.line(frame,(int(ab[0]),int(ab[1])),(int(bb[0]),int(bb[1])),color,2)
				
				label = "{} {} (from {})".format(cls,id,camera)
				left = min([point[0] for point in bbox])
				top	 = min([point[1] for point in bbox])
				frame = cv2.putText(frame,"{}".format(label),(int(left),int(top - 10)),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),3)
				frame = cv2.putText(frame,"{}".format(label),(int(left),int(top - 10)),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)

		  
		frame = cv2.resize(frame,(1920,1080))
		
		cv2.imshow("frame",frame)
		key = cv2.waitKey(int(1000/float(frame_rate)))
		if key == ord('q') or frame_idx > 1800:
			break
		
		# get next frame
		ret,frame = cap.read()
		frame_idx += 1	   
		
	cv2.destroyAllWindows()
	cap.release()
	


# if __name__ == "__main__":
	# csv_file = "/home/worklab/Data/dataset_alpha/rectified_all_img_re.csv"
	# sequence = "/home/worklab/Data/cv/video/ground_truth_video_06162021/trimmed/p1c5_00000.mp4"
	
	# plot_rectified_objects(sequence,csv_file,frame_rate = 10.0)