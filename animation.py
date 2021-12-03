import csv
import numpy as np
import cv2
import itertools

def plot_vehicle_csv(
        csv_file0,
        csv_file1,
        params
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
    type_colors = {"rec": (0,255,0),
                   "raw": (0,0,255)}
    
    def read_frame_data(csvfile, x_offset=None):

        y_offset = 30
        with open(csvfile,"r") as f:
            all_frame_data = {}
            read = csv.reader(f)
            HEADERS = True
            # for row in read:
            for row in itertools.islice(read, params["max_rows"]):
                #get rid of first row
                if HEADERS:
                    if len(row) > 0 and row[0] == "Frame #":
                        HEADERS = False
                    continue
                frame_idx = int(float(row[0]))
                if frame_idx < params["min_frame"] or frame_idx > params["max_frame"]:
                    frame_idx+=1
                    continue
                
                id = int(float(row[2]))
                outlier = row[10]=='outlier'
                
                # c. store projected footprint coords if they exist
                if row[27] != '':
                    footprint = np.array(row[27:35]).astype(float).reshape(4,2)
                    rec_bbox = np.array([footprint[:,0],footprint[:,1]]).transpose()*params["rs"]
                    # rec_bbox[:,0] -=x_offset
                    # rec_bbox[:,1] +=y_offset
                else:
                    rec_bbox = np.zeros((4,2))
                if frame_idx in all_frame_data.keys():
                    all_frame_data[frame_idx].append([id,rec_bbox, outlier])
                else:
                    all_frame_data[frame_idx] = [[id,rec_bbox, outlier]]
        
        # calculate offsets
        allx = [box[1][0,0] for frame in all_frame_data.keys() if (params["min_frame"]<frame<params["min_frame"]+10) for box in all_frame_data[frame]]
        allx = [x for x in allx if x!=0]
        ret = False
        if x_offset is None:
            x_offset = -(min(allx)-30*params["rs"])
            xmax = max(allx)/params["rs"]
            ret = True
        for frame_idx, frame in all_frame_data.items():
            frame1 = [[box[0],np.add(box[1], [x_offset, y_offset]),box[2]] for box in frame]
            all_frame_data[frame_idx] = frame1   

        if ret:
            return all_frame_data, x_offset, xmax
        else:
            return all_frame_data
        
    def draw_box(frame_data, input_frame, DRAW_BASE, color, put_text = False):

        for box in frame_data:
            # plot each box
            id  = box[0]
            bbox_lmcs   = box[1]
            outlier = box[2]
            # flip the y axis
            bbox_lmcs[:,1] = input_frame.shape[0]-bbox_lmcs[:,1]
            if put_text:
                thickness = 2
            else: thickness = 3
            for a in range(len(bbox_lmcs)):
                ab = bbox_lmcs[a]
                for b in range(len(bbox_lmcs)):
                    bb = bbox_lmcs[b]
                    if DRAW_BASE[a][b] == 1 or DRAW_BASE[b][a] == 1:
                        if outlier:
                            frame = cv2.line(input_frame,(int(ab[0]),int(ab[1])),(int(bb[0]),int(bb[1])),(45,68,10),thickness)
                        else:    
                            frame = cv2.line(input_frame,(int(ab[0]),int(ab[1])),(int(bb[0]),int(bb[1])),color,thickness)
            if put_text:
                label = str(id)
                left = bbox_lmcs[0,0]
                top	 = bbox_lmcs[0,1]
                frame = cv2.putText(frame,"{}".format(label),(int(left),int(top)),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),3)
                frame = cv2.putText(frame,"{}".format(label),(int(left),int(top)),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
                
        try:
            return frame
        except: # in case frame_data is empty, simply return the blank background      
            return input_frame


    all_frame_data0, x_offset, xmax0 = read_frame_data(csv_file0, x_offset=None)
    if csv_file1:
        all_frame_data1 = read_frame_data(csv_file1, x_offset=x_offset)
        put_text0 = False
    else:
        all_frame_data1 = {}
        put_text0 = True
        
    DRAW_BASE = [[0,1,1,1], #bfl
             [0,0,1,1], #bfr
             [0,0,0,1], #bbl
             [0,0,0,0]] #bbr
    
    frame_idx = 0
    
    while True:
        # print(frame_idx)
        if frame_idx < params["min_frame"]:
            frame_idx+=1
            continue
        if frame_idx > params["max_frame"]:
            break
        blank = 255*np.ones((50*params["rs"],int(xmax0)*params["rs"],3), np.uint8)
        if frame_idx in all_frame_data0.keys():   
            frame_data0 = all_frame_data0[frame_idx]
        else:
            frame_data0 = []
        if frame_idx in all_frame_data1.keys():
            frame_data1 = all_frame_data1[frame_idx]
        else:
            frame_data1 = []
        
        frame0 = draw_box(frame_data0, blank, DRAW_BASE,type_colors["raw"], put_text=put_text0) # raw detection
        frame = draw_box(frame_data1, frame0, DRAW_BASE,type_colors["rec"], put_text=True) # rectified
        
        y_offset = 50
        frame = cv2.putText(frame,"Frame {}".format(frame_idx),(10,y_offset),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
        y_offset += 30
        frame = cv2.putText(frame,"Raw detection",(10,y_offset),cv2.FONT_HERSHEY_PLAIN,2,type_colors["raw"],2)
        y_offset += 30
        frame = cv2.putText(frame,"Rectified footprint",(10,y_offset),cv2.FONT_HERSHEY_PLAIN,2,type_colors["rec"],2)
        
        
        if params["ds"]:
            frame = cv2.resize(frame,(1920,1080))
        cv2.imshow("frame",frame)
        if params["frame_rate"] == 0:
            key = cv2.waitKey(0)
        else:
            key = cv2.waitKey(int(1000/float(params["frame_rate"])))
        if key == ord('q') or frame_idx >params["max_frame"]:
            break
        if key == ord("p"):
            cv2.waitKey(0)

        frame_idx += 1
        
    cv2.destroyAllWindows()

    	

if __name__ == "__main__":
    
    data_path = r"E:\I24-postprocess\MC_tracking" 
    # raw_path = data_path+r"\MC_reinterpolated.csv"
    gt_path = r"E:\I24-postprocess\benchmark\TM_1000_GT.csv"
    raw_path = r"E:\I24-postprocess\benchmark\TM_1000_pollute_DA.csv"
    da_path = data_path+r"\DA\MC_tsmn.csv"
    rec_path = data_path+r"\rectified\MC_tsmn.csv"
    	
    params = {
        "min_frame": 1000, # frame index to start and end animation, you'll need to hand tune these numbers
        "max_frame": 2000,
        "show_LMCS" : True,
        "save" : False,
        "frame_rate": 10, # normal: 30fps, set 0 to advance by pressing any key
        "ds" : False, # downsampling
        "rs": 10, # pixel resampling rate
        "max_rows": 50000 # an estimate of the max rows of csv reader for large files, so that it reads only part of the data for plotting
        }
    plot_vehicle_csv(gt_path, raw_path, params) # plot both paths
    # plot_vehicle_csv(gt_path, None, params) # set the second arg to None to plot only one path
    