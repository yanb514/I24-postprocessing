"""
Script to add to an output 2D tracking CSV in I24 format by computing a 3D bounding box for each
vehicle fully within the frame. The resulting bounding boxes are appended to a new file

Created on Thu Apr 22 14:06:56 2021

@author: worklab
"""

import csv
import numpy as np
import multiprocessing as mp
import cv2
import queue
import time

from utils import fit_3D_boxes,get_avg_frame,find_vanishing_point,calc_diff,plot_vp,plot_3D_ordered
from axis_labeler import Axis_Labeler

import argparse


def annotate_3d_box(box_queue,results_queue,CONTINUE,vp):
    """
    box_queue - an mp.queue with each element [diff image, 2D bbox, approx direction of travel,obj_idx,frame_idx]
    results_queue - an mp.queue to which results are written
    CONTINUE - shared mp value that indicates whether each process should continue
    vps - a list of [vp1,vp2,vp3] where each vp is (vpx,vpy) in image coordinates
    
    Repeatedly checks the queue for a box to process, and if one exists dequeues it, processes it, and writes
    resulting box (or None in case of error) to the results_queue
    """
    CONTINUE_COPY = True
    
    while CONTINUE_COPY:
        
        with CONTINUE.get_lock(): 
            CONTINUE_COPY = CONTINUE.value 
        
        try:
            [diff,box, direction,obj_idx,frame_idx,frame] = box_queue.get(timeout = 0)
            
        except queue.Empty:
            continue
        
        # fit box
        box_3d = fit_3D_boxes(diff,box,vp[0],vp[1],vp[2],granularity = 1e-03,e_init = 3e-01,show = False, verbose = False,obj_travel = direction)
        
        result = [box_3d,obj_idx,frame_idx,diff,vp,frame]
        results_queue.put(result)
        
        
def process_boxes(sequence,label_file,downsample = 1,SHOW = True, timeout = 20,threshold = 30):
    downsample = 2

    # load or compute average frame
    
    try:
        name =  "config/" + sequence.split("/")[-1].split(".mp4")[0].split("_")[1] + "_avg.png"
        avg_frame = cv2.imread(name)
        if avg_frame is None:
            raise FileNotFoundError
    except:
        avg_frame = get_avg_frame(sequence,ds = downsample).astype(np.uint8)
        name = "config/" + sequence.split("/")[-1].split(".mp4")[0].split("_")[1] + "_avg.png"
        cv2.imwrite(name,avg_frame)
   
    
    
    
    # get axes annotations
    try:
        name = "config/" + sequence.split("/")[-1].split(".mp4")[0].split("_")[1] + "_axes.csv"
        labels = []
        with open(name,"r") as f:
            read = csv.reader(f)
            for row in read:
                if len(row) == 5:
                    row = [int(float(item)) for item in row]
                elif len(row) > 5:
                    row = [int(float(item)) for item in row[:5]] + [float(item) for item in row[5:]]
                labels.append(np.array(row))
        show_vp = False
                        
    except FileNotFoundError:
        labeler = Axis_Labeler(sequence,ds = downsample)
        labeler.run()
        labels = labeler.axes
        show_vp = True
    
    # get vanishing points
    if True:    
        lines1 = []
        lines2 = []
        lines3 = []
        for item in labels:
            if item[4] == 0:
                lines1.append(item)
            elif item[4] == 1:
                lines2.append(item)
            elif item[4] == 2:
                lines3.append(item)
        
        # get all axis labels for a particular axis orientation
        vp1 = find_vanishing_point(lines1)
        vp2 = find_vanishing_point(lines2)
        vp3 = find_vanishing_point(lines3)
        vps = [vp1,vp2,vp3]

    if show_vp:
        plot_vp(sequence,vp1 = vp1,vp2 = vp2,vp3 = vp3, ds  = downsample)
        
    # load csv file annotations into list
    box_labels = []
    with open(label_file,"r") as f:
        read = csv.reader(f)
        HEADERS = True
        for row in read:
            
            if not HEADERS:
                box_labels.append(row)
            if len(row) > 0:
                if HEADERS and row[0][0:5] == "Frame":
                    HEADERS = False # all header lines have been read at this point
        
    # load sequence with videoCapture object and get first frame
    frame_idx = 0
    cap  = cv2.VideoCapture(sequence)
    ret,frame = cap.read()
    if not ret:
        print("Could not open VideoCapture object")
    
    # downsample first frame
    if downsample != 1:
            frame = cv2.resize(frame,(frame.shape[1]//downsample,frame.shape[0]//downsample))
    diff = calc_diff(frame,avg_frame)


    # resize average frame
    if avg_frame is None:
        avg_frame = np.zeros(frame.shape).astype(np.uint8)
    avg_frame = cv2.resize(avg_frame,(frame.shape[1],frame.shape[0]))
    
    
    # mp shared variables
    
    box_queue = mp.Queue()
    results_queue = mp.Queue()
    CONTINUE = mp.Value("i")
    with CONTINUE.get_lock(): 
        CONTINUE.value =  1
    
    # start worker processes
    pids = []
    for n in range(mp.cpu_count() - 8):
        p = mp.Process(target=annotate_3d_box, args=(box_queue,results_queue,CONTINUE,vps))
        pids.append(p)
    for p in pids:
        p.start()
    
            
    # main loop - queue and collect
    all_results = [] 
    frame_results = {}
    box_count_per_frame = {}
    start_time = time.time()
    count = 0
    result = "None"
    errors = 0
    removed_boxes = 0
    print(len(box_labels))
    time_since_last_result = time.time()    
 
    try:
        while len(all_results) < len(box_labels) - removed_boxes: 
            
            if count <  len(box_labels):
            
                box = box_labels[count]
                # format box
                bbox = np.array(box[4:8]).astype(float) / downsample
                direction = np.array(box[8:10]).astype(float) / downsample
                obj_idx = int(box[2])
                labeled_frame_idx = int(box[0])
                
                if labeled_frame_idx not in box_count_per_frame.keys():
                    box_count_per_frame[labeled_frame_idx] = 1
                else:
                    box_count_per_frame[labeled_frame_idx] += 1
                
                # advance current frame if necessary
                if frame_idx != labeled_frame_idx:
                    if frame_idx > labeled_frame_idx:
                        print("\n Oh No! labels out of order!")
                        
                    while frame_idx < labeled_frame_idx:
                        ret = cap.grab()
                        frame_idx += 1
                    ret,frame = cap.retrieve()

                    if downsample != 1:
                        frame = cv2.resize(frame,(frame.shape[1]//downsample,frame.shape[0]//downsample))
                    diff = calc_diff(frame,avg_frame,threshold = threshold)
                
                # cv2.imshow("frame",diff)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # add box to box_queue (add extra dimension so it is a list of 1 box)
                
                # don't add truncated boxes
                trunc = -10
                if bbox[0] < trunc or bbox[1] < trunc or bbox[2] > frame.shape[1] - trunc or bbox[3] > frame.shape[0] - trunc:
                    removed_boxes += 1
                else:
                    if SHOW:
                        inp = [diff.copy(),[bbox],direction,obj_idx,frame_idx,frame.copy()]
                    else:
                        inp = [diff.copy(),[bbox],direction,obj_idx,frame_idx,None]
                    box_queue.put(inp)
                count += 1
                
                #test 
                #fit_3D_boxes(diff,[bbox],vps[0],vps[1],vps[2],granularity = 3e-01,e_init = 1e-01,show = True, verbose = False)
                # bbox = bbox.astype(int)
                # frame2  = cv2.rectangle(frame.copy(),(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0,0),3)
                # cv2.imshow("frame",frame)
                # cv2.waitKey(1)
                
                
            elif count == box_labels:
                cap.release()
                cv2.destroyAllWindows()
            
            bps = np.round(len(all_results) / (time.time() - start_time),3)
            print("\rFrame {}, {}/{} boxes queued, {}/{} 3D boxes collected ({} bps), Errors: {}".format(frame_idx,count,len(box_labels),len(all_results),len(box_labels)-removed_boxes,bps,errors),end = '\r', flush = True)  

            # get result if any new results are ready
            try:
                
                result = results_queue.get(timeout = 0)
                if type(result[0]) == str:
                    all_results.append([])
                    box_3d = []
                    errors += 1
                
                else:
                    all_results.append(result[0:2])    
                    box_3d = result[0][0]
                
                time_since_last_result = time.time()
                out_obj_idx = result[1]
                out_frame_idx = result[2]
                out_diff = result[3].copy()
                vp = result[4]
                out_frame = result[5]
            
                
                out_frame = out_diff.copy()
                
                # save result
                if out_frame_idx not in frame_results.keys():
                    frame_results[out_frame_idx] = [(out_obj_idx,box_3d)]
                else:
                    frame_results[out_frame_idx].append((out_obj_idx,box_3d))
                
                plot_3D_ordered(out_frame,box_3d,label = "Object {}".format(out_obj_idx))
            
                if SHOW:
                    cv2.imshow("3D estimated boxes",out_frame)
                    
                    cv2.waitKey(1)
                # fr = plot_3D_ordered(diff,box_3d[0])
                # cv2.imshow("3D Estimated Bboxes",fr)
                # cv2.waitKey(1)             
        
            
           
            
            except queue.Empty:
                
                if time.time() - time_since_last_result > timeout:
                    break
                continue
            
            
            
            
            
        
        print("\nFinished collecting processed boxes")
        cv2.destroyAllWindows()
        with CONTINUE.get_lock(): 
            CONTINUE.value =  0
        for p in pids:
            p.terminate()
            time.sleep(0.1)
            p.join()
        
        print("All worker processes terminated. Writing output.")
        write_csv_3D(label_file,frame_results,downsample = downsample)
        
    
    except KeyboardInterrupt:# as E:
        cap.release()
        #print("Caught Exception: ", E)
        for p in pids:
            p.terminate()
            p.join()
        print("All worker processes terminated.")
        raise KeyboardInterrupt
        
    print("Reached end of processing")
    
    
    
def write_csv_3D(label_file,frame_results,downsample = 1):
    # load csv file annotations into list
    output_rows = []
    with open(label_file,"r") as f:
        read = csv.reader(f)
        HEADERS = True
        for row in read:
            
            
            if HEADERS:
                if len(row) > 0 and row[0][0:5] == "Frame":
                    HEADERS = False # all header lines have been read at this point
                    new_labels = ["fbrx","fbry","fblx","fbly","bbrx","bbry","bblx","bbly","ftrx","ftry","ftlx","ftly","btrx","btry","btlx","btly"]
                    row = row + new_labels
        
            else:
                # see if there is a 3D bbox associated with that object and frame
                obj_idx = int(row[2])
                frame_idx = int(row[0])
                
                if frame_idx in frame_results.keys():
                    for oidx,box in frame_results[frame_idx]:
                        if oidx == obj_idx:
                            if len(box) == 8:
                                flat_box = [coord*downsample for point in box for coord in point]
                                row = row + flat_box
                            else:
                                row = row + [None for i in range(16)]
                else:
                     row = row + [None for i in range(16)]
                    
            output_rows.append(row)
    
    # write final output file
    outfile = label_file.split(".csv")[0] + "_3D.csv"
    with open(outfile, mode='w') as f:
        out = csv.writer(f, delimiter=',')
        out.writerows(output_rows)
        
    print("Wrote output rows")
 
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
            
            if len(row) == 29: # has 3D bbox
                try:
                    bbox = np.array(row[13:]).astype(float).astype(int).reshape(8,2) #* downsample
                except:
                    # if there was a previous box for this object, use it instead
                    try:
                        NOMATCH = True
                        prev = 1
                        while prev < 4 and NOMATCH:
                            for row in frame_labels[frame_idx -prev]: 
                                if int(row[2]) == obj_idx:
                                    bbox = np.array(row[13:]).astype(float).astype(int).reshape(8,2) * downsample
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
    
    
    
if __name__ == "__main__":    
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("sequence",help = "p_c_000__")
        parser.add_argument("--show",action = "store_true")
        parser.add_argument("--skip_fitting",action = "store_true")
        parser.add_argument("-framerate",help = "output video framerate", type = int,default = 10)
        parser.add_argument("-threshold",help = "diff calculation threshold", type = int,default = 30)
    
        args = parser.parse_args()
        sequence = args.sequence
        SHOW = args.show
        framerate = args.framerate
        threshold = args.threshold
        skip_fitting = args.skip_fitting
    
    except:
        skip_fitting = False
        sequence = "p2c4_00001"
        SHOW = True
        threshold = 30
        framerate = 10
        
    # define file paths
    vid_sequence = "/home/worklab/Data/cv/video/5_min_18_cam_October_2020/ingest_session_00005/recording/record_{}.mp4".format(sequence)
    labels = "/home/worklab/Documents/derek/i24-dataset-gen/output/track_corrected_unique/record_{}_track_outputs_corrected.csv".format(sequence)
    labels_3D = "/home/worklab/Documents/derek/i24-dataset-gen/output/track_corrected_unique/record_{}_track_outputs_corrected_3D.csv".format(sequence)

    if not skip_fitting:
        process_boxes(vid_sequence,labels,downsample = 2,SHOW = SHOW,threshold = threshold,timeout = 60)    

    plot_3D_csv(vid_sequence,labels_3D,framerate = framerate)

