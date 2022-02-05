import torch
import numpy as np
import cv2
import sys, os
import csv
from scipy.optimize import linear_sum_assignment

from homography import Homography, load_i24_csv



class MOT_Evaluator():
    
    def __init__(self,gt_path,pred_path,homography,params = None):
        """
        gt_path - string, path to i24 csv file for ground truth tracks
        pred_path - string, path to i24 csv for predicted tracks
        homography - Homography object containing relevant scene information
        params - dict of parameter values to change
        """
        
        self.match_iou = 0
        self.cutoff_frame = 10000
        self.sequence = None 
        
        self.gt_mode = "im" # must be im, space or state - controls which to treat as the ground truth
        
        # store homography
        self.hg = homography
        
        # data is stored as a dictionary of lists - each key corresponds to one frame
        # and each list item corresponds to one object
        # load ground truth data
        _,self.gt = load_i24_csv(gt_path)

        # load pred data
        _,self.pred = load_i24_csv(pred_path)
        
        
        if params is not None:
            if "match_iou" in params.keys():
                self.match_iou = params["match_iou"]
            if "cutoff_frame" in params.keys():
                self.cutoff_frame = params["cutoff_frame"]
            if "sequence" in params.keys():
                self.sequence = params["sequence"]
                
        
        # create dict for storing metrics
        n_classes = len(self.hg.class_heights.keys())
        class_confusion_matrix = np.zeros([n_classes,n_classes])
        self.m = {
            "FP":0,
            "FP edge-case":0,
            "FP @ 0.2":0,
            "FN @ 0.2":0,
            "FN":0,
            "TP":0,
            "pre_thresh_IOU":[],
            "match_IOU":[],
            "state_err":[],
            "im_bot_err":[],
            "im_top_err":[],
            "cls":class_confusion_matrix,
            "ids":{},
            "gt_ids":[],
            "pred_ids":[]
            }
        
        units = {}
        units["Match IOU"]           = ""
        units["Pre-threshold IOU"]   = ""
        units["Width precision"]     = "ft"
        units["Height precision"]    = "ft"
        units["Length precision"]    = "ft"
        units["Velocity precision"]  = "ft/s"
        units["X precision"]         = "ft"
        units["Y precision"]         = "ft"
        units["Bottom im precision"] = "px"
        units["Top im precision"]    = "px"
        self.units = units
        
        if self.sequence is not None:
            self.cap = cv2.VideoCapture(self.sequence)
            
    
    def iou(self,a,b):
        """
        Description
        -----------
        Calculates intersection over union for all sets of boxes in a and b
    
        Parameters
        ----------
        a : tensor of size [batch_size,4] 
            bounding boxes
        b : tensor of size [batch_size,4]
            bounding boxes.
    
        Returns
        -------
        iou - float between [0,1]
            average iou for a and b
        """
        
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        
        minx = max(a[0], b[0])
        maxx = min(a[2], b[2])
        miny = max(a[1], b[1])
        maxy = min(a[3], b[3])
        
        intersection = max(0, maxx-minx) * max(0,maxy-miny)
        union = area_a + area_b - intersection + 1e-06
        iou = intersection/union
        
        return iou
            
    def evaluate(self):
        
        frame_num = -1
        
        # for each frame:
        for f_idx in range(self.cutoff_frame):
            
            print("\rAggregating metrics for frame {}/{}".format(f_idx,self.cutoff_frame),end = "\r",flush = True)
            
            if self.sequence:
                _,im = self.cap.read()
            
            try:
                gt = self.gt[f_idx]
            except KeyError:
                if f_idx in self.pred.keys():
                    self.m["FP"] += len(self.pred[f_idx])
                    for item in self.pred[f_idx]:
                        id = int(item[2])
                        if id not in self.m["pred_ids"]:
                            self.m["pred_ids"].append(id)
                continue
            
            try:
                pred = self.pred[f_idx]
            except KeyError:
                if f_idx in self.gt.keys():
                    self.m["FN"] += len(self.gt[f_idx])
                    for item in self.gt[f_idx]:
                        id = int(item[2])
                        if id not in self.m["gt_ids"]:
                            self.m["gt_ids"].append(id)
                continue
                    
            # store ground truth as tensors
            gt_classes = []
            gt_ids = []
            gt_im = []
            gt_velocities = []
            for box in gt:
                gt_im.append(np.array(box[11:27]).astype(float))
                gt_ids.append(int(box[2]))
                gt_classes.append(box[3])
                vel = float(box[38]) if len(box[38]) > 0 else 0
                gt_velocities.append(vel)
                
            gt_im = torch.from_numpy(np.stack(gt_im)).reshape(-1,8,2)
            
            # two pass estimate of object heights
            heights = self.hg.guess_heights(gt_classes)
            gt_state = self.hg.im_to_state(gt_im,heights = heights)
            repro_boxes = self.hg.state_to_im(gt_state)
            refined_heights = self.hg.height_from_template(repro_boxes,heights,gt_im)
            
            # get other formulations for boxes
            gt_state = self.hg.im_to_state(gt_im,heights = refined_heights)
            gt_space = self.hg.state_to_space(gt_state)
            
            gt_velocities = torch.tensor(gt_velocities).float()
            gt_state = torch.cat((gt_state,gt_velocities.unsqueeze(1)),dim = 1)
            
            # store pred as tensors (we start from state)
            pred_classes = []
            pred_ids = []
            pred_state = []
            for box in pred:
                # x,y,l,w,h,direction,v
                # print(box)  
                pred_state.append(np.array([box[39],box[40],box[43],box[42],box[44],box[35],box[38]]).astype(float))
                pred_ids.append(int(box[2]))
                pred_classes.append(box[3])
            
            pred_state = torch.from_numpy(np.stack(pred_state)).reshape(-1,7).float()
            pred_space = self.hg.state_to_space(pred_state)
            pred_im = self.hg.state_to_im(pred_state)
            # convert meter to feet ><
            pred_state = pred_state * 3.281
            pred_space = pred_space * 3.281  
                
        
            # compute matches based on space location ious
            first = gt_space.clone()
            boxes_new = torch.zeros([first.shape[0],4])
            boxes_new[:,0] = torch.min(first[:,0:4,0],dim = 1)[0]
            boxes_new[:,2] = torch.max(first[:,0:4,0],dim = 1)[0]
            boxes_new[:,1] = torch.min(first[:,0:4,1],dim = 1)[0]
            boxes_new[:,3] = torch.max(first[:,0:4,1],dim = 1)[0]
            first = boxes_new
            
            second = pred_space.clone()
            boxes_new = torch.zeros([second.shape[0],4])
            boxes_new[:,0] = torch.min(second[:,0:4,0],dim = 1)[0]
            boxes_new[:,2] = torch.max(second[:,0:4,0],dim = 1)[0]
            boxes_new[:,1] = torch.min(second[:,0:4,1],dim = 1)[0]
            boxes_new[:,3] = torch.max(second[:,0:4,1],dim = 1)[0]
            second = boxes_new
            # second = second*3.281
            
            # find distances between first and second
            ious = np.zeros([len(first),len(second)])
            for i in range(0,len(first)):
                for j in range(0,len(second)):
                    ious[i,j] =  self.iou(first[i],second[j].data.numpy())
                    
            # get matches and keep those above threshold
            a, b = linear_sum_assignment(ious,maximize = True)
            matches = []
            gt_im_matched_idxs = []
            pred_im_matched_idxs = []
            for i in range(len(a)):
                iou = ious[a[i],b[i]]
                self.m["pre_thresh_IOU"].append(iou)
                if iou >= self.match_iou:
                    
                    matches.append([a[i],b[i]])
                    gt_im_matched_idxs.append(a[i])
                    pred_im_matched_idxs.append(b[i])
                    
                    self.m["match_IOU"].append(iou)
            
            # plot
            if True and self.sequence:
                # gt                
                unmatched_idxs = []
                for i in range(len(gt_im)):
                    if i not in gt_im_matched_idxs:
                        unmatched_idxs.append(i)
                gt_im_unmatched = gt_im[unmatched_idxs]
                
                # preds
                unmatched_idxs = []
                for i in range(len(pred_im)):
                    if i not in pred_im_matched_idxs:
                        unmatched_idxs.append(i)
                pred_im_unmatched = pred_im[unmatched_idxs]
                
                pred_im_matched = pred_im[pred_im_matched_idxs]
                gt_im_matched   = gt_im[gt_im_matched_idxs]
                
                self.hg.plot_boxes(im, pred_im_matched, color = (255,0,0)) # blue
                self.hg.plot_boxes(im,gt_im_matched,color = (0,255,0))     # green
                
                self.hg.plot_boxes(im, gt_im_unmatched,color = (0,0,255),thickness =2)     # red
                self.hg.plot_boxes(im, pred_im_unmatched,color = (0,100,255),thickness =2) # orange

                cv2.imshow("frame",im)
                
                key = cv2.waitKey(1)
                if key == ord("p"):
                    cv2.waitKey(0)
                    
                    
                    
            
            # of the pred objects not in b, dont count as FP those that fall outside of frame 
            for i in range(len(pred_ids)):
                if i not in b: # no match
                    obj = pred_im[i]
                    if obj[0,0] < 0 or obj[2,0] < 0 or obj[0,0] > 1920 or obj[2,0] > 1920:
                         self.m["FP edge-case"] += 1
                         continue
                    if obj[0,1] < 0 or obj[2,1] < 0 or obj[0,1] > 1080 or obj[2,1] > 1080:
                         self.m["FP edge-case"] += 1
                         
            
            # count FP, FN, TP
            self.m["TP"] += len(matches)
            self.m["FP"] += max(0,(len(pred_state) - len(matches)))
            self.m["FN"] += max(0,(len(gt_state) - len(matches)))
            
            self.m["FP @ 0.2"] += max(0,len(pred_state) - len(a))
            self.m["FN @ 0.2"] += max(0,len(gt_state) - len(a))
            
            for match in matches:
                # for each match, store error in L,W,H,x,y,velocity
                state_err = torch.clamp(torch.abs(pred_state[match[1]] - gt_state[match[0]]),0,500)
                self.m["state_err"].append(state_err)
                
                # for each match, store absolute 3D bbox pixel error for top and bottom
                bot_err = torch.clamp(torch.mean(torch.sqrt(torch.sum(torch.pow(pred_im[match[1],0:4,:] - gt_im[match[0],0:4,:],2),dim = 1))),0,500)
                top_err = torch.clamp(torch.mean(torch.sqrt(torch.sum(torch.pow(pred_im[match[1],4:8,:] - gt_im[match[0],4:8,:],2),dim = 1))),0,500)
                self.m["im_bot_err"].append(bot_err)
                self.m["im_top_err"].append(top_err)
            
            # for each match, store whether the class was predicted correctly or incorrectly, on a per class basis
            # index matrix by [true class,pred class]
            for match in matches:
                cls_string = gt_classes[match[0]]
                gt_cls = self.hg.class_dict[cls_string]
                cls_string = pred_classes[match[1]]
                if cls_string == '':
                    continue
                pred_cls = self.hg.class_dict[cls_string]
                self.m["cls"][gt_cls,pred_cls] += 1
            
            # store the ID associated with each ground truth object
            for match in matches:
                gt_id = gt_ids[match[0]]
                pred_id = pred_ids[match[1]]
                
                try:
                    if pred_id != self.m["ids"][gt_id][-1]:
                        self.m["ids"][gt_id].append(pred_id)
                except KeyError:
                    self.m["ids"][gt_id] = [pred_id]
                    
                if pred_id not in self.m["pred_ids"]:
                    self.m["pred_ids"].append(pred_id)
                if gt_id not in self.m["gt_ids"]:
                    self.m["gt_ids"].append(gt_id)    
        
        if self.sequence:
            self.cap.release()
            cv2.destroyAllWindows()
        
        # at the end:
        metrics = {}
        metrics["iou_threshold"] = self.match_iou
        metrics["True unique objects"] = len(self.m["gt_ids"])
        metrics["Predicted unique objects"] = len(self.m["pred_ids"])
        metrics["TP"] = self.m["TP"]
        metrics["FP"] = self.m["FP"]
        metrics["FN"] = self.m["FN"]
        metrics["FP edge-case"] = self.m["FP edge-case"]
        metrics["FP @ 0.2"] = self.m["FP @ 0.2"]
        metrics["FN @ 0.2"] = self.m["FN @ 0.2"]
        
        # Compute detection recall, detection precision, detection False alarm rate
        metrics["Recall"] = self.m["TP"]/(self.m["TP"]+self.m["FN"])
        metrics["Precision"] = self.m["TP"]/(self.m["TP"]+self.m["FP"])
        metrics["False Alarm Rate"] = self.m["FP"]/self.m["TP"]
        # Compute fragmentations - # of IDs assocated with each GT
        metrics["Fragmentations"] = sum([len(self.m["ids"][key])-1 for key in self.m["ids"]])
        
        # Count ID switches - any time a pred ID appears in two GT object sets
        count = 0
        for pred_id in self.m["pred_ids"]:
            pred_id_count = 0
            for gt_id in self.m["ids"]:
                if pred_id in self.m["ids"][gt_id]:
                    pred_id_count += 1
                    
            if pred_id_count > 1:
                count += (pred_id_count -1) # penalize for more than one gt being matched to the same pred_id
        metrics["ID switches"] = count
        
        # Compute MOTA
        metrics["MOTA"] = 1 - (self.m["FN"] +  metrics["ID switches"] + self.m["FP"])/(self.m["TP"])
        metrics["MOTA edge-case"]  = 1 - (self.m["FN"] +  metrics["ID switches"] + self.m["FP"]- self.m["FP edge-case"])/(self.m["TP"])
        metrics["MOTA @ 0.2"] = 1 - (self.m["FN @ 0.2"] +  metrics["ID switches"] + self.m["FP @ 0.2"])/(self.m["TP"])
        
        # Compute average detection metrics in various spaces
        ious = np.array(self.m["match_IOU"])
        iou_mean_stddev = np.mean(ious),np.std(ious)
        
        pre_ious = np.array(self.m["pre_thresh_IOU"])
        pre_iou_mean_stddev = np.mean(pre_ious),np.std(pre_ious)
        
        state = torch.stack(self.m["state_err"])
        state_mean_stddev = torch.mean(state,dim = 0), torch.std(state,dim = 0)
        
        bot_err = torch.stack(self.m["im_bot_err"])
        bot_mean_stddev = torch.mean(bot_err),torch.std(bot_err)
        
        top_err = torch.stack(self.m["im_top_err"])
        top_mean_stddev = torch.mean(top_err),torch.std(top_err)
        
        metrics["Pre-threshold IOU"]   = pre_iou_mean_stddev
        metrics["Match IOU"]           = iou_mean_stddev
        metrics["Width precision"]     = state_mean_stddev[0][3],state_mean_stddev[1][3]
        metrics["Height precision"]    = state_mean_stddev[0][4],state_mean_stddev[1][4]
        metrics["Length precision"]    = state_mean_stddev[0][2],state_mean_stddev[1][2]
        metrics["Velocity precision"]  = state_mean_stddev[0][6],state_mean_stddev[1][6]
        metrics["X precision"]         = state_mean_stddev[0][0],state_mean_stddev[1][0]
        metrics["Y precision"]         = state_mean_stddev[0][1],state_mean_stddev[1][1]
        metrics["Bottom im precision"] = bot_mean_stddev
        metrics["Top im precision"]    = top_mean_stddev
        
        self.metrics = metrics
        self.print_metrics()
        
    def print_metrics(self):
        print("\n")

        for name in self.metrics:
            try: 
                unit = self.units[name]
                print("{:<30}: {:.2f}{} avg., {:.2f}{} st.dev.".format(name,self.metrics[name][0],unit,self.metrics[name][1],unit))
            except:
                print("{:<30}: {:.3f}".format(name,self.metrics[name]))
            
    

if __name__ == "__main__":
    
    camera_name = "p1c4"
    sequence_idx = 0
    pred_path = r"E:\I24-postprocess\June_5min\rectified\rectified_{}_{}_track_outputs_3D.csv".format(camera_name,sequence_idx)
    gt_path = r"E:\I24-postprocess\June_5min\FOR ANNOTATORS\rectified_{}_{}_track_outputs_3D.csv".format(camera_name,sequence_idx)
    
    vp_file = r"C:\Users\wangy79\Documents\I24_trajectory\manual-track-labeler-main\DATA\vp\{}_axes.csv".format(camera_name)
    point_file = r"C:\Users\wangy79\Documents\I24_trajectory\manual-track-labeler-main\DATA\tform\{}_im_lmcs_transform_points.csv".format(camera_name)
    
    sequence = r"E:\I24-postprocess\June_5min\Raw Video\{}_{}.mp4".format(camera_name,sequence_idx)

    
    # we have to define the scale factor for the transformation, which we do based on the first frame of data
    labels,data = load_i24_csv(gt_path)
    frame_data = data[0]
    # convert labels from first frame into tensor form
    boxes = []
    classes = []
    for item in frame_data:
        if len(item[11]) > 0:
            boxes.append(np.array(item[11:27]).astype(float))
            classes.append(item[3])
    boxes = torch.from_numpy(np.stack(boxes))
    boxes = torch.stack((boxes[:,::2],boxes[:,1::2]),dim = -1)
    
    # load homography
    hg = Homography()
    hg.add_i24_camera(point_file,vp_file,camera_name)
    heights = hg.guess_heights(classes)
    hg.scale_Z(boxes,heights,name = camera_name)
    
    params = {
        "cutoff_frame": 1000,
        "match_iou":0.51,
        "sequence":sequence
        }
    
    ev = MOT_Evaluator(gt_path,pred_path,hg,params = params)
    ev.evaluate()
    ev.print_metrics()
