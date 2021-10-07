'''
Below code is borrowed from Derek Gloudemans
'''
import torch
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from homography import Homography, load_i24_csv
import utils
import numpy.linalg as LA

class MOT_Evaluator():
    
    def __init__(self,gt_path,rec_path,tf_path, camera_name, homography,params = None):
        """
        gt_path - string, path to i24 csv file for ground truth tracks
        rec_path - string, path to i24 csv for rectified tracks
        homography - Homography object containing relevant scene information
        params - dict of parameter values to change
        """
        self.mode = ''
        self.match_iou = 0
        self.cutoff_frame = 10000
    
        # store homography
        self.hg = homography
        
        # data is stored as a groupby object. get frame f_idx by "self.gt.get_group(f_idx)"
        # load ground truth data
        gt = utils.read_data(gt_path)
        gt = utils.img_to_road(gt,tf_path, camera_name)
        self.gt = gt
        
        # load rec data
        self.rec = utils.read_data(rec_path)
        
        
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
            "ids":{}, # key: gt_id, value: matched rec_id
            "gt_ids":[],
            "rec_ids":[],
            "Changed ID pair":[],
            "trajectory_score": {} # key: gt_id, value: score of the matched rec_id
            }
        
        units = {}
        units["Match IOU"]           = ""
        units["Pre-threshold IOU"]   = ""
        units["Trajectory score"]   = ""
        units["Width precision"]     = "ft"
        units["Height precision"]    = "ft"
        units["Length precision"]    = "ft"
        units["Velocity precision"]  = "ft/s"
        units["X precision"]         = "ft"
        units["Y precision"]         = "ft"
        self.units = units
        
            
    
    def iou(self,a,b):
        """
        Description
        -----------
        Calculates intersection over union for all sets of boxes in a and b
    
        Parameters
        ----------
        a : array of size [1,8] 
            bounding boxes in relative coords
        b : array of size [1,8] 
            bounding boxes in relative coords
    
        Returns
        -------
        iou - float between [0,1] if a, b are valid boxes, -1 otherwise
            average iou for a and b
        """
        if ~np.isnan(np.sum([a,b])): # if no Nan in any measurements
            p = Polygon([(a[2*i],a[2*i+1]) for i in range(int(len(a)/2))])
            q = Polygon([(b[2*i],b[2*i+1]) for i in range(int(len(b)/2))])
            if (p.intersects(q)):
                intersection_area = p.intersection(q).area
                union_area = p.union(q).area
                iou = float(intersection_area/union_area)
            else:
                iou = 0 
        else:
            iou = 0
        return iou

    def score_trajectory(self):
        '''
        compute euclidean distance between GT trajectories and rec trajectories
        '''
        gt_groups = self.gt.groupby('ID')
        rec_groups = self.rec.groupby('ID')
        
        if self.metrics['Matched IDs']:
           for gt_id in self.metrics['Matched IDs']:
               rec_id = self.metrics['Matched IDs'][gt_id]
               gt_car = gt_groups.get_group(gt_id)
               rec_car = rec_groups.get_group(rec_id[0])
               start = max(gt_car['Frame #'].iloc[0],rec_car['Frame #'].iloc[0])
               end = min(gt_car['Frame #'].iloc[-1],rec_car['Frame #'].iloc[-1])
               gt_car = gt_car.loc[(gt_car['Frame #'] >= start) & (gt_car['Frame #'] <= end)]
               rec_car = rec_car.loc[(rec_car['Frame #'] >= start) & (rec_car['Frame #'] <= end)]
               Y1 = np.array(gt_car[['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']])
               Yre = np.array(rec_car[['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']])
               diff = Y1-Yre
               self.m["trajectory_score"][gt_id] = np.nanmean(LA.norm(diff,axis=1))
        else:
            print('Run evaluate first.')
        scores = list(self.m["trajectory_score"].values())
        scores_mean_std = np.mean(scores),np.std(scores)
        self.metrics["Trajectory score"] = scores_mean_std
        
        return
    
    def evaluate(self):
        
        # for each frame:
        gt_frames = self.gt.groupby('Frame #')
        rec_frames = self.rec.groupby('Frame #')
        
        for f_idx in range(self.cutoff_frame):
            
            print("\rAggregating metrics for frame {}/{}".format(f_idx,self.cutoff_frame),end = "\r",flush = True)
            
            try:
                gt = gt_frames.get_group(f_idx)
            except KeyError:
                if f_idx in rec_frames.groups.keys():
                    frame = rec_frames.get_group(f_idx)
                    self.m["FP"] += len(frame)
                    ids =  frame['ID'].values
                    for id in ids:
                        if id not in self.m["rec_ids"]:
                            self.m["rec_ids"].append(id)
                continue
            
            try:
                rec = rec_frames.get_group(f_idx)
            except KeyError:
                if f_idx in gt_frames.groups.keys():
                    frame = gt_frames.get_group(f_idx)
                    self.m["FN"] += len(frame)
                    ids =  frame['ID'].values
                    for id in ids:
                        if id not in self.m["gt_ids"]:
                            self.m["gt_ids"].append(id)
                continue
                    
            # store ground truth as tensors
            gt_ids = gt['ID'].values
            gt_space = np.array(gt[['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']])
            
            # store rec as tensors (we start from state)
            rec_ids = rec['ID'].values
            rec_space = np.array(rec[['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']])
        
            # compute matches based on space location ious
            first = gt_space
            second = rec_space
        
            # find distances between first and second
            ious = np.zeros([len(first),len(second)])
            for i in range(0,len(first)):
                for j in range(0,len(second)):
                    ious[i,j] =  self.iou(first[i],second[j])
                    
            # get matches and keep those above threshold
            a, b = linear_sum_assignment(ious,maximize = True) # a,b are row,col: matched idx pair of the matrix ious
            matches = []
            gt_im_matched_idxs = []
            rec_im_matched_idxs = []
            for i in range(len(a)):
                iou = ious[a[i],b[i]]
                self.m["pre_thresh_IOU"].append(iou)
                if iou >= self.match_iou:
                    
                    matches.append([a[i],b[i]])
                    gt_im_matched_idxs.append(a[i])
                    rec_im_matched_idxs.append(b[i])
                    
                    self.m["match_IOU"].append(iou)
                      
            # store the ID associated with each ground truth object
            for match in matches:
                gt_id = gt_ids[match[0]]
                rec_id = rec_ids[match[1]]
                if gt_id != rec_id and (gt_id, rec_id) not in self.m['Changed ID pair']:
                    self.m['Changed ID pair'].append((gt_id, rec_id))
                try:
                    if rec_id != self.m["ids"][gt_id][-1]:
                        self.m["ids"][gt_id].append(rec_id)
                except KeyError:
                    self.m["ids"][gt_id] = [rec_id]
                    
                if rec_id not in self.m["rec_ids"]:
                    self.m["rec_ids"].append(rec_id)
                if gt_id not in self.m["gt_ids"]:
                    self.m["gt_ids"].append(gt_id)    

            self.m["TP"] += len(matches)
            self.m["FP"] += max(0,(len(rec_space) - len(matches)))
            self.m["FN"] += max(0,(len(gt_space) - len(matches)))
            
            self.m["FP @ 0.2"] += max(0,len(rec_space) - len(a))
            self.m["FN @ 0.2"] += max(0,len(gt_space) - len(a))
            
        # at the end:
        metrics = {}
        metrics["TP"] = self.m["TP"]
        metrics["FP"] = self.m["FP"]
        metrics["FN"] = self.m["FN"]
        metrics["FP @ 0.2"] = self.m["FP @ 0.2"]
        metrics["FN @ 0.2"] = self.m["FN @ 0.2"]
        metrics["iou_threshold"] = self.match_iou
        metrics["True unique objects"] = len(self.m["gt_ids"])
        metrics["recicted unique objects"] = len(self.m["rec_ids"])
        
        # Compute detection recall, detection precision, detection False alarm rate
        metrics["Recall"] = self.m["TP"]/(self.m["TP"]+self.m["FN"])
        metrics["Precision"] = self.m["TP"]/(self.m["TP"]+self.m["FP"])
        metrics["False Alarm Rate"] = self.m["FP"]/self.m["TP"]
        
        # Compute fragmentations - # of IDs assocated with each GT
        metrics["Fragmentations"] = sum([len(self.m["ids"][key])-1 for key in self.m["ids"]])
        metrics["Matched IDs"] = self.m["ids"]
        # Count ID switches - any time a rec ID appears in two GT object sets
        count = 0
        switched_ids = []
        for rec_id in self.m["rec_ids"]:
            rec_id_count = 0
            for gt_id in self.m["ids"]:
                if rec_id in self.m["ids"][gt_id]:
                    rec_id_count += 1
       
            if rec_id_count > 1:
                switched_ids.append(rec_id)
                count += (rec_id_count -1) # penalize for more than one gt being matched to the same rec_id
        metrics["ID switches"] = (count, switched_ids)
        metrics["Changed ID pair"] = self.m["Changed ID pair"]
         
        # Compute average detection metrics in various spaces
        ious = np.array(self.m["match_IOU"])
        iou_mean_stddev = np.mean(ious),np.std(ious)
        
        pre_ious = np.array(self.m["pre_thresh_IOU"])
        pre_iou_mean_stddev = np.mean(pre_ious),np.std(pre_ious)
    
        
        metrics["Pre-threshold IOU"]   = pre_iou_mean_stddev
        metrics["Match IOU"]           = iou_mean_stddev
       
        
        self.metrics = metrics
        # self.print_metrics()
        
    def print_metrics(self):
        print("\n")

        for name in self.metrics:
            if name == "Matched IDs": 
                continue
            try: 
                unit = self.units[name]
                print("{:<30}: {:.2f}{} avg., {:.2f}{} st.dev.".format(name,self.metrics[name][0],unit,self.metrics[name][1],unit))
            except:
                try: 
                    print("{:<30}: {:.3f}".format(name,self.metrics[name]))
                except:
                    print("{:<30}: {}".format(name,self.metrics[name]))


if __name__ == "__main__":
    
    camera_name = "p1c4"
    sequence_idx = 0
    rec_path = r"E:\I24-postprocess\June_5min\rectified\{}_{}.csv".format(camera_name,sequence_idx)
    gt_path = r"E:\I24-postprocess\June_5min\FOR ANNOTATORS\rectified_{}_{}_track_outputs_3D.csv".format(camera_name,sequence_idx)
    
    vp_file = r"C:\Users\wangy79\Documents\I24_trajectory\manual-track-labeler-main\DATA\vp\{}_axes.csv".format(camera_name)
    point_file = r"C:\Users\wangy79\Documents\I24_trajectory\manual-track-labeler-main\DATA\tform\{}_im_lmcs_transform_points.csv".format(camera_name)
    tf_path = r"C:\Users\wangy79\Documents\I24_trajectory\manual-track-labeler-main\DATA\tform"
    
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
        "cutoff_frame": 100,
        "match_iou":0.51,
        }
    
    ev = MOT_Evaluator(gt_path,rec_path,tf_path, camera_name, hg, params = params)
    ev.evaluate()
    ev.score_trajectory()
    ev.print_metrics()
