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
from utils_data_association import count_overlaps
import numpy.linalg as LA
import matplotlib.pyplot as plt
import utils_vis as vis
import pickle

class MOT_Evaluator():
    
    def __init__(self,gt_path,rec_path,tf_path, camera_name, homography,params = None):
        """
        gt_path - string, path to i24 csv file for ground truth tracks
        rec_path - string, path to i24 csv for rectified tracks
        homography - Homography object containing relevant scene information
        params - dict of parameter values to change
        """
        self.gtmode = ""
        self.recmode = ""
        self.match_iou = 0
        self.cutoff_frame = 10000
    
        # store homography
        self.hg = homography
        self.tf_path = tf_path
        self.camera_name = camera_name
        
        # data is stored as a groupby object. get frame f_idx by "self.gt.get_group(f_idx)"
        # load ground truth data
        # start with meter
        cols_to_convert = ["speed","x","y","width","length","height"]
        pts = ["fbr_x","fbr_y","fbl_x","fbl_y","bbr_x","bbr_y","bbl_x","bbl_y"]
        self.gt = utils.read_data(gt_path)
        if np.mean(self.gt.y.values) > 40:      
            self.gt[cols_to_convert] = self.gt[cols_to_convert] / 3.281
        if "bbr_x" not in self.gt or np.mean(self.gt.bbr_y.values) > 40 or "Manual" in self.gt["Generation method"].unique():
            self.gt = utils.img_to_road(self.gt, tf_path, camera_name)
        
        # calculate GT velocities TODO:assume freeflow constant velocity
        # self.gt = utils.calc_dynamics(self.gt) # finite difference
        self.gt = self.gt.groupby("ID").apply(self.estimate_speed).reset_index(drop=True)
        
        # load rec data
        self.rec = utils.read_data(rec_path)
        if "veh rear x" in self.rec:
            self.rec = self.rec.rename(columns={"veh rear x": "x", "veh center y":"y", "Object ID": "ID"})
        if np.mean(self.rec.y.values) > 40:      
            self.rec[cols_to_convert] = self.rec[cols_to_convert] / 3.281
        if "bbr_x" not in self.rec or np.mean(self.rec.bbr_y.values) > 40:
            self.rec = utils.img_to_road(self.rec, tf_path, camera_name)
         
        
        if params is not None:
            if "match_iou" in params.keys():
                self.match_iou = params["match_iou"]
            if "cutoff_frame" in params.keys():
                self.cutoff_frame = params["cutoff_frame"]
            if "sequence" in params.keys():
                self.sequence = params["sequence"]
            if "gtmode" in params.keys():
                self.gtmode = params["gtmode"]
            if "recmode" in params.keys():
                self.recmode = params["recmode"]
            if "score_threshold" in params.keys():
                self.score_threshold = params["score_threshold"]
        
        if self.recmode != "rec":
            self.rec = self.rec.groupby("ID").apply(utils.calc_dynamics_car).reset_index(drop=True)
            
        # select under cut-off frames
        if self.cutoff_frame:
            self.gt = self.gt[self.gt["Frame #"]<=self.cutoff_frame]
            self.rec = self.rec[self.rec["Frame #"]<=self.cutoff_frame]
            
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
            "ids_rec":{}, # key: rec_id, value: matched gt_id
            "gt_ids":[],
            "rec_ids":[],
            "Changed ID pair":[],
            "trajectory_score": {}, # key: gt_id, value: score of the matched rec_id
            "ids > score":[],
            "overlap_gt": 0,
            "overlap_rec": 0,
            "space_gap_gt":[],
            "space_gap_rec":[],
            "overlap_rec_ids": set()
            }
        
        units = {}
        units["Match IOU"]           = ""
        units["Pre-threshold IOU"]   = ""
        units["Trajectory score"]    = ""
        units["Spacing before"]      = ""
        units["Spacing after"]       = ""
        units["Width precision"]     = "ft"
        units["Height precision"]    = "ft"
        units["Length precision"]    = "ft"
        units["Velocity precision"]  = "ft/s"
        units["X precision"]         = "ft"
        units["Y precision"]         = "ft"
        units["df"]                  = "m"
        self.units = units
        
            
        if self.sequence is not None:
            self.cap = cv2.VideoCapture(self.sequence)
    
    def estimate_speed(self, car):
        temp = car[~car["bbr_x"].isna()]
        if len(temp)<2:
            return None
        v_bbr = (max(temp.bbr_x.values)-min(temp.bbr_x.values))/(max(temp.Timestamp.values)-min(temp.Timestamp.values))
        v_fbr = (max(temp.fbr_x.values)-min(temp.fbr_x.values))/(max(temp.Timestamp.values)-min(temp.Timestamp.values))
        avgv = (v_bbr+v_fbr)/2
        car["speed"] = avgv if avgv<50 else np.nan
        return car
        
    def iou(self,a,b):
        """
        Description
        -----------
        Calculates intersection over union for all sets of boxes in a and b
    
        Parameters
        ----------
        a : tensor of size [8,3] 
            bounding boxes in relative coords
        b : array of size [8,3] 
            bounding boxes in relative coords
    
        Returns
        -------
        iou - float between [0,1] if a, b are valid boxes, -1 otherwise
            average iou for a and b
        """
        # if has invalid measurements
        if torch.isnan(a).any() or torch.isnan(b).any():
            return 0
        # ignore the top
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

    def score_trajectory(self):
        '''
        compute euclidean distance between GT trajectories and rec trajectories
        '''
        # convert back to meter
        if self.units["df"] == "ft":
            cols_to_convert = ["fbr_x",	"fbr_y","fbl_x"	,"fbl_y","bbr_x","bbr_y","bbl_x","bbl_y", "speed","x","y","width","length","height"]
            self.rec[cols_to_convert] = self.rec[cols_to_convert] / 3.281
            self.gt[cols_to_convert] = self.gt[cols_to_convert] / 3.281
            self.units["df"] = "m"
        
        gt_groups = self.gt.groupby('ID')
        rec_groups = self.rec.groupby('ID')
        
        if self.metrics['Matched IDs rec']:
           for rec_id in self.metrics['Matched IDs rec']:
               gt_id = self.metrics['Matched IDs rec'][rec_id][0]
               gt_car = gt_groups.get_group(gt_id)
               rec_car = rec_groups.get_group(rec_id)
               start = max(gt_car['Frame #'].iloc[0],rec_car['Frame #'].iloc[0])
               end = min(gt_car['Frame #'].iloc[-1],rec_car['Frame #'].iloc[-1])
               gt_car = gt_car.loc[(gt_car['Frame #'] >= start) & (gt_car['Frame #'] <= end)]
               rec_car = rec_car.loc[(rec_car['Frame #'] >= start) & (rec_car['Frame #'] <= end)]
               Y1 = np.array(gt_car[['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']])
               Yre = np.array(rec_car[['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']])
               try:
                   diff = Y1-Yre
                   score = np.nanmean(LA.norm(diff,axis=1))
                   if score > self.score_threshold:
                       self.m["ids > score"].append(rec_id)
                   self.m["trajectory_score"][rec_id] = score
               except ValueError:
                   print("Encounter unmatched dimension when computing trajectory score")
        else:
            print('Run evaluate first.')
        scores = list(self.m["trajectory_score"].values())   
        scores = [x for x in scores if np.isnan(x) == False]
        scores_mean_std = np.nanmean(scores),np.nanstd(scores)
        metrics = {}
        metrics["Trajectory score"] = scores_mean_std
        metrics["IDs > score threshold"] = self.m["ids > score"]
        
        if hasattr(self, "metrics"):
            self.metrics = dict(list(self.metrics.items()) + list(metrics.items()))
        else:
            self.metrics = metrics
        return
    
    def iou_ts(self,a,b):
        """
        Description
        -----------
        Calculates intersection over union for track a and b in time-space diagram
    
        Parameters
        ----------
        a : 1x8
        b : 1x8
        Returns
        -------
        iou - float between [0,1] 
        """
        a,b = np.reshape(a,(1,-1)), np.reshape(b,(1,-1))
    
        p = Polygon([(a[0,2*i],a[0,2*i+1]) for i in range(4)])
        q = Polygon([(b[0,2*i],b[0,2*i+1]) for i in range(4)])

        intersection_area = p.intersection(q).area
        union_area = min(p.area, q.area)
        iou = float(intersection_area/union_area)
                
        return iou
    
    def get_invalid(self, df):
        '''
        valid: length covers more than 50% of the FOV
        invalid: length covers less than 10% of FOV, or
                crashes with any valid tracks
        undetermined: tracks that are short but not overlaps with any valid tracks
        '''
        # convert units
        if np.mean(df.y.values) > 40:   
            cols_to_convert = ["fbr_x",	"fbr_y","fbl_x"	,"fbl_y","bbr_x","bbr_y","bbl_x","bbl_y", "speed","x","y","width","length","height"]
            df[cols_to_convert] = df[cols_to_convert] / 3.281
        xmin, xmax = min(df["x"].values),max(df["x"].values)
        groups = df.groupby("ID")
        groupList = list(groups.groups)
        pts = ['bbr_x','bbr_y','fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']
        
        valid = {}
        invalid = set()
        for carid, group in groups:
            if (max(group.x.values)-min(group.x.values)>0.4*(xmax-xmin)): # long tracks
                frames = group["Frame #"].values
                first = group.head(1)
                last = group.tail(1)
                x0, x1 = max(first.bbr_x.values[0],first.fbr_x.values[0]),min(first.bbr_x.values[0],first.fbr_x.values[0])
                x2, x3 = min(last.bbr_x.values[0],last.fbr_x.values[0]),max(last.bbr_x.values[0],last.fbr_x.values[0])
                y0, y1 = max(first.bbr_y.values[0],first.bbl_y.values[0]),min(first.bbr_y.values[0],first.bbl_y.values[0])
                y2, y3 = min(last.bbr_y.values[0],last.bbl_y.values[0]),max(last.bbr_y.values[0],last.bbl_y.values[0])
                t0,t1 = min(frames), max(frames)
                valid[carid] = [np.array([t0,x0,t0,x1,t1,x2,t1,x3]),np.array([t0,y0,t0,y1,t1,y2,t1,y3])]
                
        # check crash within valid
        valid_list = list(valid.keys())
        for i,car1 in enumerate(valid_list):
            bx,by = valid[car1]
            for car2 in valid_list[i+1:]:
                ax,ay = valid[car2]
                ioux = self.iou_ts(ax,bx)
                iouy = self.iou_ts(ay,by)
                if ioux > 0 and iouy > 0: # trajectory overlaps with a valid track
                    if bx[4]-bx[0] > ax[4]-ax[0]: # keep the longer track    
                        invalid.add(car2)
                    else:
                        invalid.add(car1)
        valid = set(valid.keys())
        valid = valid-invalid
        print("Valid tracklets: {}/{}".format(len(valid),len(groupList)))

        return valid
    
    def evaluate_tracks(self):
        '''
        Compute:
        # valid and invalid tracks
            - valid if x covers the range of camera FOV
            - invalid if otherwise or the ratio of missing data is > threshold
        # collisions
        ** IN SPACE ** 
        Returns
        -------
        None.

        '''
        
        
        valid = self.get_invalid(self.rec)
        groups = self.rec.groupby("ID")
        self.metrics["Valid tracklets/total"] = "{} / {}".format(len(valid),groups.ngroups)
        
        # invalid_gt, valid_gt, invalid_rec, valid_rec = [],[],[],[]
        # xmin, xmax, _, _ = utils.get_camera_range(self.gt['camera'].dropna().unique())
        # xrange = xmax-xmin
        # alpha = 0.5
        # xmin, xmax = xmin + alpha*xrange, xmax-alpha*xrange # buffered 1-2*alpha%
        
        # # invalid if tracks don't cover xmin to xmax, or tracks has < 5 valid measurements
        # print("Evaluating tracks...")
        # gt_groups = self.gt.groupby("ID")
            
        # for gt_id, gt in gt_groups:
        #     x_df = gt[["bbr_x","bbl_x","fbr_x","fbl_x"]]
        #     xleft = x_df.min().min()
        #     xright = x_df.max().max()
        #     # missing_rate = x_df[["bbr_x"]].isna().sum().values[0]/len(x_df)
        #     valid_meas = gt.count().bbr_x
        #     if xleft > xmin or xright < xmax or valid_meas < 5:
        #         invalid_gt.append(gt_id)
        #     else:
        #         valid_gt.append(gt_id)
                
        # # do the same for rec
        # rec_groups = self.rec.groupby("ID")
        # for rec_id, rec in rec_groups:
        #     x_df = rec[["bbr_x","bbl_x","fbr_x","fbl_x"]]
        #     xleft = x_df.min().min()
        #     xright = x_df.max().max()
        #     valid_meas = rec.count().bbr_x
        #     if xleft > xmin or xright < xmax or valid_meas < 5:
        #         invalid_rec.append(rec_id)
        #     else:
        #         valid_rec.append(rec_id)
        
        # summarize
        # metrics = {}
        # metrics["# Invalid IDs/total (before)"] = "{} / {}".format(len(invalid_gt),gt_groups.ngroups)
        # metrics["# Invalid IDs/total (after)"] = "{} / {}".format(len(invalid_rec),rec_groups.ngroups)
        # metrics["Invalid rec IDs"] = invalid_rec
        # # metrics["Occurances of collision in rec"] = len(overlaps_rec)
        # if hasattr(self, "metrics"):
        #     self.metrics = dict(list(self.metrics.items()) + list(metrics.items()))
        # else:
        #     self.metrics = metrics
            
        # plot some invalid tracks
        # for rec_id in invalid_rec[:5]:
        #     car = rec_groups.get_group(rec_id)
        #     vis.plot_track_df(car, title=str(rec_id))
        return
    
    def evaluate(self):
        # TODO: convert gt and rec unit from m to ft
        cols_to_convert = ["fbr_x",	"fbr_y","fbl_x","fbl_y","bbr_x","bbr_y","bbl_x","bbl_y", "speed","x","y","width","length","height"]
        if "height" not in self.gt:
            self.gt["height"] = 0
        if "height" not in self.rec:
            self.rec["height"] = 0
        if self.units["df"] == "m":
            self.gt[cols_to_convert] = self.gt[cols_to_convert] * 3.281
            self.rec[cols_to_convert] = self.rec[cols_to_convert] * 3.281
            self.units["df"] = "ft"
        
        # for each frame:
        gt_frames = self.gt.groupby('Frame #')
        rec_frames = self.rec.groupby('Frame #')
        
        for f_idx in range(self.cutoff_frame):
            
            print("\rAggregating metrics for frame {}/{}".format(f_idx,self.cutoff_frame),end = "\r",flush = True)
            if self.sequence:
                _,im = self.cap.read()
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
            gt_classes = gt["Object class"].values
            
            if self.gtmode == "gt": # start from image
                # TODO: fill nan as 0 for velocity
                gt_im = np.array(gt[["fbrx","fbry",	"fblx",	"fbly",	"bbrx",	"bbry",	"bblx",	"bbly",	"ftrx",	"ftry",	"ftlx",	"ftly",	"btrx",	"btry",	"btlx",	"btly"]])   
                gt_im = torch.from_numpy(np.stack(gt_im)).reshape(-1,8,2)
            
                # two pass estimate of object heights
                heights = self.hg.guess_heights(gt_classes)
                gt_state = self.hg.im_to_state(gt_im,heights = heights)
                repro_boxes = self.hg.state_to_im(gt_state)
                refined_heights = self.hg.height_from_template(repro_boxes,heights,gt_im)
                
                # get other formulations for boxes
                gt_state = self.hg.im_to_state(gt_im,heights = refined_heights)
                gt_space = self.hg.state_to_space(gt_state)
                
                gt_velocities = gt["speed"].values
                gt_velocities = torch.tensor(gt_velocities).float()
                gt_state = torch.cat((gt_state,gt_velocities.unsqueeze(1)),dim = 1)
                
            else: # start from space (raw, DA)
                gt_space = np.array(gt[['fbr_x','fbr_y', 'fbl_x','fbl_y','bbr_x','bbr_y','bbl_x', 'bbl_y']])
                gt_space = torch.from_numpy(np.stack(gt_space)).reshape(-1,4,2)
                gt_space = torch.cat((gt_space,gt_space),dim = 1)
                d = gt_space.size()[0]
                zero_heights =  torch.zeros((d,8,1))
                gt_space = torch.cat([gt_space,zero_heights],dim=2)
                gt_im = self.hg.space_to_im(gt_space)
            
            
            # store pred as tensors (we start from state)
            rec_ids = rec['ID'].values
            rec_classes = rec["Object class"].values
            
            if self.recmode == "da":
                # rec_space = np.array(rec[['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']])
                rec_space = np.array(rec[['fbr_x','fbr_y', 'fbl_x','fbl_y','bbr_x','bbr_y','bbl_x', 'bbl_y']])
                rec_space = torch.from_numpy(np.stack(rec_space)).reshape(-1,4,2)
                rec_space = torch.cat((rec_space,rec_space),dim = 1)
                d = rec_space.size()[0]
                zero_heights =  torch.zeros((d,8,1))
                rec_space = torch.cat([rec_space,zero_heights],dim=2)
                rec_im = self.hg.space_to_im(rec_space)
                
                heights = self.hg.guess_heights(rec_classes)
                rec_state = self.hg.im_to_state(rec_im,heights = heights)
                # TODO: estimate speed from space
                rec_velocities = rec["speed"].values
                rec_velocities = torch.tensor(rec_velocities).float()
                rec_state = torch.cat((rec_state,rec_velocities.unsqueeze(1)),dim = 1)
                
            elif self.recmode == "rec": # start from states
                rec_state = np.array(rec[["x","y","length","width","height","direction","speed"]])
                rec_state = torch.from_numpy(np.stack(rec_state)).reshape(-1,7).float()
                rec_space = self.hg.state_to_space(rec_state)
                rec_im = self.hg.state_to_im(rec_state)                         
            
            else: # start from image
                rec_im = np.array(rec[["fbrx","fbry","fblx",	"fbly",	"bbrx",	"bbry",	"bblx",	"bbly",	"ftrx",	"ftry",	"ftlx",	"ftly",	"btrx",	"btry",	"btlx",	"btly"]])   
                rec_im = torch.from_numpy(np.stack(rec_im)).reshape(-1,8,2)
            
                # two pass estimate of object heights
                heights = self.hg.guess_heights(rec_classes)
                rec_state = self.hg.im_to_state(rec_im,heights = heights)
                repro_boxes = self.hg.state_to_im(rec_state)
                refined_heights = self.hg.height_from_template(repro_boxes,heights,rec_im)
                
                # get other formulations for boxes
                rec_state = self.hg.im_to_state(rec_im,heights = refined_heights)
                rec_space = self.hg.state_to_space(rec_state)

                rec_velocities = rec["speed"].values
                rec_velocities = torch.tensor(rec_velocities).float()
                rec_state = torch.cat((rec_state,rec_velocities.unsqueeze(1)),dim = 1)
                
            # compute matches based on space location ious
            first = gt_space.clone() # xmin,ymin,xmax,ymax
            boxes_new = torch.zeros([first.shape[0],4])
            boxes_new[:,0] = torch.min(first[:,0:4,0],dim = 1)[0]
            boxes_new[:,2] = torch.max(first[:,0:4,0],dim = 1)[0]
            boxes_new[:,1] = torch.min(first[:,0:4,1],dim = 1)[0]
            boxes_new[:,3] = torch.max(first[:,0:4,1],dim = 1)[0]
            first = boxes_new
            
            second = rec_space.clone()
            boxes_new = torch.zeros([second.shape[0],4])
            boxes_new[:,0] = torch.min(second[:,0:4,0],dim = 1)[0]
            boxes_new[:,2] = torch.max(second[:,0:4,0],dim = 1)[0]
            boxes_new[:,1] = torch.min(second[:,0:4,1],dim = 1)[0]
            boxes_new[:,3] = torch.max(second[:,0:4,1],dim = 1)[0]
            second = boxes_new
        
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
            
            # find relative distance, overlaps (crashes) and spacing in first and second
            d = len(first)
            dx,dy = torch.ones((d,d))*(999), torch.ones((d,d))*(-1) # upper triangle matrices
            for i in range(0,d):
                for j in range(i+1,d):
                    dx[i][j] = abs(first[i][0]-first[j][0])
                    dy[i][j] = abs(first[i][1]-first[j][1])
                    if self.iou(first[i],first[j]) > 0:
                        self.m["overlap_gt"] += 1
            # extract leader and spacing information
            dy[dy<0] = float('nan')
            for i in range(d):
                js = torch.where((dy[i]>=0)&(dy[i]<0.3)) # all potential leaders of i
                if len(dx[i][js])> 0: 
                    # j = torch.argmin(dx[i][js]) # immediate leader
                    self.m["space_gap_gt"].append(min(dx[i][js]))

            
            d = len(second)
            dx,dy = torch.ones((d,d))*(999), torch.ones((d,d))*(-1) # upper triangle matrices
            for i in range(0,d):
                for j in range(i+1,d):
                    dx[i][j] = abs(second[i][0]-second[j][0])
                    dy[i][j] = abs(second[i][1]-second[j][1])
                    if self.iou(second[i],second[j]) > 0:
                        self.m["overlap_rec"] += 1
                        self.m["overlap_rec_ids"].add((rec_ids[i],rec_ids[j]))
            # extract leader and spacing information
            dy[dy<0] = float('nan')
            for i in range(d):
                js = torch.where((dy[i]>=0)&(dy[i]<0.3)) # all potential leaders of i
                if len(dx[i][js])> 0: 
                    # j = torch.argmin(dx[i][js]) # immediate leader
                    self.m["space_gap_rec"].append(min(dx[i][js]))
                    
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
                for i in range(len(rec_im)):
                    if i not in rec_im_matched_idxs:
                        unmatched_idxs.append(i)
                rec_im_unmatched = rec_im[unmatched_idxs]
                
                rec_im_matched = rec_im[rec_im_matched_idxs]
                gt_im_matched   = gt_im[gt_im_matched_idxs]
                
                self.hg.plot_boxes(im,rec_im_matched, color = (255,0,0))   # blue
                self.hg.plot_boxes(im,gt_im_matched,color = (0,255,0))     # green
                
                self.hg.plot_boxes(im, gt_im_unmatched,color = (0,0,255),thickness =2)     # red, FN
                self.hg.plot_boxes(im, rec_im_unmatched,color = (0,100,255),thickness =2) # orange, FP

                cv2.imshow("frame",im)
                
                key = cv2.waitKey(1)
                if key == ord("p"):
                    cv2.waitKey(0)
                    
                if key == ord("q"):
                    self.cap.release()
                    cv2.destroyAllWindows()
                    return
                
                    
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
                try:
                    if gt_id != self.m["ids_rec"][rec_id][-1]:
                        self.m["ids_rec"][rec_id].append(gt_id)
                except KeyError:
                    self.m["ids_rec"][rec_id] = [gt_id]    
                if rec_id not in self.m["rec_ids"]:
                    self.m["rec_ids"].append(rec_id)
                if gt_id not in self.m["gt_ids"]:
                    self.m["gt_ids"].append(gt_id)    


            # of the pred objects not in b, dont count as FP those that fall outside of frame 
            for i in range(len(rec_ids)):
                if i not in b: # no match
                    obj = rec_im[i]
                    if obj[0,0] < 0 or obj[2,0] < 0 or obj[0,0] > 1920 or obj[2,0] > 1920:
                         self.m["FP edge-case"] += 1
                         continue
                    if obj[0,1] < 0 or obj[2,1] < 0 or obj[0,1] > 1080 or obj[2,1] > 1080:
                         self.m["FP edge-case"] += 1
                         
            self.m["TP"] += len(matches)
            invalid_rec = torch.sum(torch.sum(torch.isnan(rec_space), dim=1),dim=1)>0
            invalid_gt = torch.sum(torch.sum(torch.isnan(gt_space), dim=1),dim=1)>0
            self.m["FP"] += max(0,(len(rec_space)-sum(invalid_rec) - len(matches)))
            self.m["FN"] += max(0,(len(gt_space)-sum(invalid_gt) - len(matches)))
            
            # self.m["FP @ 0.2"] += max(0,len(rec_space)-sum(invalid_rec) - len(a))
            # self.m["FN @ 0.2"] += max(0,len(gt_space)-sum(invalid_gt) - len(a))
            
            # if self.recmode == "state":
            for match in matches:
                # for each match, store error in L,W,H,x,y,velocity
                state_err = torch.clamp(torch.abs(rec_state[match[1]] - gt_state[match[0]]),0,500)
                self.m["state_err"].append(state_err)
                
                # for each match, store absolute 3D bbox pixel error for top and bottom
                bot_err = torch.clamp(torch.mean(torch.sqrt(torch.sum(torch.pow(rec_im[match[1],0:4,:] - gt_im[match[0],0:4,:],2),dim = 1))),0,500)
                top_err = torch.clamp(torch.mean(torch.sqrt(torch.sum(torch.pow(rec_im[match[1],4:8,:] - gt_im[match[0],4:8,:],2),dim = 1))),0,500)
                self.m["im_bot_err"].append(bot_err)
                self.m["im_top_err"].append(top_err)
         
        if self.sequence:
            self.cap.release()
            cv2.destroyAllWindows()
        
        # at the end:
        metrics = {}
        metrics["TP"] = self.m["TP"]
        metrics["FP"] = self.m["FP"]
        metrics["FN"] = self.m["FN"]
        # metrics["FP @ 0.2"] = self.m["FP @ 0.2"]
        # metrics["FN @ 0.2"] = self.m["FN @ 0.2"]
        # metrics["iou_threshold"] = self.match_iou
        metrics["True unique objects"] = len(self.m["gt_ids"])
        metrics["recicted unique objects"] = len(self.m["rec_ids"])
        metrics["FP edge-case"] = self.m["FP edge-case"]
        
        # Compute detection recall, detection precision, detection False alarm rate
        metrics["Recall"] = self.m["TP"]/(self.m["TP"]+self.m["FN"])
        metrics["Precision"] = self.m["TP"]/(self.m["TP"]+self.m["FP"])
        metrics["False Alarm Rate"] = self.m["FP"]/self.m["TP"]
        
        # Compute fragmentations - # of IDs assocated with each GT
        metrics["Fragmentations"] = sum([len(self.m["ids"][key])-1 for key in self.m["ids"]])
        metrics["Fragments"] = [self.m["ids"][key] for key in self.m["ids"] if len(self.m["ids"][key])>1]
        metrics["Matched IDs"] = self.m["ids"]
        metrics["Matched IDs rec"] = self.m["ids_rec"]
        
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
        # metrics["Changed ID pair"] = self.m["Changed ID pair"]
         
         # Compute MOTA
        metrics["MOTA"] = 1 - (self.m["FN"] +  metrics["ID switches"][0] + self.m["FP"] + metrics["Fragmentations"])/(self.m["TP"])
        metrics["MOTA edge-case"]  = 1 - (self.m["FN"] +  metrics["ID switches"][0] + self.m["FP"]- self.m["FP edge-case"]+ metrics["Fragmentations"])/(self.m["TP"])
        # metrics["MOTA @ 0.2"] = 1 - (self.m["FN @ 0.2"] +  metrics["ID switches"][0] + self.m["FP @ 0.2"])/(self.m["TP"])
        
        ious = np.array(self.m["match_IOU"])
        iou_mean_stddev = np.mean(ious),np.std(ious)
        
        pre_ious = np.array(self.m["pre_thresh_IOU"])
        pre_iou_mean_stddev = np.mean(pre_ious),np.std(pre_ious)
         
        spacing_gt = np.array(self.m["space_gap_gt"])
        spacing_gt_mean_stdev = np.mean(spacing_gt), np.std(spacing_gt)
        spacing_rec = np.array(self.m["space_gap_rec"])
        spacing_rec_mean_stdev = np.mean(spacing_rec), np.std(spacing_rec)
        
        metrics["Pre-threshold IOU"]   = pre_iou_mean_stddev
        metrics["Match IOU"]           = iou_mean_stddev
        metrics["Crashes"]          = "before:{}, after:{}".format(self.m["overlap_gt"],self.m["overlap_rec"])
        metrics["Crashed IDs"]      = self.m["overlap_rec_ids"]
        metrics["Spacing before"]          = spacing_gt_mean_stdev
        metrics["Spacing after"]          = spacing_rec_mean_stdev
        
        # if self.recmode == "rec":
        # Compute average detection metrics in various spaces
        
        state = torch.stack(self.m["state_err"])
        state_mean_stddev = np.nanmean(state,axis = 0), np.nanstd(state,axis = 0)
        # 
        metrics["Width precision"]     = state_mean_stddev[0][3],state_mean_stddev[1][3]
        metrics["Height precision"]    = state_mean_stddev[0][4],state_mean_stddev[1][4]
        metrics["Length precision"]    = state_mean_stddev[0][2],state_mean_stddev[1][2]
        metrics["Velocity precision"]  = state_mean_stddev[0][6],state_mean_stddev[1][6]
        metrics["X precision"]         = state_mean_stddev[0][0],state_mean_stddev[1][0]
        metrics["Y precision"]         = state_mean_stddev[0][1],state_mean_stddev[1][1]
            
       
        
        if hasattr(self, "metrics"):
            self.metrics = dict(list(self.metrics.items()) + list(metrics.items()))
        else:
            self.metrics = metrics
        
    def print_metrics(self):
        print("\n")

        for name in self.metrics:
            if "Matched IDs" in name: 
                continue
            try: 
                unit = self.units[name]
                print("{:<30}: {:.2f}{} avg., {:.2f}{} st.dev.".format(name,self.metrics[name][0],unit,self.metrics[name][1],unit))
            except:
                try: 
                    print("{:<30}: {:.3f}".format(name,self.metrics[name]))
                except:
                    print("{:<30}: {}".format(name,self.metrics[name]))


    def visualize(self):
        # work in space coords
        # convert back to meter
        if self.units["df"] == "ft":
            cols_to_convert = ["fbr_x",	"fbr_y","fbl_x"	,"fbl_y","bbr_x","bbr_y","bbl_x","bbl_y", "speed","x","y","width","length","height"]
            self.rec[cols_to_convert] = self.rec[cols_to_convert] / 3.281
            self.gt[cols_to_convert] = self.gt[cols_to_convert] / 3.281
            self.units["df"] = "m"
            
        # get speed and acceleration
        if self.gtmode == "gt":
            self.gt = utils.img_to_road(self.gt, self.tf_path, self.camera_name)
            self.gt["x"] = (self.gt["bbr_x"]+self.gt["bbl_x"])/2
            self.gt["y"] = (self.gt["bbr_y"]+self.gt["bbl_y"])/2
            self.gt = utils.calc_dynamics(self.gt)
        
        # plot histogram of spacing
        spacing_gt = np.array(self.m["space_gap_gt"])
        spacing_rec = np.array(self.m["space_gap_rec"])
        
        bins=np.histogram(np.hstack((spacing_gt,spacing_rec)), bins=40)[1]
        bw = bins[1]-bins[0]
        fig, ax1 = plt.subplots(1, 1)
        ax1.hist(spacing_gt, bins = bins, density = True, weights = [bw]*len(spacing_gt), facecolor='r', alpha=0.75, label="before")
        ax1.hist(spacing_rec, bins = bins,  density = True, weights = [bw]*len(spacing_rec),facecolor='g', alpha=0.75, label="after")
        ax1.set_xlabel("Spacing ({})".format(self.units["df"]))
        ax1.set_ylabel('Probability')
        ax1.set_title('Spacing distribution')
        ax1.grid()
        ax1.legend()
        
        
        # plot rectification score distribution
        if hasattr(self, "m") and "trajectory_score" in self.m and self.recmode=="rec":
            plt.figure()
            scores = list(self.m["trajectory_score"].values())
            n, bins, patches = plt.hist(scores, 50, facecolor='g', alpha=0.75)
            plt.xlabel('Trajectory score')
            plt.ylabel('ID count')
            plt.title('Trajectory score distribution')
            plt.grid(True)
            plt.show()
        
        gt_groups = self.gt.groupby("ID")
        rec_groups = self.rec.groupby("ID")
        
        # valid frames distribution
        gt_valid_frames = []
        rec_valid_frames = []
        for _,group in gt_groups:
            gt_valid_frames.append(group.count().bbrx)
        for _,group in rec_groups:
            rec_valid_frames.append(group.count().bbrx)
        bins=np.histogram(np.hstack((gt_valid_frames,rec_valid_frames)), bins=40)[1]
        bw = bins[1]-bins[0]
        fig, ax1 = plt.subplots(1, 1)
        ax1.hist(gt_valid_frames, bins = bins, density = True, weights = [bw]*len(gt_valid_frames), facecolor='r', alpha=0.75, label="before")
        ax1.hist(rec_valid_frames, bins = bins,  density = True, weights = [bw]*len(rec_valid_frames),facecolor='g', alpha=0.75, label="after")
        ax1.set_xlabel("# Valid meas per track")
        ax1.set_ylabel('Probability')
        ax1.set_title('Valid measurements distribution')
        ax1.grid()
        ax1.legend()
        
        
        # plot IDs above threshold
        if hasattr(self, "m") and "ids > score" in self.m and self.recmode =="rec":
            for i,rec_id in enumerate(self.m["ids > score"][:5]):
                plt.figure()
                if rec_id in self.metrics['Matched IDs rec'] :
                    gt_car = gt_groups.get_group(self.metrics['Matched IDs rec'][rec_id][0])
                    rec_car = rec_groups.get_group(rec_id)
                    vis.plot_track_compare(gt_car, rec_car)
                    plt.title('Above score threshold:{}'.format(rec_id))
                
        # plot crashed IDs
        # if hasattr(self, "m") and "overlap_rec_ids" in self.m:
        #     count = 0
        #     for id1, id2 in self.m["overlap_rec_ids"]:
        #         if count > 5:
        #             break
        #         plt.figure()
        #         car1 = rec_groups.get_group(id1)
        #         car2 = rec_groups.get_group(id2)
        #         vis.plot_track_compare(car1,car2)
        #         plt.title("Crashed IDs in rec {}-{}".format(id1,id2))
        #         count += 1

        # plot speed distribution before and after 
        # if self.recmode=="rec":
        plt.figure() # speed
        rec_speed = self.rec.speed.values
        gt_speed = self.gt.speed.values
        gt_speed = gt_speed[~np.isnan(gt_speed)]
        # bins=np.histogram(np.hstack((gt_speed,rec_speed)), bins=40)[1]
        # _,_,_ = plt.hist(gt_speed, bins = bins, density = True, facecolor='r', alpha=0.75, label="before")
        _,_,_ = plt.hist(rec_speed, bins = 40, density = True, facecolor='g', alpha=0.75, label="after")
        plt.xlim([0,60])
        plt.title("Speed distribution")
        plt.xlabel("Speed ({}/s)".format(self.units["df"]))
        
        # get IDs for slow cars
        # speed_mean, speed_std = np.nanmean(rec_speed), np.std(rec_speed)
        plt.figure() # acceleration
        rec_accel = self.rec.acceleration.values
        gt_accel = self.gt.acceleration.values
        gt_accel = gt_accel[~np.isnan(gt_accel)]
        # bins=np.histogram(np.hstack((gt_accel,rec_accel)), bins=40)[1]
        # _,_,_ = plt.hist(gt_accel, bins = bins, facecolor='r', alpha=0.75, label="before")
        _,_,_ = plt.hist(rec_accel, bins = 40, facecolor='g', alpha=0.75, label="after")
        # plt.xlim([-10,10])
        plt.title("Acceleration distribution")
        plt.xlabel("Acceleration ({}/s2)".format(self.units["df"]))
            
        # plot lane distribution
        plt.figure()
        self.gt = utils.assign_lane(self.gt)
        self.rec = utils.assign_lane(self.rec)
        
        width = 0.3
        x1 = self.gt.groupby('lane').ID.nunique() # count unique IDs in each lane
        plt.bar(x1.index-0.1,x1.values,color = "r", label="before",width = width)
        x2 = self.rec.groupby('lane').ID.nunique() # count unique IDs in each lane
        plt.bar(x2.index+0.1,x2.values,color = "g", label="after", width=width)
        
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Lane index')
        plt.ylabel('ID count')
        plt.title('Lane distribution')
        plt.legend()
        
        # plot time space diagram (4 lanes +1 direction)
        plt.figure()
        lanes = [1,2,3,4]
        colors = ["blue","orange","green","red"]
        for i,lane_idx in enumerate(lanes):
            lane = self.gt[self.gt['lane']==lane_idx]
            groups = lane.groupby('ID')
            j = 0
            for carid, group in groups:
                x = group['Frame #'].values
                y1 = group['bbr_x'].values
                y2 = group['fbr_x'].values
                plt.fill_between(x,y1,y2,alpha=0.5,color = colors[i], label="lane {}".format(lane_idx) if j==0 else "")
                j += 1
            try:
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            except:
                pass
            plt.xlabel('Frame #')
            plt.ylabel('x ({})'.format(self.units["df"]))
            plt.title('Time-space diagram (before)')

        plt.figure()
        for i,lane_idx in enumerate(lanes):
            lane = self.rec[self.rec['lane']==lane_idx]
            groups = lane.groupby('ID')
            j = 0
            for carid, group in groups:
                x = group['Frame #'].values
                y1 = group['bbr_x'].values
                y2 = group['fbr_x'].values
                plt.fill_between(x,y1,y2,alpha=0.5,color = colors[i], label="lane {}".format(lane_idx) if j==0 else "")
                j += 1
            try:
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            except:
                pass
            plt.xlabel('Frame #')
            plt.ylabel('x ({})'.format(self.units["df"]))
            plt.title('Time-space diagram (after)')
        
        # plot fragments
        for fragments in self.metrics["Fragments"]:
            temp = self.rec[self.rec['ID'].isin(fragments)]
            # plot x
            plt.figure()
            colors = ["blue","orange","green","red"]
            groups = temp.groupby('ID')
            j = 0
            for carid, group in groups:
                x = group['Frame #'].values
                y1 = group['bbr_x'].values
                y2 = group['fbr_x'].values
                plt.fill_between(x,y1,y2,alpha=0.5,color = colors[j%4], label="id {}".format(carid))
                j += 1
            try:
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            except:
                pass
            plt.xlabel('Frame #')
            plt.ylabel('x (m)')
            plt.title('Time-space diagram for fragments')  
            
            # plot y
            plt.figure()
            colors = ["blue","orange","green","red"]
            groups = temp.groupby('ID')
            j = 0
            for carid, group in groups:
                x = group['Frame #'].values
                y1 = group['bbr_y'].values
                y2 = group['bbl_y'].values
                plt.fill_between(x,y1,y2,alpha=0.5,color = colors[j%4], label="id {}".format(carid))
                j += 1
            try:
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            except:
                pass
            plt.xlabel('Frame #')
            plt.ylabel('bb_y (m)')
            plt.title('Time-space diagram for fragments')  
            
            
if __name__ == "__main__":
    
    camera_name = "p1c5"
    sequence_idx = 0
    sequence = None
    
    # 0616-dataset-alpha (with ground truth, 3D tracking)
    gt = r"E:\I24-postprocess\0616-dataset-alpha\FOR ANNOTATORS\rectified_{}_{}_track_outputs_3D.csv".format(camera_name,sequence_idx)
    # gtda = r"E:\I24-postprocess\0616-dataset-alpha\FOR ANNOTATORS\p1c2_0_gtda.csv"
    # gt = r"E:\I24-postprocess\0616-dataset-alpha\FOR ANNOTATORS\p1c24_gt.csv"
    raw = r"E:\I24-postprocess\0616-dataset-alpha\3D tracking\{}_{}_3D_track_outputs.csv".format(camera_name,sequence_idx)
    # DA = r"E:\I24-postprocess\0616-dataset-alpha\3D tracking\DA\{}_{}.csv".format(camera_name,sequence_idx) 
    DA = r"E:\I24-postprocess\0616-dataset-alpha\3D tracking\DA\{}_tsmn.csv".format(camera_name)
    rectified = r"E:\I24-postprocess\0616-dataset-alpha\3D tracking\rectified\{}_{}.csv".format(camera_name,sequence_idx) 
    gt_path = gt
    rec_path = DA
    
    # other paths
    vp_file = r"C:\Users\wangy79\Documents\I24_trajectory\manual-track-labeler-main\DATA\vp\{}_axes.csv".format(camera_name)
    point_file = r"C:\Users\wangy79\Documents\I24_trajectory\manual-track-labeler-main\DATA\tform\{}_im_lmcs_transform_points.csv".format(camera_name)
    tf_path = r"C:\Users\wangy79\Documents\I24_trajectory\manual-track-labeler-main\DATA\tform"
    sequence = r"E:\I24-postprocess\0616-dataset-alpha\Raw Video\{}_{}.mp4".format(camera_name,sequence_idx)
    # sequence = r"E:\I24-postprocess\0806-CIRCLES\raw video\record_51_{}_00000.mp4".format(camera_name)
    
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
        "match_iou":0.3,
        "sequence":sequence,
        "gtmode": "gt" , # "gt", "raw", "da", "rec"
        "recmode": "da",
        "score_threshold": 3
        }
    
    ev = MOT_Evaluator(gt_path,rec_path,tf_path, camera_name, hg, params = params)
    ev.evaluate()
    if params["recmode"] == "rec":
        ev.score_trajectory()
    ev.evaluate_tracks()
    ev.print_metrics()
    # ev.visualize()
    
    # save as pickle
    # filehandler = open("{}_{}_{}.pkl".format(camera_name,sequence_idx,params["recmode"]), 'wb') 
    # pickle.dump(ev, filehandler)
    
