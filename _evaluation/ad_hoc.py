#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 08:40:52 2022

@author: yanbing_wang
"""
import matplotlib.pyplot as plt
import json
from i24_database_api.db_reader import DBReader
from i24_database_api.db_writer import DBWriter
from bson.objectid import ObjectId
import numpy as np
from sklearn import linear_model


def plot_traj_attr(veh_id, dbr, attr_names):
    num = len(attr_names)
    fig, axs = plt.subplots(1, num, figsize=(3*num, 3))
    
    traj = dbr.find_one("_id", veh_id)
    for i, attr in enumerate(attr_names):
        try:
            if attr == "variance":
                series = [vector[2] for vector in traj["variance"]]
                axs[i].scatter(traj["timestamp"], series, s=0.5, label = attr)
            else:
                axs[i].scatter(traj["timestamp"], traj[attr], s=0.5, label = veh_id)
            axs[i].set_title("time v {}".format(attr))
            axs[0].legend()
        except Exception as e:
            print(e)
            pass
        
    
def plot_traj(veh_ids, dbr):
    '''
    Plot fragments with the reconciled trajectory (if col2 is specified)

    Parameters
    ----------
    fragment_list : list of ObjectID fragment _ids
    rec_id: ObjectID (optional)
    '''
    
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    
    for f_id in veh_ids: 
        f = dbr.find_one("_id", f_id)
        print(f_id, len(f["timestamp"]))
        l = str(f_id)[10:] + f["flags"][0]
        conf = np.array(f["detection_confidence"])
        axs[0].scatter(f["timestamp"], f["x_position"], s=conf*10, label=l)
        axs[1].scatter(f["timestamp"], f["y_position"], s=conf*10, label=l)
        axs[2].scatter(f["x_position"], f["y_position"], s=conf*10, label=l)
        axs[3].scatter(f["timestamp"], conf , s=0.5, label=l)

    axs[0].set_title("time v x")
    axs[1].set_title("time v y")
    axs[2].set_title("x v y")
    axs[3].set_title("time v onfidence")
    axs[0].legend()
    return axs
        
       
def plot_stitched(rec_id, rec, raw):
    '''
    plot rec_id and the fragments it stitched together
    '''
    rec_traj = rec.find_one("_id", rec_id)
    print(rec_traj["fragment_ids"])
    axs = plot_traj(rec_traj["fragment_ids"], raw)
    axs[0].scatter(rec_traj["timestamp"], rec_traj["x_position"], s=1, c='k')
    axs[1].scatter(rec_traj["timestamp"], rec_traj["y_position"],  s=1, c='k')
    axs[2].scatter(rec_traj["x_position"], rec_traj["y_position"],  s=1, c='k')
    axs[3].scatter(rec_traj["timestamp"], rec_traj["detection_confidence"],  s=1, c='k')
    
    return
      
def ransac_fit(veh_id, dbr):
    '''
    remove by confidence threshold
    remove by ransac outlier mask (x-axis)
    get total mask (both lowconf and outlier)
    apply ransac again on y-axis
    save fitx, fity and tot_mask
    
    return False if tot_mask rate is higher than 50 percent (do not consider this track)
    else True otherwise
    '''
    residual_threshold_x = 5 # tolerance in x
    residual_threshold_y = 1 # tolerance in y
    conf_threshold = 0.5
    
    track = dbr.find_one("_id", veh_id)
    length = len(track["timestamp"])
    
    # get confidence mask
    lowconf_mask = np.array(np.array(track["detection_confidence"]) < conf_threshold)
    highconf_mask = np.logical_not(lowconf_mask)
    
    # fit x only on highconf
    ransacx = linear_model.RANSACRegressor(residual_threshold=residual_threshold_x)
    X = np.array(track["timestamp"]).reshape(1, -1).T
    x = np.array(track["x_position"])
    ransacx.fit(X[highconf_mask], x[highconf_mask])
    fitx = [ransacx.estimator_.coef_[0], ransacx.estimator_.intercept_]
    inlier_mask = ransacx.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask) # mask if True
    
    # total mask (filtered by both outlier and by low confidence)
    mask1 = np.arange(length)[lowconf_mask] # all the bad indices
    mask2 = np.arange(length)[highconf_mask][outlier_mask]
    mask = np.concatenate((mask1, mask2))
    bad_ratio = len(mask)/length
    print("bad rate: {}".format(bad_ratio))
    # if bad_ratio > 0.5:
    #     return False
    
    
    
    # fit y only on mask
    ransacy = linear_model.RANSACRegressor(residual_threshold=residual_threshold_y)
    y = np.array(track["y_position"])
    ransacy.fit(X[highconf_mask][inlier_mask], y[highconf_mask][inlier_mask])
    fity = [ransacy.estimator_.coef_[0], ransacy.estimator_.intercept_]
    print(fitx)
    print(fity)
    
    
    
    # Predict data of estimated models
    line_x_ransac = ransacx.predict(X)
    line_y_ransac = ransacy.predict(X)
    # Compare estimated coefficients
    # print("Estimated coefficients (true, linear regression, RANSAC):")
    print("score x", ransacx.estimator_.score(X, x))
    print("score y", ransacy.estimator_.score(X, y))
    # print("y outlier", np.logical_not(ransacy.inlier_mask_))
    lw = 2
    axx = plt.subplot(121)
    axx.scatter(
        X[highconf_mask][inlier_mask], x[highconf_mask][inlier_mask], color="yellowgreen", marker=".", label="Inliers"
    )
    axx.scatter(
        X[highconf_mask][outlier_mask], x[highconf_mask][outlier_mask], color="gold", marker=".", label="Outliers"
    )
    
    axx.scatter(
        X[mask], x[mask], color="k", marker="o", alpha=0.1, label="tot mask"
    )
    axx.plot(
        X,
        line_x_ransac,
        color="cornflowerblue",
        linewidth=lw,
        label="RANSAC regressor",
    )
    axx.legend(loc="lower right")
    
    axy = plt.subplot(122)
    axy.scatter(
        X, y, color="yellowgreen", marker=".", label="original"
    )

    axy.scatter(
        X[mask], y[mask], color="k", marker="o", alpha=0.1, label="tot mask"
    )
    axy.plot(
        X,
        line_y_ransac,
        color="cornflowerblue",
        linewidth=lw,
        label="RANSAC regressor",
    )
    axy.legend(loc="lower right")
    plt.show()
    return True


    #%%
if __name__ == "__main__":
    with open('config.json') as f:
        config = json.load(f)
    
    trajectory_database = "trajectories"
    collection = "morose_panda--RAW_GT1"
    
    raw = DBReader(config, host = config["host"], username = config["username"], password = config["password"], port = config["port"], database_name = trajectory_database, collection_name=collection)
    rec = DBReader(config, host = config["host"], username = config["username"], password = config["password"], port = config["port"], database_name = trajectory_database, collection_name=collection+"_reconciled")
    
    #%% examine a single fragment
    veh_id = ObjectId("62e0194627b64c6330546016")
    plot_traj_attr(veh_id, raw, ["x_position", "y_position", "detection_confidence"])
    ransac_fit(veh_id, raw)
    
    #%% plot framgnets together
    # fgmt_ids = [ ObjectId('62e0193027b64c6330546003'), ObjectId('62e0194627b64c6330546016'), ObjectId('62e0195327b64c6330546026'),  ObjectId('62e0196227b64c6330546035')] # should be one
    # fgmt_ids = [ObjectId('62e018c427b64c6330545fa6'), ObjectId('62e0190427b64c6330545fe6'), ObjectId('62e0190427b64c6330545fe7')] # they should stitch to two
    fgmt_ids = [ObjectId('62e0198927b64c6330546059'), ObjectId('62e019a827b64c633054607d'), ObjectId('62e019be27b64c6330546096')] # they should stitch to two
    
    plot_traj(fgmt_ids, raw)
    
    #%% plot from a reconciled trajectory and its fragments
    rec_id = ObjectId("62e157022b084202528584bf")
    plot_stitched(rec_id, rec, raw)
    
    
    
    
    
    