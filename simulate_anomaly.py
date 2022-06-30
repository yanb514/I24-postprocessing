#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:31:02 2022

@author: yanbing_wang

Given leader(s) trajectories of each platoon
simulate the platoon downstream one by one using a car-following model

    
Objective:
    simulate some normal/abnomal followers, no lane change
"""

# for each follower do the following

dt = 0.1

for leader in platoon:
    
    pl,vl,al = xxx # read the trajectory of the platoon leader
    
    for car in num_car_in_platoon:
        
        # Initialize position, v, a for this car
        p = [p0] # position with initialized position
        v = [v0] # velocity
        a = [a0] # acceleration
        
        # simulate this car's trajectory
        for i in range(len(pl)): # for each timestep
            a_next = alpha * (vl[i] - v[i]) # choose alpha differently for each car to create normal/abnormal follower
            v_next = v[i] + a_next * dt
            p_next = p[i] + v_next * dt
            
        # save this car's trajectoy somewhere
        save(p, v, a)
        
        # continue with this car's follower, now this car becomes the new leader
        pl, vl, al = p, v, a
            
        





