#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 14:57:32 2022

@author: yanbing_wang
"""
import time
import torch
elapsed = 0

for dev in ["cuda:0","cpu"]:
    device = torch.device(dev)
    for n in range(1000):
        x = torch.rand(5000,1000,device = torch.device("cuda:3"))
        y = torch.rand(1000,2000,device = torch.device("cuda:3"))
        start = time.time()
        z = torch.matmul(x,y)
        end = time.time()
        elapsed += end-start
        if n%100 == 0:
            print("On iteration {}".format(n))
print("Average speed on {}: {} multiplications per second".format(dev,1000/elapsed))