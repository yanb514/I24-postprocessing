# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:07:38 2021

@author: wangy79
"""
import numpy as np
import utils
from shapely.geometry import Polygon
import pandas as pd
import utils_vis as vis
import scipy
import matplotlib.pyplot as plt
import utils_evaluation as ev
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

def my_nll():
    return

def nll_grid(vx,vy, alpha_x,alpha_y, T, std):
    '''
    varying alpha and T
    '''
    nll = np.zeros((len(alpha_x),len(T)))
    for i, alpha in enumerate(alpha_x):
        for j, t in enumerate(T):
            input = torch.tensor([vx*t + std*np.sqrt((alpha*t)), vy*t+std*np.sqrt((alpha_y[i]*t))])
            target = torch.tensor([vx*t, vy*t])
            var = torch.tensor([alpha*t, alpha_y[i]*t])
            nll[i,j] = loss(input, target, var)+10e-6 # assumes iid, loss([x,y]) = loss(x)+loss(y)
    return nll

def nll_grid_2(vx,vy, alpha_x,alpha_y, T, dev_x, dev_y):
    '''
    dev_x = abs(x-mu_x)
    varying dev_x and T, fix alpha_x, alpha_y
    '''
    nll = np.zeros((len(dev_x),len(T)))
    for i, _ in enumerate(dev_x):
        for j, t in enumerate(T):
            input = torch.tensor([vx*t+dev_x[i], vy*t+dev_y[i]])
            target = torch.tensor([vx*t, vy*t])
            var = torch.tensor([alpha_x*t, alpha_y*t])
            nll[i,j] = loss(input, target, var)+10e-6
    return nll

def heatmap(nll, std):
    nll = np.flip(nll, axis=0)
    # nll = np.log(nll)
    # nll[np.isnan(nll)] = 0
    fig, ax = plt.subplots()
    im = ax.imshow(nll, cmap='hot')
    ax.set_title("Negative log likelihood {} std".format(std),fontsize=10)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    ax.set_xticks(list(np.arange(len(T))))
    xlabel = [int(x/30) for x in T]
    ax.set_xticklabels(xlabel,fontsize=10)
    ax.set_xlabel("Sec of missing detections")
    ax.set_yticks(list(np.arange(len(alpha_x))))
    ylabel = ["{:.2f}/{:.2f}".format(alpha_x[i],alpha_y[i]) for i in range(len(alpha_x))]
    ax.set_yticklabels(ylabel[::-1],fontsize=10)
    ax.set_ylabel("alpha_x/y")
    fig.colorbar(im, cax=cax)
    
    plt.show()
            
def heatmap_2(nll):
    nll = np.flip(nll, axis=0)
    nll = np.log(nll)
    fig, ax = plt.subplots()
    im = ax.imshow(nll, cmap='hot')
    ax.set_title("nll heatmap",fontsize=10)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    ax.set_xticks(list(np.arange(len(T))))
    xlabel = [int(x/30) for x in T]
    ax.set_xticklabels(xlabel,fontsize=10)
    ax.set_xlabel("Sec of missing detections")
    ax.set_yticks(list(np.arange(len(dev_x))))
    ylabel = ["{:.2f}".format(x) for x in dev_x]
    ax.set_yticklabels(ylabel[::-1],fontsize=10)
    ax.set_ylabel("Deviation x")
    fig.colorbar(im, cax=cax)
    
    plt.show()
    
if __name__ == "__main__":
    loss = torch.nn.GaussianNLLLoss(full=True)
    vx = 1
    vy = 0
    std = 0
    alpha_x = np.arange(0.01,0.3,0.03)
    alpha_y = np.arange(0.01,0.3,0.03)*1/3

    T = np.arange(1,300,30) # missing detection frames
    nll = nll_grid(vx, vy, alpha_x, alpha_y, T, std)
    heatmap(nll, std)
        