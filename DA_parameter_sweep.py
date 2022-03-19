# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:07:38 2021

@author: wangy79

"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable


def nll_grid(vx,vy, alpha_x,alpha_y, T, std):
    '''
    varying alpha and T
    alpha: hyperparameter in weiner process, 
    '''
    nll = np.zeros((len(alpha_x),len(T)))
    for i, alpha in enumerate(alpha_x):
        for j, t in enumerate(T):
            input = torch.tensor([vx*t + std*np.sqrt((alpha*t)), vy*t+std*np.sqrt((alpha_y[i]*t))])
            target = torch.tensor([vx*t, vy*t])
            var = torch.tensor([alpha*t, alpha_y[i]*t])
            nll[i,j] = loss(input, target, var)+10e-6 # assumes iid, loss([x,y]) = loss(x)+loss(y)
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

    
if __name__ == "__main__":
    loss = torch.nn.GaussianNLLLoss(full=True)
    vx = -1
    vy = 0.2
    std = 6
    alpha_x = np.arange(0.01,0.3,0.03)
    alpha_y = np.arange(0.01,0.3,0.03)/2

    T = np.arange(1,300,30) # missing detection frames
    nll = nll_grid(vx, vy, alpha_x, alpha_y, T, std)
    heatmap(nll, std)
        