# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 16:49:19 2021

@author: wangy79
"""
from utils import *
import importlib
import utils
importlib.reload(utils)
import pandas as pd
import utils_optimization as opt
importlib.reload(opt)
import data_association as da
importlib.reload(da)


# read & rectify each camera df individually
data_path = pathlib.Path().absolute().joinpath('../3D tracking')
tform_path = pathlib.Path().absolute().joinpath('../tform')

# assign unique IDs to objects in each camera after DA on each camera independently
df2 = utils.read_data(data_path.joinpath('p1c2_small_stitch.csv'))
df3 = utils.read_data(data_path.joinpath('p1c3_small_stitch.csv'))
df3 = da.assign_unique_id(df2,df3)

# %% Make time-space diagram
df = pd.concat([df2, df3])
df = df[(df['bbr_y']>=3.7*1) & (df['bbr_y']<=3.7*2)]
groups = df.groupby('ID')
for carid, group in groups:
    x = group['Frame #'].values
    y1 = group['bbr_x'].values
    y2 = group['fbr_x'].values
#     plt.scatter(x,y2,s=1,label=carid)
    plt.fill_between(x,y1,y2,label=carid)
#plt.legend()
plt.xlabel('Frame #')
plt.ylabel('x')