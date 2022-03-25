#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 13:06:23 2022

@author: yanbing_wang
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Generate some non-uniformly sampled data with missing values removed
N = 100
freq = 30
t = np.arange(0, N/freq, 1/freq) + np.random.normal(0,1/freq/10,N)
dt = np.diff(t)
x = 30* np.cos(t/30) + np.random.normal(0,0.01,N)
y = 30* np.sin(t/30) + np.random.normal(0,0.01,N)
data = {'timestamp': pd.to_datetime(t, unit='s'), 'x_position': x, 'y_position': y}

# Read to dataframe and resample
df = pd.DataFrame(data, columns=data.keys()) 
df.iloc[10:50] = np.nan 
df = df.dropna()
df = df.set_index('timestamp') 
df = df.resample('0.033333333S').mean() # this is 33.33Hz

df.index = df.index.values.astype('datetime64[ns]').astype('int64')*1e-9

#%%
fig, ax = plt.subplots()
ax.scatter(t,x, label='original signal')
ax.scatter(df.index, df.x_position, label='resampled signal')
fig.autofmt_xdate()
ax.legend()
ax.set_xlim([min(t), max(t)])
