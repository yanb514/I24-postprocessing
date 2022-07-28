#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 23:28:14 2022

@author: yanbing_wang
"""
import numpy as np
import random

# apples
num = 1000
scores = np.random.rand(num,) # uniform
# scores = np.random.normal(1, 1, 1000)

# basket sizes
sizes = random.sample(range(num), 20)
sizes.sort()

# get average score for each basket
basket_scores = []
for i in range(len(sizes)-1):
    tot_score = np.sum(scores[sizes[i]:sizes[i+1]])
    basket_scores.append(tot_score/(sizes[i+1]-sizes[i]))

# get the distribution of basket_score
print("scores: mean: {:.2f}, stdev: {:.2f}".format(np.mean(scores), np.std(scores)))
print("basket_scores: mean: {:.2f}, stdev: {:.2f}".format(np.mean(basket_scores), np.std(basket_scores)))

# throw away percentage
thresh = 0.4
throwaway = 0
for i,bs in enumerate(basket_scores):
    if bs < thresh:
        throwaway += sizes[i+1]-sizes[i]

perc = throwaway / 1000
print("throw away {:.2f} % ".format(perc*100))



