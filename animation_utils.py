from matplotlib import cm
import numpy as np
import math

def getCarColor(speed, maxSpeed, carID) :
    if(carID == 316120) : return 'black'
    elif(carID == 344120) : return 'red'
    elif(carID == 399120) : return 'white'
    else :
        coolwarm = cm.get_cmap('coolwarm_r')
    
        if speed > 34 :
            return coolwarm(0.999)
        else :
            normVal = speed / 34.0
            return coolwarm(normVal)

def restructCoord(frameSnap) :
    for i in range(len(frameSnap)) :
        if frameSnap[i,9] == 1 :  # If car is going left to right
            # Transform the coordinates so that bbr_x and so on are in sync
            # with cars going from right to left
            
            temp = frameSnap[i,0]
            frameSnap[i,0] = frameSnap[i,4]
            frameSnap[i,4] = temp
            
            temp = frameSnap[i,1]
            frameSnap[i,1] = frameSnap[i,5]
            frameSnap[i,5] = temp
            
            temp = frameSnap[i,2]
            frameSnap[i,2] = frameSnap[i,6]
            frameSnap[i,6] = temp
            
            temp = frameSnap[i,3]
            frameSnap[i,3] = frameSnap[i,7]
            frameSnap[i,7] = temp
            
        # Loop to change to feet
        for j in range(0,8) :
            frameSnap[i,j] *= 3.28084
            
        if math.isnan(frameSnap[i,11]) : frameSnap[i,11] = 0
            
def fillBetweenX(xs) :
    # Minor misalignments between the coordinates causes the fill function
    # to fill color in random spaces. Fixing the numbers to be exact.
    temp = list(xs)
    temp[1] = temp[2]
    temp[3] = temp[0]
    newxs = tuple(temp)
    
    return newxs

def fillBetweenY(ys) :
    temp = list(ys)
    temp[1] = temp[0]
    temp[2] = temp[3]
    newys = tuple(temp)
    
    return newys