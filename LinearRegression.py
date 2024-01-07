#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:27:49 2018

@author: henrithomas
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
mu = .005
b1 = 0.03
b0 = -0.5
b0_best = 1.0
b1_best = 1.0
error_global = 10
b0_temp = 0.0
b1_temp = 0.0
accuracy_best = 0.0
epochs = 1000
batch_size = 10

def normalize(arr):
    arr = arr.astype(float)
    arr_min = np.amin(arr)
    arr_max = np.amax(arr)
    for m in range(0,arr.size):
        arr_temp = (arr[m] - arr_min) / (arr_max - arr_min) 
        arr[m] = arr_temp
    return arr
    
def b0_sum(batch_x,batch_y,b_0,b_1):
    b0_sum = 0.0
    for j in range(0,batch_size):
        b0_sum = b0_sum - (b_0 + b_1 * batch_x[j] - batch_y[j])**2
    return b0_sum
    
def b1_sum(batch_x,batch_y,b_0,b_1):
    b1_sum = 0.0
    for k in range(0,batch_size):
        b1_sum = b1_sum - batch_x[k] * (b_0 + b_1 * batch_x[k] - batch_y[k])**2
    return b1_sum

def error_sum(x_dat,y_dat,b_0,b_1):
    error_sum = 0.0
    for l in range(0,batch_size):
        error_sum = error_sum + (b_0 + b_1 * x_dat[l] - y_dat[l])**2
    return error_sum 

plt.xlabel('Effort')
plt.ylabel('Harvest')
plt.title('Linear Regression: Fisherman Effort vs. Harvest')

data = pd.read_csv('drift-har-eff.csv')

rows = data.shape [0]
cols = data.shape [1]

data = data.values
data = data[np.arange(0,rows),:]
data = delete(data, s_[0:1], axis = 1)

X = data[:, 0]
Y = data[:, 1]

learning = np.zeros(epochs)
X = normalize(X)
Y = normalize(Y)

plt.xlim(-.1,max(X) + .1)
plt.ylim(-.6,max(Y) + .1)

plt.figure(1)
plt.scatter(X, Y)
plt.plot (X, b1 * X + b0)
error = (1/ (2 * batch_size)) * error_sum(X,Y,b0,b1)
print('b0:',b0,'b1:',b1,'error:',error)

for i in range(0,epochs):
    #sample a batch from the data
    batch = data[np.random.randint(data.shape[0], size=batch_size), :]
    batch_X = normalize(batch[:,0])
    batch_Y = normalize(batch[:,1])
    b0_temp = b0 - mu * (1 / batch_size) * b0_sum(batch_X,batch_Y,b0,b1)
    b1_temp = b1 - mu * (1 / batch_size) * b1_sum(batch_X,batch_Y,b0,b1)
    error = (1/ (2 * batch_size)) * error_sum(X,Y,b0,b1)
    if error < error_global:    
        b0 = b0_temp
        b1 = b1_temp
        error_global = error
        learning[i] = error_global
        
print('b0:',b0,'b1:',b1,'error:',error_global)
plt.plot (X, b1 * X + b0)
plt.figure(2)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Iteration vs. Error - Learning Rate: 0.005')
plt.plot(learning)
