#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt
YesFileNames = []
NoFileNames = []
bias = 0.5
mu = 0.00001
output = 0
weights = np.random.rand(768,1)
batchesT = 250
batchesV = 100
epochs = 15000
AccuracyData = np.zeros(epochs)
ErrorData = np.zeros(epochs)
accuracyT = 0
accuracyV = 0
errorSum = 0
errorV = 0
print('learning rate: ',mu,'training batch size: ',batchesT)
print('validation batch size: ',batchesV,'number of epochs',epochs)
#Load data from files
print('Loading data...')
YesData = np.genfromtxt('FeatureMatrixYes.txt')
NoData = np.genfromtxt('FeatureMatrixNo.txt')
YesFileNames = np.genfromtxt('AuroraNamesYes.txt',dtype = None,delimiter = '\n')
NoFileNames = np.genfromtxt('AuroraNamesNo.txt',dtype = None,delimiter = '\n')
YesData = np.insert(YesData,0,1,axis = 1)
NoData = np.insert(NoData,0,0,axis = 1)

#Initialize training and validation sets
print('Initializing data...')
yesDataSize = YesData.shape[0]
noDataSize = NoData.shape[0]
yesTrainingCutoff = math.floor(YesData.shape[0] * 0.8)
noTrainingCutoff = math.floor(NoData.shape[0] * 0.8)

np.random.shuffle(YesData)
np.random.shuffle(NoData)

#split data into 80%/20% portions for training/validation
YesDataTraining = YesData[0:yesTrainingCutoff,:]
YesDataValidation = YesData[yesTrainingCutoff:yesDataSize,:]

NoDataTraining = NoData[0:noTrainingCutoff,:]
NoDataValidation = NoData[noTrainingCutoff:noDataSize,:]

YesFileNamesTraining = YesFileNames[0:yesTrainingCutoff]
YesFileNamesValidation = YesFileNames[yesTrainingCutoff:yesDataSize]

NoFileNamesTraining = NoFileNames[0:noTrainingCutoff]
NoFileNamesValidation = NoFileNames[noTrainingCutoff:noDataSize]


TrainingData = np.insert(YesDataTraining,YesDataTraining.shape[0],NoDataTraining,axis = 0)
ValidationData = np.insert(YesDataValidation,YesDataValidation.shape[0],NoDataValidation,axis = 0)

np.random.shuffle(TrainingData)
np.random.shuffle(ValidationData)

print('Running perceptron...')
#Train perceptron
#EPOCHS
for i in range(0,epochs):
    indicesTrain = np.random.randint(TrainingData.shape[0], size=batchesT)
    indicesValidate = np.random.randint(ValidationData.shape[0], size=batchesV)
    TrainingBatch = TrainingData[indicesTrain,:]
    ValidationBatch = ValidationData[indicesValidate,:]
    accuracyT = 0
    accuracyV = 0
    
    #TRAINING
    for j in range(0,batchesT):    
        FeatureVector = np.asmatrix(TrainingBatch[j,1:769])
        FeatureVectorT = np.transpose(FeatureVector)
        y = TrainingBatch[j,0]
        yhat = np.matmul(FeatureVector,weights) + bias   
        if yhat > 0:
            output = 1
        else:
            output = 0  
        error = yhat - y
        error = np.asscalar(error)
        if output == y:
            accuracyT = accuracyT + 1          
        else:
            #Update weights
            bias = bias - mu * error 
            adjust = mu * error
            weightsTemp = weights - adjust*FeatureVectorT 
            weights = weightsTemp
    AccuracyData[i] = accuracyT / batchesT
    
    #VALIDATION
    for k in range(0,batchesV):
        FeatureVector = np.asmatrix(ValidationBatch[k,1:769])
        y = ValidationBatch[k,0]
        yhat = np.matmul(FeatureVector,weights) + bias
        if yhat > 0:
            output = 1
        else:
            output = 0 
        if output == y:
            accuracyV = accuracyV + 1  
    ErrorData[i] = 1 - (accuracyV / batchesV)

plt.figure(1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy per Batch')
plt.title('Perceptron Learning (Training)')
plt.plot(AccuracyData)
plt.figure(2)
plt.xlabel('Epoch')
plt.ylabel('Error per Batch')
plt.title('Perceptron Error (Validation)')
plt.plot(ErrorData)
print('Final Error: ',ErrorData[epochs - 1] * 100,'%')

