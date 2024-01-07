import cv2
import os
import numpy as np
#working directory
homePath = '/Users/henrithomas/Desktop/School/Spring 2018/Machine Learning'
#path for images with no aurora
pathNo = '/Users/henrithomas/Desktop/School/Spring 2018/Machine Learning/AuroraImages/NoAurora/'
#path for images with aurora
pathYes = '/Users/henrithomas/Desktop/School/Spring 2018/Machine Learning/AuroraImages/YesAurora/'
fNo = open('FeatureMatrixNo.txt','w+') 
fYes = open('FeatureMatrixYes.txt','w+') 
fNameNo = open('AuroraNamesNo.txt','w+')
fNameYes = open('AuroraNamesYes.txt','w+')

FeatureMatrixNo = []
FeatureMatrixYes = []
FilesNo = []
FilesYes = []

def normalize(arr):
    arr = arr.astype(float)
    arr_min = np.amin(arr)
    arr_max = np.amax(arr)
    arr_norm = (arr - arr_min) / (arr_max - arr_min) 
    return arr_norm   
    
def preprocess(directory):
    os.chdir(directory)
    images = [name for name in os.listdir(".") if name.endswith(".jpg")]
    for i in range(0,len(images)):           
        fName = images[i]
        #load image from folder
        img = cv2.imread(fName)
        #calculate individual RGB histograms
        series1 = cv2.calcHist([img],[0],None,[256],[0,256])
        series2 = cv2.calcHist([img],[1],None,[256],[0,256])
        series3 = cv2.calcHist([img],[2],None,[256],[0,256])       
        series1T = np.transpose(series1)
        series2T = np.transpose(series2)
        series3T = np.transpose(series3)
        #create feature vector
        featureVector = np.append(series1T,[series2T,series3T])
        featureVector = np.asarray(featureVector)
        if directory == pathNo:
            FeatureMatrixNo.append(featureVector)
            FilesNo.append(fName)
        else:
            FeatureMatrixYes.append(featureVector)
            FilesYes.append(fName)
        fName = ''

print('Processing images with aurora...')
preprocess(pathYes)
print('Processing images with no aurora...')
preprocess(pathNo)

FeatureMatrixYes = np.asmatrix(FeatureMatrixYes)
FeatureMatrixNo = np.asmatrix(FeatureMatrixNo)
#normalize data
FeatureMatrixNo = normalize(FeatureMatrixNo)
FeatureMatrixYes = normalize(FeatureMatrixYes)

FilesNo = np.asmatrix(FilesNo)
FilesYes = np.asmatrix(FilesYes)
FilesNo = np.transpose(FilesNo)
FilesYes = np.transpose(FilesYes)

os.chdir(homePath)
print('Saving data...')
np.savetxt('FeatureMatrixYes.txt',FeatureMatrixYes)
np.savetxt('FeatureMatrixNo.txt',FeatureMatrixNo)
np.savetxt('AuroraNamesNo.txt',FilesNo,fmt = '%s')
np.savetxt('AuroraNamesYes.txt',FilesYes,fmt = '%s')

fNo.close()
fYes.close()
fNameNo.close()
fNameYes.close()
print('Pre-processing complete.')
