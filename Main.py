__author__ = 'chaomeng'

import csv
import io
import cv2
from Cluster_feature import build_cluster
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

def get_label(image):
    expression = image.split('.')[1]
    if "AN" in expression:
        return 1
    if "DI" in expression:
        return 2
    if "FE" in expression:
        return 3
    if "HA" in expression:
        return 4
    if "NE" in expression:
        return 5
    if "SA" in expression:
        return 6
    if "SU" in expression:
        return 7

def check_rate(predicts, testlabels):
    count = 0
    for i in range(len(testlabels)):
        if predicts[i] == testlabels[i]:
            count += 1
    return count

imageList = open('resizeList.txt', 'r')
trainImage = []
testImage = []
for line in imageList:
    oneImage = line.strip('\n')
    parts = oneImage.split('.')
    if '1' in parts[1] or '2' in parts[1]:
        trainImage.append(oneImage)
    elif '3' in parts[1]:
        testImage.append(oneImage)

trainingFeatures = []
trainlabels = []
testData = []
testlabels = []
K=3
featureParam = 35
for i in range(len(trainImage)):
    feature = build_cluster(trainImage[i], featureParam, K)
    trainingFeatures.append(feature)
    trainlabels.append(get_label(trainImage[i]))

for i in range(len(testImage)):
    feature = build_cluster(testImage[i], featureParam, K)
    testData.append(feature)
    testlabels.append(get_label(testImage[i]))

# classifer = OneVsRestClassifier(LinearSVC(random_state=0)).fit(trainingFeatures, trainlabels)
# predicts = classifer.predict(testData)
predicts = OneVsRestClassifier(LinearSVC(random_state=0)).fit(trainingFeatures, trainlabels).predict(testData)
print check_rate(predicts, testlabels)