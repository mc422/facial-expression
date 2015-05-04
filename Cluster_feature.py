__author__ = 'chaomeng'
import cv2
import sys
import numpy as np
from numpy import array
from scipy.cluster.vq import vq, kmeans, whiten
from scipy.spatial.distance import euclidean

colors = []
colors.append((0,255,0))
colors.append((255,0,0))
colors.append((0,0,255))
colors.append((255,255,0))
colors.append((0,255,255))
colors.append((255,0,255))

# return cluster's related data of a single cluster
# @param: keyLocation    keyPoints location cooridanate
# @param: center         keyPoints mean coordinate
# @return:               cluster's information data, used as image feature
def cluster_feature(keyLocation, center):
    size = len(keyLocation)
    newCenter = center.astype(int)
    std = np.std(keyLocation, axis=0).astype(int)
    return [size, newCenter[0], newCenter[1], std[0], std[1]]

# create keypoint cluster for image
# @param: image         image data to cluster
# @param: featureValue  opencv keypoint detect function FastFeatureDetector() parameter
# @param: K             cluster size
# @return               clusters' feature data
def build_cluster(image, featureValue, K):
    img = cv2.imread(image)
    fast = cv2.FastFeatureDetector(featureValue)
    orb = cv2.ORB(180)
    kp = fast.detect(img,None)
    kp, des = orb.compute(img, kp)

    # build keypoints location array for cluster
    locations = np.empty((len(kp),2))
    for i in range(len(kp)):
        loc = array((int(kp[i].pt[0]), int(kp[i].pt[1])))
        locations[i]=loc

    kcenters, distortion  = kmeans(locations, K)
    kcenters = kcenters[kcenters[:,0].argsort()]

    # cluster index: 0: left eye, 1 mouth and nose, 2: right eye
    kpCluster = {i: [] for i in range(K)}
    clusterLoc = {i: [] for i in range(K)}
    for i in range(len(kp)):
        set = 0
        minDis = sys.maxint
        for j in range(K):
            dis = euclidean(locations[i], kcenters[j])
            if dis<minDis:
                set = j
                minDis = dis
        kpCluster[set].append(kp[i])
        clusterLoc[set].append(locations[i])

    imageFeature = [len(kp)]
    for i in range(K):
        clusterFeature = cluster_feature(clusterLoc[i], kcenters[i])
        imageFeature = imageFeature + clusterFeature
    return imageFeature
    #return imageFeature

# partList = open('partOfList.txt', 'r')
# K=3
# fastFeature = 35
# for line in partList:
#     #img = cv2.imread(line.strip('\n'))
#     oneImage = line.strip('\n')
#     imageFeature = build_cluster(oneImage, fastFeature, K)
# print 'end'
