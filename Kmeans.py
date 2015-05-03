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

# # import image and extract fast feature
# img = cv2.imread('resize/KM.AN2.18.tiff')
# fast = cv2.FastFeatureDetector(30)
# orb = cv2.ORB(180)
# kp = fast.detect(img,None)
# kp, des = orb.compute(img, kp)
# img1 = cv2.drawKeypoints(img, kp, color=colors[0])
# print len(kp)
#
# # build keypoints location array for cluster
# locations = np.empty((len(kp),2))
# for i in range(len(kp)):
#     loc = array((int(kp[i].pt[0]), int(kp[i].pt[1])))
#     locations[i]=loc
#
# size = len(des)/K
# #centers = array((locations[size], locations[2*size], locations[3*size], locations[4*size], locations[5*size], locations[6*size-1]))
# #centers = array((locations[0], locations[2*size], locations[3*size], locations[4*size-1]))
# centers = array((locations[0], locations[2*size], locations[3*size-1]))
# kcenters, distortion  = kmeans(locations, centers)
#
# kpCluster = {i: [] for i in range(K)}
# for i in range(len(kp)):
#     set = 0
#     minDis = sys.maxint
#     for j in range(K):
#         dis = euclidean(locations[i], kcenters[j])
#         if dis<minDis:
#             set = j
#             minDis = dis
#     kpCluster[set].append(kp[i])
#
# pic = img
# for i in range(K):
#     pic = cv2.drawKeypoints(pic,kpCluster[i],color=colors[i], flags=0)
#     #print len(kpCluster[i])
# cv2.imshow('cluster',pic)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def findNeaghborPoint(keypoints, location):
    minIndex = 0
    minDis = sys.maxint
    for i in range(len(keypoints)):
        dis = euclidean(keypoints[i], location)
        if dis<minDis:
            minIndex = i
            minDis = dis
    return minIndex

def create_cluster_image(image, featureValue, K):
    img = cv2.imread(image)
    fast = cv2.FastFeatureDetector(featureValue)
    orb = cv2.ORB(180)
    kp = fast.detect(img,None)
    kp, des = orb.compute(img, kp)
    print len(kp)

    # build keypoints location array for cluster
    locations = np.empty((len(kp),2))
    for i in range(len(kp)):
        loc = array((int(kp[i].pt[0]), int(kp[i].pt[1])))
        locations[i]=loc

    size = len(des)/K
    #centers = array((locations[size], locations[2*size], locations[3*size], locations[4*size], locations[5*size], locations[6*size-1]))
    #centers = array((locations[0], locations[2*size], locations[3*size], locations[4*size-1]))
    Ncenters = array((locations[0], locations[2*size], locations[3*size-1]))
    kcenters, distortion  = kmeans(locations, K)
    kcenters = kcenters[kcenters[:,0].argsort()]

    kpCluster = {i: [] for i in range(K)}
    for i in range(len(kp)):
        set = 0
        minDis = sys.maxint
        for j in range(K):
            dis = euclidean(locations[i], kcenters[j])
            if dis<minDis:
                set = j
                minDis = dis
        kpCluster[set].append(kp[i])

    pic = img
    for i in range(K):
        pic = cv2.drawKeypoints(pic,kpCluster[i],color=colors[i], flags=0)

    leftIndex = findNeaghborPoint(locations, kcenters[0])
    leftEye = [kp[leftIndex]]
    pic = cv2.drawKeypoints(pic,leftEye,color=colors[5], flags=0)
    rightIndex = findNeaghborPoint(locations, kcenters[2])
    rightEye = [kp[rightIndex]]
    pic = cv2.drawKeypoints(pic,rightEye,color=colors[5], flags=0)
    MIndex = findNeaghborPoint(locations, kcenters[1])
    mouse = [kp[MIndex]]
    pic = cv2.drawKeypoints(pic,mouse,color=colors[5], flags=0)
    imageName = image
    cv2.imshow(imageName, pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    newoutput = line.replace("resize", "output")
    cv2.imwrite(newoutput, pic)

# main funciton, list of images
partList = open('partOfList.txt', 'r')
K=3
fastFeature = 35
for line in partList:
    #img = cv2.imread(line.strip('\n'))
    oneImage = line.strip('\n')
    create_cluster_image(oneImage, fastFeature, K)
print 'end'
