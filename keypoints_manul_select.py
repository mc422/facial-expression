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

# function use to find the nearest keypoint around a location
def findNeaghborPoint(keypoints, location, picked_keypoints, clusterIndex):
    minIndex = -1
    minDis = sys.maxint
    for i in range(len(keypoints)):
        dis = euclidean(keypoints[i], location)
        if i in clusterIndex:
            if dis<minDis and i not in picked_keypoints:
                minIndex = i
                minDis = dis
    return minIndex

# funciton find three feature points in eye cluster
def gather_eye_locations(keyLocation, center):
    middle = center.astype(int)
    std = np.std(keyLocation, axis=0)[0]
    left = array([center[0]-std, center[1]]).astype(int)
    right = array([center[0]+std, center[1]]).astype(int)
    return [left, middle, right]

# funciton: find nose and mouth feature points in mouse_nose cluster
def gather_mouth_locations(keyLocation, center, noseThick=3, mouthThick=3):
    std = np.std(keyLocation, axis=0)
    # cluster X and Y stand deviation
    stdX = std[0]
    stdY = std[1]
    mouthY = center[1]+stdY*2/3
    noseY = center[1]-stdY*3/2
    # nose location
    nose = array([center[0], noseY]).astype(int)
    noseLeft = array([center[0]-stdX*3/4, noseY+noseThick]).astype(int)
    noseRight = array([center[0]+stdX*3/4, noseY+noseThick]).astype(int)
    # mouth location
    mouth = array([center[0], mouthY]).astype(int)
    mouthFarLeft = array([center[0]-stdX*2, mouthY]).astype(int)
    mouthFarRight = array([center[0]+stdX*2, mouthY]).astype(int)
    mouthLeftUp = array([center[0]-stdX, mouthY-mouthThick]).astype(int)
    mouthLeftDown = array([center[0]-stdX, mouthY+mouthThick]).astype(int)
    mouthMiddleUp = array([center[0], mouthY-mouthThick]).astype(int)
    mouthMiddleDown = array([center[0], mouthY+mouthThick]).astype(int)
    mouthRightUp = array([center[0]+stdX, mouthY-mouthThick]).astype(int)
    mouthRightDown = array([center[0]+stdX, mouthY+mouthThick]).astype(int)
    allMouth = [mouthFarLeft, mouthLeftUp, mouthLeftDown, mouthMiddleUp, mouthMiddleDown, mouthRightUp, mouthRightDown, mouthFarRight]
    return [nose, noseLeft, noseRight] , allMouth

# draw circles in image to indicate these facial parts locator
def draw_facial_part_locations(pic,leftPoint, nosePoint, mouthPoint, rightPoint):
    for j in range(len(leftPoint)):
        cv2.circle(pic, (leftPoint[j][0],leftPoint[j][1]), 3, color=colors[5], thickness=4)
    for j in range(len(nosePoint)):
        cv2.circle(pic,(nosePoint[j][0],nosePoint[j][1]), 3, color=colors[5], thickness=4)
    for j in range(len(mouthPoint)):
        cv2.circle(pic,(mouthPoint[j][0],mouthPoint[j][1]), 3, color=colors[5], thickness=4)
    for j in range(len(rightPoint)):
        cv2.circle(pic,(rightPoint[j][0],rightPoint[j][1]), 3, color=colors[5], thickness=4)

def get_keypoints_picked(selected_loc, keypoints, clusterIndex):
    picked_keypoints = []
    for i in range(len(selected_loc)):
        kpIndex = findNeaghborPoint(keypoints, selected_loc[i], picked_keypoints, clusterIndex)
        picked_keypoints.append(kpIndex)
    return picked_keypoints

def draw_keyPoints(pic, kp, keyPoints):
    pointSet = []
    for i in range(len(keyPoints)):
        pointSet.append(kp[keyPoints[i]])
    pic = cv2.drawKeypoints(pic,pointSet,color=colors[5], flags=0)
    return pic

def draw_keypoints_picked(pic, kp, leftKeyPoints, rightKeyPoints, noseKeyPoints, mouthKeyPoints):
    pic = draw_keyPoints(pic, kp, leftKeyPoints)
    pic = draw_keyPoints(pic, kp, rightKeyPoints)
    pic = draw_keyPoints(pic, kp, noseKeyPoints)
    pic = draw_keyPoints(pic, kp, mouthKeyPoints)
    return pic


def fastFeature_kp(image, featureValue):
    img = cv2.imread(image)
    fast = cv2.FastFeatureDetector(featureValue)
    orb = cv2.ORB(180)
    kp = fast.detect(img,None)
    kp, des = orb.compute(img, kp)
    return kp, des

def ORBfeature_kp(image, featureValue):
    img = cv2.imread(image)
    orb = cv2.ORB(featureValue)
    kp = orb.detect(img,None)
    kp, des = orb.compute(img, kp)
    return kp, des

# function use to extract keypoint and descriptor from image
# @param: image           image we can analysis
# @apram: featureValue    parameter used in keypoint detection function
# @param: choice          1 for fastFeature, 2 for ORB feature
# @return:                keypoint set and descriptor set
def ky_desc_computer(image, featureValue, choice):
    if choice==1:
        return fastFeature_kp(image, featureValue)
    if choice==2:
        return ORBfeature_kp(image, featureValue)

def create_cluster_image(image, featureValue, K, choice):
    img = cv2.imread(image)
    kp, des = ky_desc_computer(image, featureValue, choice)
    print len(kp)

    # build keypoints location array for cluster
    locations = np.empty((len(kp),2))
    for i in range(len(kp)):
        loc = array((int(kp[i].pt[0]), int(kp[i].pt[1])))
        locations[i]=loc

    size = len(des)/K
    #centers = array((locations[size], locations[2*size], locations[3*size], locations[4*size], locations[5*size], locations[6*size-1]))
    #centers = array((locations[0], locations[2*size], locations[3*size], locations[4*size-1]))
    kcenters, distortion  = kmeans(locations, K)
    kcenters = kcenters[kcenters[:,0].argsort()]

    # cluster index: 0: left eye, 1 mouth and nose, 2: right eye
    kpCluster = {i: [] for i in range(K)}
    clusterLoc = {i: [] for i in range(K)}
    kpClusterIndex = {i: [] for i in range(K)}
    for i in range(len(kp)):
        set = 0
        minDis = sys.maxint
        for j in range(K):
            dis = euclidean(locations[i], kcenters[j])
            if dis<minDis:
                set = j
                minDis = dis
        kpClusterIndex[set].append(i)
        kpCluster[set].append(kp[i])
        clusterLoc[set].append(locations[i])

    pic = img
    for i in range(K):
        pic = cv2.drawKeypoints(pic,kpCluster[i],color=colors[i], flags=0)

    leftPoint = gather_eye_locations(clusterLoc[0], kcenters[0])
    nosePoint, mouthPoint = gather_mouth_locations(clusterLoc[1], kcenters[1])
    rightPoint = gather_eye_locations(clusterLoc[2], kcenters[2])
    #draw_facial_part_locations(pic,leftPoint, nosePoint, mouthPoint, rightPoint)

    leftKeyPoints = get_keypoints_picked(leftPoint, locations, kpClusterIndex[0])
    rightKeyPoints = get_keypoints_picked(rightPoint, locations, kpClusterIndex[2])
    noseKeyPoints = get_keypoints_picked(nosePoint, locations, kpClusterIndex[1])
    mouthKeyPoints = get_keypoints_picked(mouthPoint, locations, kpClusterIndex[1])
    pic = draw_keypoints_picked(pic, kp, leftKeyPoints, rightKeyPoints, noseKeyPoints, mouthKeyPoints)

    for i in range(len(noseKeyPoints)):
        if noseKeyPoints[i] not in kpClusterIndex[1]:
            print 'error: ' , noseKeyPoints[i]
    for i in range(len(mouthKeyPoints)):
        if mouthKeyPoints[i] not in kpClusterIndex[1]:
            print 'error: ' , mouthKeyPoints[i]

    allFeature = array([])
    zeroFeature = np.zeros(32)
    for i in range(len(leftKeyPoints)):
        if leftKeyPoints[i] != -1:
            allFeature = np.append(allFeature, des[leftKeyPoints[i]])
        else:
            allFeature = np.append(allFeature, zeroFeature)
    for i in range(len(rightKeyPoints)):
        if rightKeyPoints[i] != -1:
            allFeature = np.append(allFeature, des[rightKeyPoints[i]])
        else:
            allFeature = np.append(allFeature, zeroFeature)
    for i in range(len(noseKeyPoints)):
        if noseKeyPoints[i] != -1:
            allFeature = np.append(allFeature, des[noseKeyPoints[i]])
        else:
            allFeature = np.append(allFeature, zeroFeature)
    for i in range(len(mouthKeyPoints)):
        if mouthKeyPoints[i] != -1:
            allFeature = np.append(allFeature, des[mouthKeyPoints[i]])
        else:
            allFeature = np.append(allFeature, zeroFeature)

    imageName = image
    newoutput = imageName.replace("resize", "output")
    cv2.imwrite(newoutput, pic)

    return allFeature
    # leftEye = gather_eye_feature(locations, clusterLoc[0], kcenters[0], kp)
    # pic = cv2.drawKeypoints(pic,leftEye,color=colors[5], flags=0)
    # leftEye = gather_eye_feature(locations, clusterLoc[2], kcenters[2], kp)
    # pic = cv2.drawKeypoints(pic,leftEye,color=colors[5], flags=0)

    # # look for mid point in each cluster: left eye
    # leftIndex0 = findNeaghborPoint(locations, kcenters[0])
    #
    # leftEye = [kp[leftIndex0]]
    # pic = cv2.drawKeypoints(pic,leftEye,color=colors[5], flags=0)
    #
    #
    # rightIndex = findNeaghborPoint(locations, kcenters[2])
    # rightEye = [kp[rightIndex]]
    # pic = cv2.drawKeypoints(pic,rightEye,color=colors[5], flags=0)
    #
    #
    # MIndex = findNeaghborPoint(locations, kcenters[1])
    # mouse = [kp[MIndex]]
    # pic = cv2.drawKeypoints(pic,mouse,color=colors[5], flags=0)

# main funciton, list of images
partList = open('partOfList.txt', 'r')
happyList = open('HAPPY.txt', 'r')
dispointList = open('DISP.txt', 'r')
angryList = open('ANGRY.txt', 'r')
K=3
fastFeature = 35
# choice of keypoint extraction method, 1 for fastfeature, 2 for ORB
choice = 1
for line in angryList:
    #img = cv2.imread(line.strip('\n'))
    oneImage = line.strip('\n')
    featureVector = create_cluster_image(oneImage, fastFeature, K, choice)
    print featureVector.shape
print 'end'
