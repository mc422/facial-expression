__author__ = 'chaomeng'
import cv2
import numpy as np

img = cv2.imread('jaffe/KA.HA2.30.tiff')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# SIFT brute force
sift = cv2.SIFT()
kp, des = sift.detectAndCompute(gray,None)
print des.shape
img = cv2.drawKeypoints(gray, kp)
cv2.imwrite('sift_test.jpg', img)


# # Initiate STAR detector
# orb = cv2.ORB()
# # find the keypoints with ORB
# kp = orb.detect(img,None)
# print len(kp)
# # compute the descriptors with ORB
# kp, des = orb.compute(img, kp)
# # draw only keypoints location,not size and orientation
# img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
# cv2.imwrite('orb_test.jpg', img2)
#
# # Initiate FAST object with default values
# fast = cv2.FastFeatureDetector()
# # find and draw the keypoints
# kp = fast.detect(img,None)
# img3 = cv2.drawKeypoints(img, kp, color=(255,0,0))
# # Print all default params
# # print "Threshold: ", fast.getInt('threshold')
# # print "nonmaxSuppression: ", fast.getBool('nonmaxSuppression')
# # print "neighborhood: ", fast.getInt('type')
# # print "Total Keypoints with nonmaxSuppression: ", len(kp)
# cv2.imwrite('fast_true.jpg',img3)
#
# # Disable nonmaxSuppression
# fast.setBool('nonmaxSuppression',0)
# kp = fast.detect(img,None)
#
# print "Total Keypoints without nonmaxSuppression: ", len(kp)
#
# img4 = cv2.drawKeypoints(img, kp, color=(255,0,0))
#
# cv2.imwrite('fast_false.jpg',img4)