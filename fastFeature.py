__author__ = 'chaomeng'
import cv2
import numpy as np

img = cv2.imread('jaffe/KA.HA2.30.tiff')
img = img[50:240, 56:200]
fast = cv2.FastFeatureDetector(50)
kp = fast.detect(img,None)
img1 = cv2.drawKeypoints(img, kp, color=(255,0,0))
cv2.imshow('test1',img1)

orb = cv2.ORB(180)
#kp = orb.detect(img,None)
newkp, des = orb.compute(img, kp)
img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
cv2.imshow('test2',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()